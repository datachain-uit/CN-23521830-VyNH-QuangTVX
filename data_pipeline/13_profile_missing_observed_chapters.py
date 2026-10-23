"""Profile missing/zero cells in observed chapter tables and raw phase features.

No imputation, model fit, label change, or source-data mutation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

AS_OF = '2026-10-19'
CORE = ['TS','RV','AV','IF_raw','IF','KL','CFA','mastery_attempted_partial',
        'problem_participation','comment_participation','reply_participation',
        'response_received_rate','sentiment_valence','afe_text_coverage',
        'COELO','AFELO_proxy','ACELO','LOQV']


def qs(x):
    return "'" + str(x).replace("'", "''") + "'"


def qi(x):
    return '"' + str(x).replace('"', '""') + '"'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')


def records(con,sql):
    x=con.execute(sql)
    names=[r[0] for r in x.description]
    return [dict(zip(names,r)) for r in x.fetchall()]


def mdtable(headers,rows):
    def cell(x):
        if x is None:return '—'
        if isinstance(x,float):return f'{x:.2f}'.replace('.',',')
        if isinstance(x,int):return f'{x:,}'.replace(',','.')
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
                     +['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])


def profile(con,view_name,cohort):
    schema=con.execute(f'DESCRIBE {view_name}').fetchall()
    fields=['count(*) n_rows']
    for i,(name,dtype,*_) in enumerate(schema):
        col=qi(name)
        numeric=dtype.startswith(('BIGINT','INTEGER','DOUBLE','FLOAT','DECIMAL','HUGEINT','SMALLINT','TINYINT'))
        floating=dtype.startswith(('DOUBLE','FLOAT'))
        fields.append(f'count(*) FILTER(WHERE {col} IS NULL) missing_{i}')
        fields.append((f'count(*) FILTER(WHERE {col}=0)' if numeric else 'NULL::BIGINT')+f' zero_{i}')
        fields.append((f'count(*) FILTER(WHERE {col} IS NOT NULL AND NOT isfinite({col}))'
                       if floating else '0::BIGINT')+f' nonfinite_{i}')
        fields.append((f"count(*) FILTER(WHERE {col} IS NOT NULL AND trim({col})='')"
                       if dtype=='VARCHAR' else '0::BIGINT')+f' empty_{i}')
    row=records(con,f'SELECT {",".join(fields)} FROM {view_name}')[0]
    result=[]
    for i,(name,dtype,*_) in enumerate(schema):
        n=row['n_rows'];missing=row[f'missing_{i}'];zero=row[f'zero_{i}']
        result.append(dict(cohort=cohort,variable=name,dtype=dtype,n_rows=n,
            n_null=missing,pct_null=100*missing/n,n_nonnull=n-missing,
            n_zero=zero,pct_zero=None if zero is None else 100*zero/n,
            n_nonfinite=row[f'nonfinite_{i}'],n_empty_string=row[f'empty_{i}']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root');args=p.parse_args()
    base=Path(args.root or (Path.cwd()/'working_13-10'
                           if (Path.cwd()/'working_13-10').is_dir() else Path.cwd())).resolve()
    out=base/'observed_chapter_missingness';(out/'tables').mkdir(parents=True,exist_ok=True)
    source_paths=[base/'final_quality_tables/tables/chapter_quality.parquet',
                  base/'06_build_final_quality_tables.py',base/'09_build_phase_features.py',
                  base/'phase_features/feature_columns.json',base/'phase_features/preprocessing.json']
    source_paths += [base/f'phase_features/supervised_raw/{x}.parquet' for x in ['P1','P2','P3','P4']]
    hashes={str(x.relative_to(base)):sha(x) for x in source_paths}
    con=duckdb.connect();con.execute("SET memory_limit='2GB'");con.execute('SET threads=2')
    con.execute(f'CREATE VIEW chapters AS SELECT * FROM read_parquet({qs(source_paths[0])})')
    chapter_profile=profile(con,'chapters','observed_final_chapters')
    population=records(con,'''SELECT count(*) n_rows,count(DISTINCT(user_id,course_id)) n_pairs,
        count(DISTINCT(course_id,lo_key)) n_unique_chapters,count(DISTINCT course_id) n_courses,
        count(*) FILTER(WHERE has_video) n_with_video_source,
        count(*) FILTER(WHERE has_problem) n_with_problem_source,
        count(*) FILTER(WHERE has_social) n_with_social_source,
        count(*) FILTER(WHERE coelo_n_components=4) n_all_four_coelo_components,
        count(*) FILTER(WHERE acelo_n_components=2) n_both_acelo_components,
        count(*) FILTER(WHERE COELO IS NOT NULL AND AFELO_proxy IS NOT NULL AND ACELO IS NOT NULL) n_three_indices_available
        FROM chapters''')[0]
    source_patterns=records(con,'''SELECT has_video,has_problem,has_social,count(*) n_rows
        FROM chapters GROUP BY 1,2,3 ORDER BY n_rows DESC''')
    reasons={
        'TS':'''CASE WHEN n_videos_available=0 THEN 'NO_VIDEO_CATALOG'
          WHEN n_videos_missing_duration>0 THEN 'INCOMPLETE_VIDEO_DURATION'
          ELSE 'DURATION_NULL_OR_NONPOSITIVE' END''',
        'KL':'''CASE WHEN n_problems_available=0 THEN 'NO_PROBLEM_CATALOG'
          WHEN n_problems_missing_score>0 THEN 'INCOMPLETE_CATALOG_MAX_SCORE'
          WHEN n_scores_unresolved>0 THEN 'UNRESOLVED_ATTEMPTED_SCORE'
          ELSE 'MAX_SCORE_NULL_OR_ZERO' END''',
        'CFA':'''CASE WHEN n_problems_available=0 THEN 'NO_PROBLEM_CATALOG'
          WHEN n_efficiency_unresolved>0 THEN 'UNRESOLVED_EFFICIENCY' ELSE 'OTHER' END''',
        'sentiment_valence':'''CASE WHEN n_sentiment_comments=0 THEN 'NO_OBSERVED_SENTIMENT_COMMENT' ELSE 'OTHER' END''',
        'response_received_rate':'''CASE WHEN n_comments_filtered=0 THEN 'NO_FILTERED_COMMENT_DENOMINATOR' ELSE 'OTHER' END''',
    }
    null_reasons=[]
    for feature,reason in reasons.items():
        null_reasons += records(con,f'''SELECT {qs(feature)} AS "variable",{reason} null_reason,count(*) n_rows
            FROM chapters WHERE {qi(feature)} IS NULL GROUP BY 1,2 ORDER BY n_rows DESC''')
    phase_profile=[]
    cfg=json.loads((base/'phase_features/feature_columns.json').read_text())
    for phase in ['P1','P2','P3','P4']:
        path=base/f'phase_features/supervised_raw/{phase}.parquet'
        con.execute(f'CREATE OR REPLACE VIEW phase_raw AS SELECT {",".join(qi(x) for x in cfg["model_feature_columns"])} FROM read_parquet({qs(path)})')
        phase_profile += profile(con,'phase_raw',phase)
    for name,data in [('chapter_missing_profile',chapter_profile),('phase_supervised_raw_missing_profile',phase_profile),
                       ('null_reasons',null_reasons),('source_presence_patterns',source_patterns)]:
        json_path=out/f'{name}.json';dump(json_path,data)
        con.execute(f'COPY (SELECT * FROM read_json_auto({qs(json_path)})) TO {qs(out/"tables"/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
    checks={
        'chapter_rows_match_845238':population['n_rows']==845238,
        'chapter_keys_unique':con.execute('SELECT count(*)=count(DISTINCT(user_id,course_id,lo_key)) FROM chapters').fetchone()[0],
        'all_84_fields_profiled':len(chapter_profile)==84,
        'profile_counts_consistent':all(r['n_null']+r['n_nonnull']==r['n_rows'] and
            (r['n_zero'] is None or r['n_zero']<=r['n_nonnull']) for r in chapter_profile+phase_profile),
        'no_nonfinite_values':all(r['n_nonfinite']==0 for r in chapter_profile+phase_profile),
        'phase_55_features_each':len(phase_profile)==4*55,
        'phase_supervised_same_164429_rows':all(r['n_rows']==164429 for r in phase_profile),
        'null_reason_partitions_match':all(sum(r['n_rows'] for r in null_reasons if r['variable']==name)==
            next(r['n_null'] for r in chapter_profile if r['variable']==name) for name in reasons),
        'source_patterns_partition_rows':sum(r['n_rows'] for r in source_patterns)==population['n_rows'],
        'source_files_unchanged':all(sha(base/p)==v for p,v in hashes.items()),
    }
    assert all(checks.values()),checks
    dump(out/'validation.json',checks)
    summary=dict(as_of=AS_OF,population=population,chapter_profile=chapter_profile,
        phase_supervised_profile=phase_profile,null_reasons=null_reasons,source_patterns=source_patterns,validation=checks)
    dump(out/'summary.json',summary)
    doc=render(summary)
    report=out/'bao_cao_missing_value.md';report.write_text(doc)
    manifest=dict(as_of=AS_OF,engine_source_sha256=sha(__file__),source_files_sha256=hashes,
        report_sha256=sha(report),production_mutation='none',imputation_applied=False,
        chapter_scope='Observed final-window user-course-chapter rows; existing count-zero and score-fallback policies remain.')
    dump(out/'run_manifest.json',manifest)
    dump(out/'_SUCCESS.json',dict(manifest_sha256=sha(out/'run_manifest.json'),report_sha256=sha(report)))
    print('PROFILE_COMPLETE',population,flush=True)
    print('CORE',json.dumps([r for r in chapter_profile if r['variable'] in CORE],ensure_ascii=False),flush=True)


def render(s):
    rows=s['chapter_profile'];lookup={r['variable']:r for r in rows};n=s['population']['n_rows']
    core=[lookup[x] for x in CORE]
    def profile_table(items):
        return mdtable(['Biến','Kiểu','NULL','% NULL','Bằng 0','% bằng 0'],
            [[r[k] for k in ['variable','dtype','n_null','pct_null','n_zero','pct_zero']] for r in items])
    parts=[
        '# Missing value trên các chapter đã quan sát',
        '**Ngày:** '+s['as_of']+'. Phân tích toàn bộ bảng, không lấy mẫu.',
        '## 1. Mẫu số và ý nghĩa trước impute',
        mdtable(['Thống kê','Số lượng'],list(s['population'].items())),
        'Bảng chính là chapter_quality cuối kỳ, một dòng user–course–chapter. '
        'Bảng này không qua median-impute hoặc standard-scale. Tuy nhiên count/thời lượng vắng nguồn đã được coalesce về 0, '
        'score đã qua quy tắc giữ điểm hợp lệ/fallback binary và chỉ số dùng mean_non_null theo pipeline 06–07. '
        'Vì vậy đây không phải raw event chưa xử lý. Zero không được tính là NULL và không chứng minh vắng hoạt động ngoài thực tế.',
        '## 2. Thành phần và chỉ số chính',
        'Tỷ lệ lấy mẫu số '+f'{n:,}'.replace(',','.')+' dòng chapter quan sát cho từng cột. Các cột có thể cùng thiếu trên một dòng, không cộng % NULL.',
        profile_table(core),
        'COELO có giá trị nếu ít nhất hai thành phần có giá trị; ACELO cần ít nhất một trong KL/CFA. '
        'Do đó chỉ số tổng có thể không NULL dù thiếu một số thành phần. '
        'AFELO_proxy có thể bằng 0 khi không có comment/reply dù sentiment_valence và response_received_rate NULL.',
        '## 3. Nguyên nhân NULL của một số biến',
        mdtable(['Biến','Nhánh nguyên nhân theo code','Dòng NULL'],
                [[r['variable'],r['null_reason'],r['n_rows']] for r in s['null_reasons']]),
        'Không có tài nguyên làm mẫu số là không áp dụng/không đo được theo catalog hiện tại, khác với số đếm hoạt động bằng 0. '
        'INCOMPLETE_CATALOG_MAX_SCORE và INCOMPLETE_VIDEO_DURATION phản ánh metadata chưa đủ; '
        'NO_OBSERVED_SENTIMENT_COMMENT không đồng nghĩa model suy luận neutral/negative.',
        '## 4. Độ phủ nguồn trong cùng chapter',
        mdtable(['Có video','Có problem','Có social','Dòng chapter'],
                [[r['has_video'],r['has_problem'],r['has_social'],r['n_rows']] for r in s['source_patterns']]),
        'Các cờ phản ánh bảng nguồn đã mapping và qua cửa sổ của pipeline, không phải đo lường hành vi ngoài snapshot.',
        '## 5. Tất cả cột có NULL',
        profile_table(sorted([r for r in rows if r['n_null']>0],key=lambda r:-r['n_null'])),
        '## 6. Toàn bộ cột không có NULL, kèm tỷ lệ 0',
        profile_table([r for r in rows if r['n_null']==0]),
        'Các biến Boolean false không được cộng vào số 0 trong bảng numeric. Chuỗi rỗng và nonfinite được lưu riêng '
        'trong JSON/Parquet; kiểm tra no_nonfinite_values xác nhận không có NaN/Inf trong các bảng đã thống kê.',
        '## 7. Đối chiếu riêng với X trước impute ở P1–P4',
        'Bảng này có mẫu số 164.429 user–course có nhãn mỗi phase, khác mẫu số 845.238 chapter ở trên. '
        'Đây là supervised_raw, đã encode danh mục và áp dụng quy tắc count=0, nhưng chưa median-impute hoặc scale. '
        'Không cộng các phase thành số cặp độc lập.',
        mdtable(['Phase','Dòng','Tuổi NULL (%)','Video mean NULL (%)','Problem correct rate NULL (%)',
                 'Score ratio NULL (%)','Sentiment NULL (%)'],
                [[p,164429]+[next(r['pct_null'] for r in s['phase_supervised_profile'] if r['cohort']==p and r['variable']==col)
                    for col in ['learner_age_at_enrollment','video_mean_speed','problem_correct_rate',
                                'problem_attempted_score_ratio','sentiment_p_positive_mean']]
                 for p in ['P1','P2','P3','P4']]),
        'Thống kê đầy đủ 55 cột mỗi phase nằm trong phase_supervised_raw_missing_profile.json/Parquet. '
        'Mã 0 của danh mục chưa thấy và mã 0 hợp lệ của gender không được tự coi là NULL.',
        '## 8. File kết quả và tái lập',
        '- [Thống kê toàn bộ](summary.json)\n'
        '- [84 cột chapter — JSON](chapter_missing_profile.json)\n'
        '- [84 cột chapter — Parquet](tables/chapter_missing_profile.parquet)\n'
        '- [55 cột X × 4 phase — JSON](phase_supervised_raw_missing_profile.json)\n'
        '- [55 cột X × 4 phase — Parquet](tables/phase_supervised_raw_missing_profile.parquet)\n'
        '- [Kiểm tra](validation.json)\n'
        '- [Manifest nguồn](run_manifest.json)\n'
        '- [Mã tái lập](../13_profile_missing_observed_chapters.py)\n'
        '- [Bảng chapter nguồn](../final_quality_tables/tables/chapter_quality.parquet)',
        'Chạy từ working_13-10 với Python có DuckDB 1.3.2: python 13_profile_missing_observed_chapters.py. '
        'Không sửa nhãn, feature hoặc chính sách missing.',
    ]
    return '\n\n'.join(parts)+'\n'


if __name__=='__main__':
    main()
