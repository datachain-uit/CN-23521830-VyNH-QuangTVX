"""Audit actual problem type codes and score quality; no recoding or imputation."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import duckdb


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_audit(input_root=None, output_root=None):
    root = Path(input_root or Path(__file__).resolve().parent / 'input').resolve()
    out = Path(output_root or root.parent / 'eda_problem_types_and_scores').resolve()
    tables = out / 'tables'; tables.mkdir(parents=True, exist_ok=True)
    scratch = out / '_scratch'; scratch.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET memory_limit='2GB'")
    con.execute(f'SET temp_directory={qs(scratch)}')
    con.execute('SET preserve_insertion_order=false')
    inputs, queries = [], []
    started = time.monotonic()

    def view(name, relative):
        path = root / relative
        for f in sorted(path.parent.glob('*.parquet')):
            inputs.append({'path': str(f.relative_to(root)), 'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns})
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({qs(path)})')

    def save(name, sql):
        tick = time.monotonic()
        target = tables / f'{name}.parquet'
        con.execute(f'COPY ({sql}) TO {qs(target)} (FORMAT PARQUET, COMPRESSION ZSTD)')
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({qs(target)})')
        cur = con.execute(f'SELECT * FROM {name}')
        rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
        queries.append({'name': name, 'sql': sql, 'elapsed_seconds': round(time.monotonic()-tick, 3)})
        print(name, len(rows), 'rows', queries[-1]['elapsed_seconds'], 'seconds', flush=True)
        return rows

    view('raw_problem', 'moocubex_parquet/tables/entities/problem/*.parquet')
    view('problem_catalog', 'moocubex_model/processed/dim_problem/*.parquet')
    view('behavior', 'moocubex_parquet/tables/relations/user_problem/*.parquet')
    view('lo_problem', 'moocubex_model/processed/bridge_lo_problem/*.parquet')
    assert con.execute('SELECT count(*)=count(DISTINCT problem_id) FROM problem_catalog').fetchone()[0]
    con.execute("""CREATE TEMP TABLE chapter_map AS SELECT problem_id
        FROM lo_problem GROUP BY 1 HAVING count(DISTINCT lo_key)=1""")
    catalog_rows = save('catalog_type_text', """SELECT type,typetext,count(*) n_problems,
        count(*) FILTER(WHERE score IS NULL) max_score_null,
        count(*) FILTER(WHERE score=0) max_score_zero,
        count(*) FILTER(WHERE score<0) max_score_negative,
        min(score) min_max_score,max(score) max_max_score
        FROM raw_problem GROUP BY 1,2 ORDER BY n_problems DESC""")
    code_rows = save('catalog_type_summary', """WITH ranked AS (
        SELECT *,row_number() OVER(PARTITION BY type ORDER BY n_problems DESC,typetext) pos
        FROM catalog_type_text)
        SELECT type,sum(n_problems)::BIGINT n_problems,
          max(typetext) FILTER(WHERE pos=1) most_common_typetext,
          count(*) n_typetext_values,sum(n_problems) FILTER(WHERE pos<>1)::BIGINT nonmodal_label_problems,
          sum(max_score_null)::BIGINT max_score_null,sum(max_score_zero)::BIGINT max_score_zero
        FROM ranked GROUP BY 1 ORDER BY type""")
    direct = 'coalesce(isfinite(u.score) AND u.score>=0 AND d.problem_max_score>0 AND u.score<=d.problem_max_score+1e-9,false)'
    fallback = 'coalesce(u.is_correct IN (0,1) AND d.problem_max_score>0 AND isfinite(d.problem_max_score),false)'
    behavior_rows = save('behavior_by_type_and_mapping', f"""SELECT d.problem_type AS "type",d.problem_type_text typetext,
        d.problem_id IS NOT NULL catalog_problem_found,m.problem_id IS NOT NULL mapped_to_chapter,
        count(*) n_rows,count(*) FILTER(WHERE u.score IS NULL) score_null,
        count(*) FILTER(WHERE u.score<0) score_negative,
        count(*) FILTER(WHERE d.problem_max_score IS NULL OR d.problem_max_score<=0) max_score_missing_or_zero,
        count(*) FILTER(WHERE {direct}) score_direct_valid,
        count(*) FILTER(WHERE NOT({direct}) AND {fallback}) score_fallback_possible,
        count(*) FILTER(WHERE NOT({direct}) AND NOT({fallback})) score_unresolved,
        count(*) FILTER(WHERE d.problem_max_score>0 AND isfinite(u.score) AND u.score>d.problem_max_score+1e-9) score_exceeds_max,
        count(*) FILTER(WHERE {direct} AND u.score>1e-9 AND u.score<d.problem_max_score-1e-9) partial_credit_rows,
        count(*) FILTER(WHERE {direct} AND u.score>1e-9 AND u.score<d.problem_max_score-1e-9 AND u.is_correct=0) partial_credit_correctness_zero,
        min(u.attempts) attempts_min,max(u.attempts) attempts_max,
        count(*) FILTER(WHERE u.attempts>1) repeated_attempt_rows
        FROM behavior u LEFT JOIN problem_catalog d USING(problem_id) LEFT JOIN chapter_map m USING(problem_id)
        GROUP BY 1,2,3,4 ORDER BY n_rows DESC""")
    count_columns = ['n_rows','score_null','score_negative','max_score_missing_or_zero',
                     'score_direct_valid','score_fallback_possible','score_unresolved','score_exceeds_max',
                     'partial_credit_rows','partial_credit_correctness_zero','repeated_attempt_rows']
    sums = ','.join(f'sum({k})::BIGINT {k}' for k in count_columns)
    scoped = []
    for scope, condition in [('ALL_SOURCE','TRUE'),('MAPPED_CHAPTER','mapped_to_chapter')]:
        scoped.append(f"SELECT {qs(scope)} AS \"scope\",type,{sums},min(attempts_min) attempts_min,max(attempts_max) attempts_max FROM behavior_by_type_and_mapping WHERE {condition} GROUP BY type")
    by_type = save('behavior_by_type', ' UNION ALL '.join(scoped))
    totals = save('scope_totals', f"SELECT scope,{sums} FROM behavior_by_type GROUP BY scope ORDER BY scope")
    by_scope = {r['scope']: r for r in totals}
    prior = json.loads((root.parent / 'eda_engagement_features_comparison/summary.json').read_text())
    old_all, old_map = prior['problem_source_quality'][0], prior['problem_mapped_quality'][0]
    checks = {
        'all_source_row_count_matches_prior_eda': by_scope['ALL_SOURCE']['n_rows']==old_all['n_rows'],
        'all_source_score_null_matches_prior_eda': by_scope['ALL_SOURCE']['score_null']==old_all['score_null'],
        'mapped_row_count_matches_prior_eda': by_scope['MAPPED_CHAPTER']['n_rows']==old_map['n_rows'],
        'mapped_score_categories_match_prior_eda': all(by_scope['MAPPED_CHAPTER'][k]==old_map[k] for k in ['score_direct_valid','score_fallback_possible','score_unresolved','score_exceeds_max']),
        'catalog_profile_covers_all_catalog_rows': sum(r['n_problems'] for r in catalog_rows)==con.execute('SELECT count(*) FROM raw_problem').fetchone()[0],
        'score_categories_partition_rows': all(r['score_direct_valid']+r['score_fallback_possible']+r['score_unresolved']==r['n_rows'] for r in behavior_rows),
    }
    summary = {'catalog_by_type_text':catalog_rows,'catalog_by_type':code_rows,'behavior_by_type':by_type,
               'scope_totals':totals,'checks':checks,
               'unknown_catalog_behavior':sum(r['n_rows'] for r in behavior_rows if not r['catalog_problem_found'])}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    assert all(checks.values()),checks
    manifest = {'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'duckdb_version':duckdb.__version__,
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':inputs,'queries':queries,
                'elapsed_seconds':round(time.monotonic()-started,3),'sampling':'none',
                'policies':{'type':'preserve raw type and typetext; modal label is diagnostic, never automatic recoding',
                            'score':'catalog score is the question maximum; behavior score is learner earned score',
                            'scope':'all supplied problem logs; mapped scope uses exactly one chapter; no enrollment-time filtering',
                            'fallback':'diagnostic only, not an approved replacement of partial earned points',
                            'classification':'no labels or aggregate engagement indices created'}}
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    fmt=lambda n:f'{int(n):,}'.replace(',','.')
    labels={1:'Một đáp án',2:'Nhiều đáp án',3:'Bình chọn',4:'Điền khuyết',5:'Tự luận/chủ quan',6:'Đúng/sai',9:'Lập trình'}
    lines=['# EDA mã loại problem và bất thường điểm','',
           'Thống kê trên toàn bộ dữ liệu đang cung cấp. Mã type thực tế không được tự thay bằng mã 0/1/2 trong mô tả paper. '
           'Cột nhãn phổ biến dưới đây mô tả phần lớn bản ghi, không khẳng định mọi bản ghi của cùng type đều nhất quán.','',
           '| type | typetext phổ biến | Diễn giải | Số problem | Số problem khác nhãn phổ biến của mã |',
           '|---|---|---|---:|---:|']
    for r in code_rows:
        lines.append(f'| {r["type"]} | {r["most_common_typetext"]} | {labels.get(r["type"],"Chưa xác định")} | {fmt(r["n_problems"])} | {fmt(r["nonmodal_label_problems"] or 0)} |')
    lines += ['',f'Tổng metadata problem: {fmt(sum(r["n_problems"] for r in code_rows))}. '
              f'Có {fmt(sum(r["nonmodal_label_problems"] or 0 for r in code_rows))} bản ghi dùng typetext khác nhãn phổ biến của mã. '
              'Đây là các ứng viên bất nhất, chưa kết luận nguyên nhân và không tự sửa.','',
              '## Log người học nối được duy nhất về chương','',
              '| type | Số log | score null | score âm | Thiếu/0 điểm tối đa | score vượt tối đa | Điểm từng phần |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted((r for r in by_type if r['scope']=='MAPPED_CHAPTER'),key=lambda r:(r['type'] is None,r['type'] or 0)):
        lines.append('| '+str(r['type'])+' | '+' | '.join(fmt(r[k]) for k in ['n_rows','score_null','score_negative','max_score_missing_or_zero','score_exceeds_max','partial_credit_rows'])+' |')
    lines += ['', '## Giới hạn và hệ quả xử lý','',
              '- Không đồng nhất problem.type với một trong ba nhóm khái quát trong paper. Giữ cả type và typetext; kiểm tra các cặp khác thường theo ngữ cảnh.','- Phân biệt score trong entities/problem (điểm tối đa theo catalog) và score trong user-problem (điểm người học).',
              '- Thiếu điểm không mặc nhiên do câu tự luận. Các thống kê theo loại cần được xem trước khi chọn quy tắc.',
              '- Điểm từng phần hợp lệ có thể xuất hiện ở nhiều loại; thay bằng is_correct × max_score sẽ làm mất độ lớn điểm từng phần.',
              '- Không tự đổi score null/âm thành 0; không tự loại câu tự luận hoặc tất cả bản ghi có loại khác thường.',
              '- attempts vẫn là số tổng hợp, không cung cấp timestamp riêng cho từng lần thử.',
              f'- Có {fmt(summary["unknown_catalog_behavior"])} log không có metadata problem nối được; không suy đoán loại câu hỏi cho các log này.',
              f'- Đạt {sum(checks.values())}/{len(checks)} kiểm tra bảo toàn số liệu và đối chiếu EDA trước đó.','',
              'Nguồn tham chiếu: [MOOCCubeX course documentation](https://github.com/THU-KEG/MOOCCubeX/blob/main/docs/course-en.md). '
              'Các mã và số lượng trong báo cáo được đếm từ input thực tế.','']
    (out/'bao_cao_problem_types.md').write_text('\n'.join(lines),encoding='utf-8')
    con.close()
    print('COMPLETE',json.dumps(totals,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    run_audit()
