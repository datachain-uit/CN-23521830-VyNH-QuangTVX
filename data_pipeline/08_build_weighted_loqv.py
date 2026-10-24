"""Additional final-window LOQV with an AFELO alpha coefficient (default 0.1).

Reuses completed chapter indices; saves chapter and user-course comparison
tables separately. The final master is the LEFT JOIN anchor, including all
unobserved pairs. No sentiment inference, feature recomputation or fitting.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import duckdb

DEFAULT_ALPHA = 0.1


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def run_weighted_loqv(quality_root=None, output_root=None, memory_limit='2GB', threads=2, alpha=DEFAULT_ALPHA):
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('AFELO alpha must be finite and in (0,1].')
    normalizer = 2.0 + alpha
    raw_weights = {'COELO': 1.0, 'AFELO_proxy': alpha, 'ACELO': 1.0}
    weights = {name: value / normalizer for name, value in raw_weights.items()}
    alpha_tag = format(alpha, '.12g').replace('.', 'p')
    formula_version = f'weighted_euclidean_alpha_{alpha_tag}_v2'
    base = Path.cwd() / 'working_13-10' if (Path.cwd() / 'working_13-10').is_dir() else Path.cwd()
    source = Path(quality_root or base / 'final_quality_tables').resolve()
    out = Path(output_root or source.parent / f'weighted_loqv_alpha_{alpha_tag}').resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('Weighted outputs must be separate from source quality tables.')
    manifest_path = source / 'run_manifest.json'
    source_manifest = json.loads(manifest_path.read_text())
    assert source_manifest['finalization']['afelo_mode'] == 'sentiment', 'Run notebook 07 first.'
    source_success = json.loads((source / '_SUCCESS.json').read_text())
    assert source_success['manifest_sha256'] == sha256(manifest_path), 'Source completion marker is stale.'
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    assert all(math.isfinite(v) and v > 0 for v in weights.values())
    chapter_path = source / 'tables/chapter_quality.parquet'
    final_files = sorted((source / 'tables/user_course_final').glob('*.parquet'))
    assert chapter_path.is_file() and final_files
    input_hashes = {'source_manifest': sha256(manifest_path), 'chapter_quality': sha256(chapter_path),
        'user_course_parts': {p.name: sha256(p) for p in final_files}}
    for path in [out, out / 'tables', out / '_scratch']:
        path.mkdir(parents=True, exist_ok=True)
    success = out / '_SUCCESS.json'
    if success.exists():
        success.unlink()
    con = duckdb.connect()
    con.execute(f'SET threads={int(threads)}')
    con.execute(f'SET memory_limit={qs(memory_limit)}')
    con.execute(f'SET temp_directory={qs(out / "_scratch")}')
    con.execute("SET max_temp_directory_size='4GB'")
    con.execute('SET preserve_insertion_order=false')
    con.execute(f'CREATE VIEW chapters AS SELECT * FROM read_parquet({qs(chapter_path)})')
    con.execute(f'CREATE VIEW baseline AS SELECT * FROM read_parquet({qs(source / "tables/user_course_final/*.parquet")})')
    invalid = con.execute('''SELECT count(*) FROM chapters WHERE
        (COELO IS NOT NULL AND (NOT isfinite(COELO) OR COELO NOT BETWEEN 0 AND 1)) OR
        (AFELO_proxy IS NOT NULL AND (NOT isfinite(AFELO_proxy) OR AFELO_proxy NOT BETWEEN 0 AND 1)) OR
        (ACELO IS NOT NULL AND (NOT isfinite(ACELO) OR ACELO NOT BETWEEN 0 AND 1))''').fetchone()[0]
    assert invalid == 0, 'Source indices must be finite and in [0,1].'
    co, af, ac = (weights[k] for k in ['COELO', 'AFELO_proxy', 'ACELO'])
    complete = 'COELO IS NOT NULL AND AFELO_proxy IS NOT NULL AND ACELO IS NOT NULL'
    distance = f'CASE WHEN {complete} THEN sqrt((pow(1-COELO,2)+{alpha}*pow(1-AFELO_proxy,2)+pow(1-ACELO,2))/{normalizer}) END'
    queries = {}

    def save(name, sql):
        queries[name] = sql
        target = out / 'tables' / f'{name}.parquet'
        partial = target.with_suffix('.parquet.partial')
        con.execute(f'COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)')
        partial.replace(target)
        con.execute(f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({qs(target)})')
        print(name, 'built', flush=True)
        return target

    save('chapter_loqv_weighted', f'''SELECT c.* EXCLUDE(LOQV),LOQV LOQV_equal,
        {co}*pow(1-COELO,2) weighted_error_coelo,
        {af}*pow(1-AFELO_proxy,2) weighted_error_afelo,
        {ac}*pow(1-ACELO,2) weighted_error_acelo,
        {distance} LOQV_weighted,
        {co} weight_coelo,{af} weight_afelo,{ac} weight_acelo,{alpha} afelo_alpha,
        {qs(formula_version)} weighted_formula_version FROM chapters c''')
    con.execute('''CREATE TEMP TABLE weighted_aggregates AS SELECT user_id,course_id,
        count(*) n_chapters_observed_weighted,count(LOQV_weighted) n_chapters_scored_weighted,
        CASE WHEN count(LOQV_weighted)=count(*) THEN avg(LOQV_weighted) END q_weighted
        FROM chapter_loqv_weighted GROUP BY 1,2''')
    save('user_course_loqv_weighted', f'''WITH joined AS (
        SELECT b.* EXCLUDE(chapter_details,q,y,label_status,n_chapters_scored),
            b.q q_equal,b.y y_equal,b.label_status label_status_equal,
            b.n_chapters_scored n_chapters_scored_equal,
            coalesce(w.n_chapters_scored_weighted,0) n_chapters_scored_weighted,w.q_weighted
        FROM baseline b LEFT JOIN weighted_aggregates w USING(user_id,course_id))
        SELECT *,CASE WHEN q_weighted<0.2 THEN 'E' WHEN q_weighted<0.4 THEN 'G'
                     WHEN q_weighted>=0.4 THEN 'A' END y_weighted,
            CASE WHEN n_chapters_observed=0 THEN 'NO_OBSERVED_CHAPTER'
                 WHEN q_weighted IS NULL THEN 'INCOMPLETE_OBSERVED_CHAPTERS'
                 ELSE 'LABELED_OBSERVED_CHAPTERS' END label_status_weighted,
            q_weighted-q_equal q_change,
            {co} weight_coelo,{af} weight_afelo,{ac} weight_acelo,{alpha} afelo_alpha,
            {qs(formula_version)} weighted_formula_version FROM joined''')

    def records(sql):
        cursor = con.execute(sql)
        names = [c[0] for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    summary = {
        'alpha': alpha, 'raw_weights': raw_weights, 'weights': weights,
        'weighted_formula_version': formula_version,
        'chapter_rows': con.execute('SELECT count(*) FROM chapter_loqv_weighted').fetchone()[0],
        'chapter_scored': con.execute('SELECT count(LOQV_weighted) FROM chapter_loqv_weighted').fetchone()[0],
        'user_course_rows': con.execute('SELECT count(*) FROM user_course_loqv_weighted').fetchone()[0],
        'labeled_user_courses': con.execute('SELECT count(q_weighted) FROM user_course_loqv_weighted').fetchone()[0],
        'label_status': records('SELECT label_status_weighted,count(*) n_rows FROM user_course_loqv_weighted GROUP BY 1 ORDER BY n_rows DESC'),
        'class_distribution': records('''WITH classes(class_label) AS (VALUES ('E'),('G'),('A'))
            SELECT class_label,count(*) FILTER(WHERE y_equal=class_label) n_equal,
                count(*) FILTER(WHERE y_weighted=class_label) n_weighted
            FROM classes CROSS JOIN user_course_loqv_weighted WHERE q_weighted IS NOT NULL GROUP BY 1
            ORDER BY CASE class_label WHEN 'E' THEN 0 WHEN 'G' THEN 1 ELSE 2 END'''),
        'class_by_split': records('SELECT split,y_weighted,count(*) n_rows FROM user_course_loqv_weighted WHERE y_weighted IS NOT NULL GROUP BY 1,2 ORDER BY 1,2'),
        'transitions': records('SELECT y_equal,y_weighted,count(*) n_rows FROM user_course_loqv_weighted WHERE y_weighted IS NOT NULL GROUP BY 1,2 ORDER BY 1,2'),
        'q_distribution': records('''SELECT min(q_equal) min_q_equal,min(q_weighted) min_q_weighted,
            avg(q_equal) mean_q_equal,avg(q_weighted) mean_q_weighted,
            quantile_cont(q_weighted,[0.01,0.05,0.25,0.5,0.75,0.95,0.99]) weighted_quantiles,
            count(*) FILTER(WHERE y_equal<>y_weighted) n_changed_classes
            FROM user_course_loqv_weighted WHERE q_weighted IS NOT NULL''')[0],
    }
    fixture = con.execute(f'''WITH anchors(COELO,AFELO_proxy,ACELO) AS
        (VALUES (1.0,1.0,1.0),(0.0,0.0,0.0),(1.0,0.0,1.0),(0.0,1.0,1.0),(1.0,NULL,1.0))
        SELECT {distance} FROM anchors''').fetchall()
    expected = [0,1,math.sqrt(alpha/(2+alpha)),math.sqrt(1/(2+alpha)),None]
    checks = {
        'weighted_distance_anchor_vectors': all((x[0] is None and y is None) or (x[0] is not None and y is not None and abs(x[0]-y)<1e-12) for x,y in zip(fixture,expected)),
        'chapter_rows_preserved_and_unique': con.execute('SELECT count(*)=(SELECT count(*) FROM chapters) AND count(*)=count(DISTINCT (user_id,course_id,lo_key)) FROM chapter_loqv_weighted').fetchone()[0],
        'chapter_keys_preserved': con.execute('SELECT count(*)=0 FROM chapters ANTI JOIN chapter_loqv_weighted USING(user_id,course_id,lo_key)').fetchone()[0],
        'master_rows_preserved_and_unique': con.execute('SELECT count(*)=(SELECT count(*) FROM baseline) AND count(*)=count(DISTINCT (user_id,course_id)) FROM user_course_loqv_weighted').fetchone()[0],
        'master_keys_preserved': con.execute('SELECT count(*)=0 FROM baseline ANTI JOIN user_course_loqv_weighted USING(user_id,course_id)').fetchone()[0],
        'null_chapter_coverage_unchanged': con.execute('SELECT count(*)=0 FROM chapter_loqv_weighted WHERE (LOQV_equal IS NULL)<>(LOQV_weighted IS NULL)').fetchone()[0],
        'null_user_course_coverage_unchanged': con.execute('SELECT count(*)=0 FROM user_course_loqv_weighted WHERE (q_equal IS NULL)<>(q_weighted IS NULL) OR n_chapters_scored_equal<>n_chapters_scored_weighted').fetchone()[0],
        'distances_and_q_in_unit_interval': con.execute('SELECT (SELECT count(*) FROM chapter_loqv_weighted WHERE NOT isfinite(LOQV_weighted) OR LOQV_weighted NOT BETWEEN 0 AND 1)=0 AND (SELECT count(*) FROM user_course_loqv_weighted WHERE NOT isfinite(q_weighted) OR q_weighted NOT BETWEEN 0 AND 1)=0').fetchone()[0],
        'missing_chapter_has_no_q_or_label': con.execute('SELECT count(*)=0 FROM user_course_loqv_weighted WHERE (n_chapters_observed=0 OR n_chapters_scored_weighted<n_chapters_observed) AND (q_weighted IS NOT NULL OR y_weighted IS NOT NULL)').fetchone()[0],
        'q_is_mean_of_chapter_distances': con.execute('''SELECT count(*)=0 FROM user_course_loqv_weighted f JOIN weighted_aggregates a USING(user_id,course_id)
            WHERE (f.q_weighted IS NULL)<>(a.q_weighted IS NULL) OR abs(f.q_weighted-a.q_weighted)>1e-12 OR f.n_chapters_observed<>a.n_chapters_observed_weighted''').fetchone()[0],
        'classification_thresholds_unchanged': con.execute("SELECT count(*)=0 FROM user_course_loqv_weighted WHERE y_weighted IS DISTINCT FROM CASE WHEN q_weighted<0.2 THEN 'E' WHEN q_weighted<0.4 THEN 'G' WHEN q_weighted>=0.4 THEN 'A' END").fetchone()[0],
        'original_q_and_y_preserved': con.execute('''SELECT count(*)=0 FROM baseline b JOIN user_course_loqv_weighted w USING(user_id,course_id)
            WHERE b.q IS DISTINCT FROM w.q_equal OR b.y IS DISTINCT FROM w.y_equal OR b.label_status IS DISTINCT FROM w.label_status_equal''').fetchone()[0],
        'recorded_alpha_and_normalized_weights_correct': con.execute(f'''SELECT count(*)=0 FROM user_course_loqv_weighted
            WHERE afelo_alpha<>{alpha} OR abs(weight_coelo+weight_afelo+weight_acelo-1)>1e-12
            OR abs(weight_afelo/weight_coelo-afelo_alpha)>1e-12''').fetchone()[0],
        'normalized_contributions_match_squared_distance': con.execute('''SELECT count(*)=0 FROM chapter_loqv_weighted
            WHERE abs(LOQV_weighted*LOQV_weighted-weighted_error_coelo-weighted_error_afelo-weighted_error_acelo)>1e-12''').fetchone()[0],
    }
    checks['source_files_unchanged'] = input_hashes == {
        'source_manifest': sha256(manifest_path), 'chapter_quality': sha256(chapter_path),
        'user_course_parts': {p.name: sha256(p) for p in final_files}}
    assert all(checks.values()), checks
    summary['validation'] = checks
    manifest = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_root': str(source), 'output_root': str(out), 'input_sha256': input_hashes,
        'engine_source_sha256': globals().get('WEIGHTED_SOURCE_SHA256') or sha256(__file__),
        'duckdb_version': duckdb.__version__, 'alpha': alpha, 'raw_weights': raw_weights,
        'weights': weights, 'formula_version': formula_version,
        'formula': f'sqrt(((1-COELO)^2 + {alpha}*(1-AFELO_proxy)^2 + (1-ACELO)^2)/{normalizer})',
        'formula_family': 'Normalized alpha-weighted Euclidean distance; alpha=1 recovers Review equation (1)',
        'aggregation': 'mean chapter LOQV only if every observed chapter has all three indices',
        'class_thresholds': {'E': 'q<0.2','G': '0.2<=q<0.4','A': 'q>=0.4'},
        'unobserved': 'preserve master row; q/y remain null',
        'parameter_origin': 'explicit user alpha; no fitting or class-balancing threshold selection',
        'source_sentiment': source_manifest['finalization'].get('sentiment_model'),
        'role': 'Final-window target/audit; never use as phase prediction features', 'sql': queries}
    dump_json(out / 'summary.json', summary)
    dump_json(out / 'validation.json', checks)
    dump_json(out / 'run_manifest.json', manifest)
    write_report(out, summary)
    dump_json(success, {'manifest_sha256': sha256(out / 'run_manifest.json')})
    con.close()
    print('WEIGHTED_COMPLETE', json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    return summary


def write_report(out, summary):
    fmt = lambda n: f'{int(n):,}'.replace(',', '.')
    alpha = summary['alpha']; weights = summary['weights']
    lines = [f'# LOQV với alpha AFELO = {alpha:g}', '',
        f'`LOQV_weighted = sqrt(((1-COELO)^2 + {alpha:g}*(1-AFELO_proxy)^2 + (1-ACELO)^2)/{2+alpha:g})`.', '',
        f'Alpha={alpha:g} là hệ số tương đối của bình phương sai lệch AFELO khi COELO và ACELO có hệ số 1. '
        f'Sau chuẩn hóa, trọng số COELO/AFELO/ACELO là {weights["COELO"]:.12g} / '
        f'{weights["AFELO_proxy"]:.12g} / {weights["ACELO"]:.12g}. '
        'Alpha không phải trọng số AFELO sau chuẩn hóa.', '',
        'Mẫu số 2+alpha giúp giữ thang [0,1]. Alpha=1 khôi phục công thức (1), Mục 2.3 của Review; '
        'alpha<1 là biến thể do người dùng chỉ định. Giá trị nhỏ hơn nghĩa là gần vector lý tưởng (1,1,1) hơn.', '',
        'q_weighted là trung bình LOQV_weighted của mọi chương quan sát, chỉ tính khi tất cả chương đó đủ ba chỉ số. '
        'E nếu q<0,2; G nếu 0,2≤q<0,4; A nếu q≥0,4. Không tính khoảng cách từ vector chỉ số trung bình khóa.', '',
        f'- Chương quan sát: {fmt(summary["chapter_rows"])}; đủ điều kiện LOQV: {fmt(summary["chapter_scored"])}.',
        f'- Giữ đủ {fmt(summary["user_course_rows"])} user–course; {fmt(summary["labeled_user_courses"])} cặp có nhãn.', '',
        '| Lớp | Trọng số đều trước đây | Trọng số mới |', '|---|---:|---:|']
    lines += [f'| {r["class_label"]} | {fmt(r["n_equal"])} | {fmt(r["n_weighted"])} |' for r in summary['class_distribution']]
    lines += ['', f'Số cặp đổi lớp: {fmt(summary["q_distribution"]["n_changed_classes"])}.', '',
        '## Các bảng', '',
        '- `tables/chapter_loqv_weighted.parquet`: user–course–lo_key; giữ toàn bộ feature/chỉ số chương, LOQV_equal, '
        'ba weighted_error_*, LOQV_weighted, afelo_alpha và trọng số chuẩn hóa.',
        '- `tables/user_course_loqv_weighted.parquet`: user–course; giữ các feature scalar/tổng hợp từ master, '
        'q_equal/y_equal, q_weighted/y_weighted, số chương và trạng thái. '
        'chapter_details đầy đủ còn ở master nguồn; có thể LEFT JOIN bằng user_id/course_id, '
        'hoặc lấy chi tiết từ bảng chương mới.', '',
        '## Chuyển lớp', '', '| Lớp cũ | Lớp mới | Số cặp |', '|---|---|---:|']
    lines += [f'| {r["y_equal"]} | {r["y_weighted"]} | {fmt(r["n_rows"])} |' for r in summary['transitions']]
    lines += ['', '## Độ phủ và giới hạn', '',
        'Chương chưa quan sát không được tạo thêm. Các cặp thiếu chương quan sát hoặc thiếu chỉ số vẫn giữ q/y null. '
        'Ba chỉ số thành phần, sentiment và dữ liệu nguồn được kế thừa từ notebook 07. '
        'Alpha mới do người dùng chỉ định, không fit từ dev/test và không điều chỉnh ngưỡng để cân bằng lớp.', '',
        'AFELO=0 ở chương không có comment/reply vẫn theo quy ước hiện tại. Với COELO=ACELO=1 và AFELO=0, '
        f'LOQV mới là sqrt(alpha/(2+alpha))≈{math.sqrt(alpha/(2+alpha)):.6f} thay vì sqrt(1/3)≈0,577. '
        'Điều chỉnh trọng số giảm ảnh hưởng của AFELO nhưng không chứng minh độ phủ sentiment hoặc tính hợp lệ của proxy đã cải thiện.', '',
        'Các bảng là nhãn/kiểm toán cuối kỳ; feature P1–P4 cần tính riêng đến cutoff. '
        'Split course và giới hạn chồng thời gian giữ như báo cáo 06/07.', '',
        '## Kiểm tra', '', f'Đạt {sum(summary["validation"].values())}/{len(summary["validation"])} kiểm tra: '
        'vector neo, bảo toàn khóa/dòng, độ phủ null, ngưỡng, cách tổng hợp và checksum nguồn. '
        'Nguồn, SQL, trọng số và checksum nằm trong run_manifest.json.', '']
    (Path(out) / 'bao_cao_weighted_loqv.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-root')
    parser.add_argument('--output-root')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()
    run_weighted_loqv(args.quality_root, args.output_root, alpha=args.alpha)
