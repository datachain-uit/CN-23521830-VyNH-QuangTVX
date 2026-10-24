"""Read-only checks supporting the CQ V4 formula explanation; train/dev only.

Run with DuckDB installed. No builders, inference, or test row data are executed.
Only this audit's aggregate evidence.json is written.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    import duckdb
    evidence = {
        'date': '2026-10-23',
        'scope': 'source/config/manifest inspection and independent train/dev checks',
        'test_row_data_read': False,
        'builder_or_model_executed': False,
        'python': sys.version.split()[0],
        'duckdb': duckdb.__version__,
        'git_commit': None,
        'git_note': 'Local main has no commits; use SHA-256 and recorded manifests.',
    }
    configs = read('phase_features_v4/feature_columns.json')
    prep = read('phase_features_v4/preprocessing.json')
    dictionary = read('phase_features_v4/feature_dictionary.json')
    numeric = configs['numeric_features']
    categorical = configs['categorical_features']
    evidence['inventory'] = {
        'numeric': len(numeric), 'categorical': len(categorical),
        'learner': len(configs['model_feature_columns']),
        'selector': len(configs['selector_feature_columns']),
        'dictionary': len(dictionary),
        'dictionary_matches_allowlist': {x['name'] for x in dictionary} == set(numeric + categorical),
        'preprocessing_matches_numeric': set(prep['numeric']) == set(numeric),
        'raw_conversion': read('input/moocubex_parquet/manifest.json'),
        'dimension_bridge': read('input/moocubex_model/build_manifest.json'),
    }
    comparisons = [
        ('15_build_final_quality_v2.py', 'final_quality_v2/run_manifest.json', 'script_sha256'),
        ('16_build_phase_features_v2.py', 'phase_features_v2/run_manifest.json', 'source_sha256'),
        ('18_build_final_quality_v3.py', 'final_quality_v3/run_manifest.json', 'source_sha256'),
        ('19_build_phase_features_v3.py', 'phase_features_v3/run_manifest.json', 'source_sha256'),
        ('20_build_fullA.py', 'fullA/run_manifest.json', 'source_sha256'),
        ('21_build_final_quality_v4.py', 'final_quality_v4/run_manifest.json', 'builder_sha256'),
        ('22_build_phase_features_v4.py', 'phase_features_v4/run_manifest.json', 'builder_sha256'),
        ('cq_v4_common.py', 'final_quality_v4/run_manifest.json', 'common_sha256'),
        ('07_compute_afelo_sentiment.py', 'afelo_sentiment_cardiff/run_manifest.json', 'script_sha256'),
        ('09_build_phase_features.py', None, None),
        ('04_eda_engagement_features_comparison.py', None, None),
        ('05_build_review_time_phases.py', None, None),
        ('06_build_final_quality_tables.py', None, None),
        ('01_moocubex_raw_to_parquet.ipynb', None, None),
        ('02_build_dimensions_bridges.ipynb', None, None),
    ]
    evidence['code_versions'] = []
    for source, manifest, key in comparisons:
        actual = sha(source)
        recorded = read(manifest).get(key) if manifest else None
        evidence['code_versions'].append({
            'source': source, 'current_sha256': actual, 'manifest': manifest,
            'recorded_sha256': recorded,
            'matches_recorded': actual == recorded if recorded else None,
        })
    # The script hash differs; separately compare the SQL actually saved by its run.
    tree = ast.parse((ROOT / '15_build_final_quality_v2.py').read_text())
    checkpoint = read('final_quality_v2/step_checkpoints.json')
    target_steps = {'video_p95_reference', 'user_video_ts', 'chapter_raw',
                    'chapter_p95_reference', 'chapter_quality'}
    evidence['v2_recorded_sql_comparison'] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'save' and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value in target_steps):
            continue
        name, query = node.args[0].value, node.args[1]
        if isinstance(query, ast.Constant):
            current_sql = query.value
        elif isinstance(query, ast.JoinedStr):
            fragments = []
            for fragment in query.values:
                if isinstance(fragment, ast.Constant):
                    fragments.append(fragment.value)
                else:
                    assert ast.unparse(fragment.value) == 'qs(VERSION)'
                    fragments.append("'final-quality-v2-p95-signed-fa-2026-10-20'")
            current_sql = ''.join(fragments)
        else:
            raise ValueError(name)
        key = next(k for k in checkpoint if k.endswith('/' + name + '.parquet'))
        evidence['v2_recorded_sql_comparison'][name] = {
            'checkpoint_key': key, 'exact_sql_match': current_sql == checkpoint[key]['sql'],
            'sql_sha256': hashlib.sha256(checkpoint[key]['sql'].encode()).hexdigest(),
        }
    evidence['sentiment'] = {
        'model_lock': read('final_quality_v2/sentiment/model_lock.json'),
        'new_inference': read('final_quality_v2/sentiment/inference_config.json'),
        'cached_inference': read('afelo_sentiment_cardiff/run_manifest.json')['inference'],
        'cached_packages': read('afelo_sentiment_cardiff/run_manifest.json')['packages'],
    }
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1200MB'")
    con.execute("SET TimeZone='UTC'")
    phases = ['P1', 'P2', 'P3', 'P4']
    paths = [str(ROOT / 'phase_features_v4/supervised_raw' / p / 'train.parquet') for p in phases]
    con.execute('CREATE VIEW fit_rows AS SELECT * FROM read_parquet([' +
                ','.join(sql_string(p) for p in paths) + '])')
    row_count = con.execute('SELECT count(*) FROM fit_rows').fetchone()[0]
    medians = con.execute('SELECT ' + ','.join('median(' + x + ')' for x in numeric) + ' FROM fit_rows').fetchone()
    moments = []
    for name, median in zip(numeric, medians):
        value = f'coalesce({name}::DOUBLE,{median!r})'
        moments += [f'avg({value})', f'stddev_pop({value})']
    values = con.execute('SELECT ' + ','.join(moments) + ' FROM fit_rows').fetchone()
    evidence['preprocessing_recalculation'] = {'fit_rows': row_count, 'expected_fit_rows': prep['fit_rows'], 'numeric': {}}
    for i, name in enumerate(numeric):
        stored = prep['numeric'][name]
        mean, std = values[2*i:2*i+2]
        scale = std if math.isfinite(std) and std > 1e-12 else 1.0
        diffs = {'median': abs(medians[i] - stored['median']),
                 'mean': abs(mean - stored['mean']), 'scale': abs(scale - stored['scale'])}
        passed = all(diffs[k] <= 1e-8 * max(1.0, abs(stored[k])) for k in diffs)
        evidence['preprocessing_recalculation']['numeric'][name] = {'pass': passed, 'absolute_differences': diffs}
    transforms = []
    for phase in phases:
        for split in ['train', 'dev']:
            raw = ROOT / 'phase_features_v4/supervised_raw' / phase / (split + '.parquet')
            ready = ROOT / 'phase_features_v4/model_ready' / phase / (split + '.parquet')
            differences = []
            for name in numeric:
                p = prep['numeric'][name]
                expression = f'((coalesce(r.{name}::DOUBLE,{p["median"]!r})-{p["mean"]!r})/{p["scale"]!r})'
                differences.append(f'coalesce(abs(m.{name}-{expression})>1e-12 OR m.{name} IS NULL,true)')
            sql = f'''SELECT count(*),count(*) FILTER(WHERE r.row_id IS NULL OR m.row_id IS NULL OR
                {' OR '.join(differences)}) FROM read_parquet({sql_string(raw)}) r
                FULL JOIN read_parquet({sql_string(ready)}) m USING(row_id)'''
            rows, mismatches = con.execute(sql).fetchone()
            transforms.append({'phase': phase, 'split': split, 'rows': rows, 'mismatches': mismatches})
    evidence['raw_ready_transforms'] = transforms
    encoders = []
    static = ROOT / 'phase_features_v4/tables/course_static_metadata.parquet'
    for name, signature in [('learner', None), ('teacher_set', 'teacher_signature'),
                            ('school_set', 'school_signature'), ('field_set', 'field_signature')]:
        expected = ('SELECT DISTINCT user_id token FROM fit_rows' if signature is None else
                    f'SELECT DISTINCT {signature} token FROM read_parquet({sql_string(static)}) c '
                    f'JOIN (SELECT DISTINCT course_id FROM fit_rows) t USING(course_id) WHERE {signature} IS NOT NULL')
        encoder = ROOT / 'phase_features_v4/encoders' / (name + '.parquet')
        mismatch = con.execute(f'''SELECT count(*) FROM read_parquet({sql_string(encoder)}) e
            FULL JOIN ({expected}) t USING(token) WHERE e.token IS NULL OR t.token IS NULL''').fetchone()[0]
        count = con.execute(f'SELECT count(*) FROM read_parquet({sql_string(encoder)})').fetchone()[0]
        encoders.append({'name': name, 'tokens': count, 'mismatches': mismatch})
    evidence['train_vocabulary_checks'] = encoders
    con.close()
    evidence['validation_pass'] = (
        row_count == prep['fit_rows']
        and all(x['pass'] for x in evidence['preprocessing_recalculation']['numeric'].values())
        and all(x['mismatches'] == 0 for x in transforms + encoders)
        and all(x['exact_sql_match'] for x in evidence['v2_recorded_sql_comparison'].values())
        and evidence['inventory']['dictionary_matches_allowlist']
    )
    evidence['open_findings'] = [
        '15_build_final_quality_v2.py hash differs from the recorded run; five key stored SQL steps match exactly.',
        'V2 P95 target references remain frozen; they are not V4 train-only feature preprocessing.',
        'D06 human source-event review and scientific validity/sentiment accuracy are outside these checks.',
    ]
    (OUT / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'validation_pass': evidence['validation_pass'], 'fit_rows': row_count,
                      'numeric_checked': len(numeric), 'transforms': transforms,
                      'vocabularies': encoders}, ensure_ascii=False))


if __name__ == '__main__':
    main()
