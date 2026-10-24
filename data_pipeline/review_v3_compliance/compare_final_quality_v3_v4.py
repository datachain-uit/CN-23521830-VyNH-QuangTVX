"""Read-only, exact comparison of the current final-quality V3 and V4 Parquets."""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    roots = {v: ROOT / f'final_quality_v{v}' for v in (3, 4)}
    result = {'scope': 'Read-only exact comparison; original values use IS DISTINCT FROM / EXCEPT ALL',
              'versions': {}, 'main_parts': [], 'support_tables': {}, 'variants': {}}
    with tempfile.TemporaryDirectory(prefix='cq-v3-v4-compare-') as scratch:
        con = duckdb.connect()
        con.execute("SET threads=2; SET memory_limit='1200MB'; SET preserve_insertion_order=false")
        con.execute(f'SET temp_directory={sql_path(scratch)}')

        def view(name, path):
            con.execute(f'CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM read_parquet({sql_path(path)})')

        def schema(name):
            return {r[0]: r[1] for r in con.execute('DESCRIBE ' + name).fetchall()}

        def records(sql):
            cursor = con.execute(sql)
            return [dict(zip([x[0] for x in cursor.description], row)) for row in cursor.fetchall()]

        def multiset_diff(a, b, columns):
            selection = ','.join(map(quote, columns))
            return {f'{left}_minus_{right}': con.execute(
                f'SELECT count(*) FROM (SELECT {selection} FROM {left} EXCEPT ALL SELECT {selection} FROM {right})'
            ).fetchone()[0] for left, right in [(a, b), (b, a)]}

        for version, root in roots.items():
            view(f'final{version}', root / 'tables/user_course_final/*.parquet')
            result['versions'][str(version)] = {
                'schema': schema(f'final{version}'),
                'population': records(f'''SELECT count(*) n_rows,count(y_alpha_1) n_labeled,
                    count(*) FILTER(WHERE y_alpha_1 IS NULL) n_unlabeled FROM final{version}''')[0],
                'split_counts': records(f'''SELECT split,count(*) n_pairs,count(DISTINCT course_id) n_courses
                    FROM final{version} WHERE y_alpha_1 IS NOT NULL GROUP BY split ORDER BY split'''),
                'classes': records(f'''SELECT y_alpha_0p1,y_alpha_1,count(*) n
                    FROM final{version} GROUP BY 1,2 ORDER BY 1,2'''),
                'classes_by_split_alpha_1': records(f'''SELECT split,y_alpha_1 y,count(*) n
                    FROM final{version} WHERE y_alpha_1 IS NOT NULL GROUP BY 1,2 ORDER BY 1,2'''),
            }
        old_schema, new_schema = schema('final3'), schema('final4')
        result['main_schema_diff'] = {
            'added': sorted(new_schema.keys() - old_schema.keys()),
            'removed': sorted(old_schema.keys() - new_schema.keys()),
            'type_changes': {k: [old_schema[k], new_schema[k]] for k in old_schema.keys() & new_schema.keys()
                             if old_schema[k] != new_schema[k]},
        }
        scalar_columns = [k for k in old_schema if k not in ['split', 'chapter_details']]
        for index, path in enumerate(sorted((roots[3] / 'tables/user_course_final').glob('*.parquet')), 1):
            view('old', path)
            view('new', roots[4] / 'tables/user_course_final' / path.name)
            expressions = ['count(*) n_joined',
                'count(*) FILTER(WHERE a.user_id IS NULL OR b.user_id IS NULL) missing_key_rows',
                'count(*) FILTER(WHERE a.split IS DISTINCT FROM b.split) split_changed',
                'count(*) FILTER(WHERE a.chapter_details IS DISTINCT FROM b.chapter_details) nested_changed',
                '''count(*) FILTER(WHERE b.chapter_details IS DISTINCT FROM
                   list_transform(a.chapter_details,c -> struct_update(c,split:=b.split))) nested_non_split_changed''']
            expressions += [f'count(*) FILTER(WHERE a.{quote(k)} IS DISTINCT FROM b.{quote(k)}) AS {quote(k)}'
                            for k in scalar_columns]
            row = records('SELECT ' + ','.join(expressions) +
                          ' FROM old a FULL JOIN new b USING(user_id,course_id)')[0]
            row['part'] = path.name
            result['main_parts'].append(row)
            if index % 8 == 0:
                print(f'Main: {index}/32 parts compared', flush=True)
        sum_keys = [k for k in result['main_parts'][0] if k != 'part']
        result['main_totals'] = {k: sum(r[k] for r in result['main_parts']) for k in sum_keys}
        result['split_transition_default'] = records('''SELECT a.split v3_split,b.split v4_split,count(*) n
            FROM final3 a JOIN final4 b USING(user_id,course_id) WHERE a.y_alpha_1 IS NOT NULL
            GROUP BY 1,2 ORDER BY 1,2''')
        view('course_random3', roots[3] / 'tables/user_course_splits_course_random.parquet')
        result['v3_course_random_counts'] = records('''SELECT split,count(*) n_pairs,count(DISTINCT course_id) n_courses
            FROM course_random3 GROUP BY 1 ORDER BY 1''')
        result['split_transition_course_random'] = records('''SELECT a.split v3_split,b.split v4_split,count(*) n
            FROM course_random3 a JOIN final4 b USING(user_id,course_id) GROUP BY 1,2 ORDER BY 1,2''')
        result['v4_metadata'] = records('''SELECT
            count(*) FILTER(WHERE y_alpha_1 IS NOT NULL AND (course_family_id IS NULL OR enrollment_id IS NULL OR split IS NULL)) labeled_missing_metadata,
            count(*) FILTER(WHERE y_alpha_1 IS NULL AND (enrollment_id IS NOT NULL OR split IS NOT NULL)) unlabeled_with_assignment,
            count(DISTINCT course_family_id) FILTER(WHERE y_alpha_1 IS NOT NULL) n_families,
            count(DISTINCT enrollment_id) n_enrollment_ids FROM final4''')[0]
        result['course_overlap'] = {}
        for table in ['final3', 'course_random3', 'final4']:
            result['course_overlap'][table] = con.execute(f'''SELECT count(*) FROM
                (SELECT course_id FROM {table} WHERE split IS NOT NULL GROUP BY 1 HAVING count(DISTINCT split)>1)''').fetchone()[0]
        result['family_leakage_v4'] = con.execute('''SELECT count(*) FROM
            (SELECT course_family_id FROM final4 WHERE split IS NOT NULL GROUP BY 1 HAVING count(DISTINCT split)>1)''').fetchone()[0]
        view('members4', roots[4] / 'course_family_members.parquet')
        result['family_overlap_under_v4_definition'] = {}
        for table in ['final3', 'course_random3', 'final4']:
            result['family_overlap_under_v4_definition'][table] = con.execute(f'''SELECT count(*) FROM
                (SELECT m.course_family_id FROM {table} a JOIN members4 m USING(course_id)
                 WHERE a.split IS NOT NULL GROUP BY 1 HAVING count(DISTINCT a.split)>1)''').fetchone()[0]

        old_tables = {p.name: p for p in (roots[3] / 'tables').glob('*.parquet')}
        new_tables = {p.name: p for p in (roots[4] / 'tables').glob('*.parquet')}
        result['table_files_removed_or_replaced'] = sorted(old_tables.keys() - new_tables.keys())
        for name in sorted(old_tables.keys() & new_tables.keys()):
            view('old', old_tables[name]); view('new', new_tables[name])
            a, b = schema('old'), schema('new')
            info = {'v3_columns': len(a), 'v4_columns': len(b),
                    'added': sorted(b.keys() - a.keys()), 'removed': sorted(a.keys() - b.keys()),
                    'rows_v3': con.execute('SELECT count(*) FROM old').fetchone()[0],
                    'rows_v4': con.execute('SELECT count(*) FROM new').fetchone()[0],
                    'byte_identical': digest(old_tables[name]) == digest(new_tables[name])}
            if name == 'user_course_splits.parquet':
                info['comparison'] = 'Assignment table contract changed; split transitions compared above'
            elif info['byte_identical']:
                info['comparison'] = 'SHA256 identical'
            else:
                columns = [k for k in a if k in b and k != 'split']
                info['original_non_split_values'] = multiset_diff('old', 'new', columns)
            result['support_tables'][name] = info
            print(f'Support: {name}', flush=True)

        for alpha in ['0p1', '1']:
            for version in [3, 4]:
                view(f'variant{version}', roots[version] / f'variants/alpha_{alpha}/user_course_labels.parquet')
            a, b = schema('variant3'), schema('variant4')
            result['variants'][alpha] = {'v3_columns': len(a), 'v4_columns': len(b),
                'added': sorted(b.keys() - a.keys()), 'removed': sorted(a.keys() - b.keys()),
                'original_non_split_values': multiset_diff('variant3', 'variant4', [k for k in a if k in b and k != 'split'])}
            print(f'Variant alpha={alpha} compared', flush=True)
        policies = {v: json.loads((p / 'formula_policy.json').read_text()) for v, p in roots.items()}
        result['formula_policy_changes'] = {k: {'v3': policies[3].get(k), 'v4': policies[4].get(k)}
            for k in sorted(policies[3].keys() | policies[4].keys()) if policies[3].get(k) != policies[4].get(k)}
        result['temporal_overlap'] = json.loads((roots[4] / 'temporal_overlap.json').read_text())
        result['temporal_overlap'].pop('overlap_families', None)
        totals = result['main_totals']
        result['original_data_preserved'] = (totals['missing_key_rows'] == 0
            and totals['nested_non_split_changed'] == 0 and all(totals[k] == 0 for k in scalar_columns)
            and all(not any(v.get('original_non_split_values', {}).values()) for v in result['support_tables'].values())
            and all(not any(v['original_non_split_values'].values()) for v in result['variants'].values()))
        result['elapsed_seconds'] = round(time.monotonic() - started, 2)
        con.close()
    output = OUT / 'final_quality_v3_v4_comparison.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n')
    print(json.dumps({'original_data_preserved': result['original_data_preserved'],
                     'rows': totals['n_joined'], 'split_changed': totals['split_changed'],
                     'elapsed_seconds': result['elapsed_seconds'], 'report': str(output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
