#!/usr/bin/env python3
"""Read-only audit of available enrollment lineage and offering candidates.

Writes only audit JSON; never rebuilds or changes source datasets.
Original JSON is not locally available: the earliest audited records are Parquet.
"""
import datetime
import json
import tempfile
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'enrollment_offering_audit'


def metadata_split_checks(query):
    group_sql = '''WITH labeled AS(SELECT DISTINCT course_id,split FROM targets), g AS(
        SELECT n.nm,trim(n.about) about,n.school_ids,n.teacher_ids,count(*) nc,
            count(DISTINCT t.split) ns,
            list(struct_pack(course_id:=n.course_id,split:=t.split) ORDER BY n.course_id) courses
        FROM signatures n LEFT JOIN labeled t USING(course_id)
        WHERE length(trim(coalesce(n.about,'')))>50 AND len(n.school_ids)>0 AND len(n.teacher_ids)>0
        GROUP BY 1,2,3,4 HAVING count(*)>1)'''
    query('metadata_split_overlap', group_sql + ''' SELECT count(*) groups_same_metadata,
        sum(nc) course_ids,count(*) FILTER(WHERE ns>1) labeled_groups_crossing_splits FROM g''')
    query('metadata_split_examples', group_sql + ''' SELECT nm,school_ids,teacher_ids,courses
        FROM g WHERE ns>1 ORDER BY nm''')


def main():
    OUT.mkdir(exist_ok=True)
    result = {'audited_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'earliest_available_source': 'Notebook 01 Parquet, not original JSON',
              'queries': {}, 'metrics': {}}
    with tempfile.TemporaryDirectory(prefix='paper3-offering-audit-') as scratch:
        con = duckdb.connect()
        for setting in ["TimeZone='UTC'", 'threads=2', "memory_limit='1500MB'", 'preserve_insertion_order=false',
                        "max_temp_directory_size='3GB'", f"temp_directory='{scratch}'"]:
            con.execute('SET ' + setting)

        def view(name, rel):
            path = str(ROOT / rel).replace("'", "''")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")

        def query(name, sql):
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            result['queries'][name] = sql
            result['metrics'][name] = rows
            print(name, json.dumps(rows, ensure_ascii=False, default=str), flush=True)
            return rows

        def scalars(name, statements):
            # Run large joins separately so their hash tables do not coexist.
            row = {k: con.execute(sql).fetchone()[0] for k, sql in statements.items()}
            result['queries'][name] = statements
            result['metrics'][name] = [row]
            print(name, json.dumps(row, ensure_ascii=False, default=str), flush=True)

        for name, rel in {
            'raw_user': 'input/moocubex_parquet/tables/entities/user/*.parquet',
            'raw_course': 'input/moocubex_parquet/tables/entities/course/*.parquet',
            'courses': 'input/moocubex_model/processed/dim_course/*.parquet',
            'enroll': 'input/moocubex_model/processed/bridge_user_course_enrollment/*.parquet',
            'old_times': 'review_time_phases/tables/user_course_phase_times.parquet',
            'times': 'final_quality_v2/tables/user_course_phase_times.parquet',
            'final': 'final_quality_v2/tables/user_course_final/*.parquet',
            'splits': 'final_quality_v2/tables/course_splits.parquet',
            'targets': 'phase_features_v2/tables/labeled_user_course_splits.parquet',
            'schools': 'input/moocubex_parquet/tables/relations/course_school/*.parquet',
            'teachers': 'input/moocubex_parquet/tables/relations/course_teacher/*.parquet',
        }.items():
            view(name, rel)
        query('raw_users', '''SELECT count(*) user_rows,count(DISTINCT id) distinct_user_ids,
            sum(len(course_order)) course_entries,
            count(*) FILTER(WHERE coalesce(len(course_order),0)<>coalesce(len(enroll_time),0)) mismatched_arrays
            FROM raw_user''')
        con.execute('''CREATE TEMP TABLE flat AS SELECT id user_id,idx array_index,
            'C_'||cast(list_extract(course_order,idx) AS VARCHAR) course_id,
            list_extract(enroll_time,idx) enroll_time_raw,
            try_cast(list_extract(enroll_time,idx) AS TIMESTAMP) enroll_time
            FROM raw_user CROSS JOIN unnest(range(1,coalesce(len(course_order),0)+1)) t(idx)''')
        query('same_id_reenrollments', '''WITH g AS(SELECT user_id,course_id,count(*) n,
            count(DISTINCT enroll_time_raw) nt FROM flat GROUP BY 1,2)
            SELECT sum(n) enrollment_entries,count(*) user_course_pairs,
            count(*) FILTER(WHERE n>1) repeated_pairs,
            count(*) FILTER(WHERE n>1 AND nt>1) repeated_pairs_with_different_times,max(n) max_per_pair FROM g''')
        query('raw_bridge_reconciliation', '''SELECT
            (SELECT count(*) FROM flat ANTI JOIN enroll USING(user_id,course_id,array_index)) missing_entries,
            (SELECT count(*) FROM enroll ANTI JOIN flat USING(user_id,course_id,array_index)) extra_entries,
            (SELECT count(*) FROM flat f JOIN enroll e USING(user_id,course_id,array_index)
             WHERE f.enroll_time IS DISTINCT FROM e.enroll_time
                OR f.enroll_time_raw IS DISTINCT FROM e.enroll_time_raw) changed_timestamps''')
        con.execute('DROP TABLE flat')
        query('course_rows', '''SELECT (SELECT count(*) FROM raw_course) raw_rows,
            (SELECT count(DISTINCT id) FROM raw_course) raw_ids,
            (SELECT count(*) FROM courses) dimension_rows,
            (SELECT count(*) FROM courses WHERE source_row_count>1) ids_with_collapsed_source_rows''')
        counts = []
        for name in ['enroll', 'old_times', 'times', 'final', 'targets']:
            counts += query(name + '_keys', f'''SELECT '{name}' table_name,count(*) n_rows,
                count(DISTINCT(user_id,course_id)) n_keys,count(DISTINCT course_id) n_courses FROM {name}''')
        scalars('lineage_keys', {
            'bridge_missing_in_old_times': 'SELECT count(*) FROM enroll ANTI JOIN old_times USING(user_id,course_id)',
            'old_times_extra': 'SELECT count(*) FROM old_times ANTI JOIN enroll USING(user_id,course_id)',
            'old_times_missing_in_v2': 'SELECT count(*) FROM old_times ANTI JOIN times USING(user_id,course_id)',
            'v2_extra': 'SELECT count(*) FROM times ANTI JOIN old_times USING(user_id,course_id)',
            'eligible_missing_in_final': '''SELECT count(*) FROM times t LEFT JOIN final f USING(user_id,course_id)
                WHERE t.time_eligible AND f.user_id IS NULL''',
            'final_ineligible_or_extra': '''SELECT count(*) FROM final f LEFT JOIN times t USING(user_id,course_id)
                WHERE t.user_id IS NULL OR NOT t.time_eligible''',
        })
        query('time_statuses', 'SELECT time_status,count(*) n FROM times GROUP BY 1 ORDER BY 1')
        for phase in ['P1', 'P2', 'P3', 'P4']:
            view('x_' + phase, f'phase_features_v2/features/{phase}/*.parquet')
            view('y_' + phase, f'phase_features_v2/supervised_raw/{phase}/*.parquet')
            scalars(phase + '_keys', {
                'feature_rows': f'SELECT count(*) FROM x_{phase}',
                'feature_keys': f'SELECT count(DISTINCT(user_id,course_id)) FROM x_{phase}',
                'missing_final_keys': f'SELECT count(*) FROM final ANTI JOIN x_{phase} USING(user_id,course_id)',
                'extra_feature_keys': f'SELECT count(*) FROM x_{phase} ANTI JOIN final USING(user_id,course_id)',
                'supervised_rows': f'SELECT count(*) FROM y_{phase}',
                'supervised_keys': f'SELECT count(DISTINCT(user_id,course_id)) FROM y_{phase}',
                'missing_labeled_keys': f'SELECT count(*) FROM targets ANTI JOIN y_{phase} USING(user_id,course_id)',
                'extra_supervised_keys': f'SELECT count(*) FROM y_{phase} ANTI JOIN targets USING(user_id,course_id)',
            })
        con.execute("""CREATE TEMP TABLE names AS SELECT course_id,course_name,about,
            lower(regexp_replace(trim(course_name),'\\s+',' ','g')) nm
            FROM courses WHERE course_name IS NOT NULL AND trim(course_name)<>''""")
        query('same_title_courses', '''WITH g AS(SELECT nm,count(*) n FROM names GROUP BY 1)
            SELECT count(*) FILTER(WHERE n>1) title_groups,sum(n) FILTER(WHERE n>1) course_ids,max(n) max_ids FROM g''')
        con.execute('''CREATE TEMP TABLE user_names AS SELECT e.user_id,n.nm,
            count(DISTINCT e.course_id) n_courses,min(e.enroll_time) first_enroll,max(e.enroll_time) last_enroll
            FROM enroll e JOIN names n USING(course_id) GROUP BY 1,2 HAVING count(DISTINCT e.course_id)>1''')
        query('same_title_user_candidates', '''SELECT count(*) user_title_pairs,
            count(DISTINCT user_id) users,sum(n_courses) enrollment_rows,
            count(*) FILTER(WHERE first_enroll<last_enroll) different_timestamps,
            count(*) FILTER(WHERE first_enroll=last_enroll) identical_timestamps FROM user_names''')
        query('candidate_examples', '''SELECT g.user_id,n.nm,
            list(struct_pack(course_id:=e.course_id,enroll_time:=e.enroll_time) ORDER BY e.enroll_time,e.course_id) courses
            FROM (SELECT * FROM user_names WHERE date_diff('day',first_enroll,last_enroll)>90
              ORDER BY length(nm) DESC,n_courses DESC,user_id LIMIT 5) g
            JOIN enroll e USING(user_id) JOIN names n ON e.course_id=n.course_id AND n.nm=g.nm GROUP BY 1,2''')
        con.execute('''CREATE TEMP TABLE signatures AS SELECT n.*,
            s.school_ids,t.teacher_ids FROM names n
            LEFT JOIN (SELECT course_id,list(DISTINCT school_id ORDER BY school_id) school_ids FROM schools GROUP BY 1) s USING(course_id)
            LEFT JOIN (SELECT course_id,list(DISTINCT teacher_id ORDER BY teacher_id) teacher_ids FROM teachers GROUP BY 1) t USING(course_id)''')
        query('stronger_metadata_candidates', '''WITH g AS(SELECT nm,trim(about) a,school_ids,teacher_ids,count(*) n
            FROM signatures WHERE length(trim(coalesce(about,'')))>50
            AND len(school_ids)>0 AND len(teacher_ids)>0 GROUP BY 1,2,3,4 HAVING count(*)>1)
            SELECT count(*) groups_same_title_description_school_teachers,sum(n) course_ids FROM g''')
        metadata_split_checks(query)
        query('labeled_title_split_overlap', '''WITH g AS(SELECT n.nm,count(DISTINCT t.split) ns,
            count(DISTINCT t.course_id) nc FROM names n JOIN targets t USING(course_id) GROUP BY 1)
            SELECT count(*) FILTER(WHERE ns>1) candidate_groups_crossing_splits,
            sum(nc) FILTER(WHERE ns>1) labeled_course_ids FROM g''')
        query('actual_course_split_overlap', '''SELECT count(*) n_course_ids_in_multiple_splits FROM
            (SELECT course_id FROM targets GROUP BY 1 HAVING count(DISTINCT split)>1)''')
        con.close()
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + '\n')
    print('Saved', OUT / 'audit.json', flush=True)


if __name__ == '__main__':
    main()
