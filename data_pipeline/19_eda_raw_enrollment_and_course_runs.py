#!/usr/bin/env python3
"""Audit arrays and course content directly in the earliest available Parquet.

Does not replace the missing original JSON or infer official course/run IDs.
Writes small audit artifacts only, without changing production datasets.
"""
import datetime
import json
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'raw_enrollment_course_runs_eda'


def time_source_checks(c, q):
    sources = [
        ('video', '_stage/video_window.parquet', 'event_time', 'event_time', 'count(*)'),
        ('problem', '_stage/problem_features/*.parquet', 'first_problem_time', 'last_problem_time', 'sum(n_problem_events)'),
        ('comment', '_stage/comment_window.parquet', 'event_time', 'event_time', 'count(*)'),
        ('reply', '_stage/reply_window.parquet', 'event_time', 'event_time', 'count(*)'),
    ]
    for kind, rel, first, last, count in sources:
        path = str(ROOT / 'final_quality_v2' / rel).replace("'", "''")
        q('labeled_' + kind + '_coverage', f'''SELECT '{kind}' source,{count} n_events,
            min({first}) first_event,max({last}) last_event
            FROM read_parquet('{path}') e JOIN labeled USING(user_id,course_id)''')


def cohort_checks(c, q):
    c.execute('''CREATE TEMP TABLE registration_stats AS SELECT course_id,count(*) n_enrolled,
        min(enroll_time) first_enroll,quantile_cont(enroll_time,0.05) p05_enroll,
        quantile_cont(enroll_time,0.5) median_enroll,quantile_cont(enroll_time,0.95) p95_enroll,
        max(enroll_time) last_enroll FROM registrations GROUP BY 1''')
    c.execute('''CREATE TEMP TABLE cohort_pairs AS WITH s AS(SELECT course_id,list(DISTINCT school_id ORDER BY school_id) ids FROM schools GROUP BY 1),
        t AS(SELECT course_id,list(DISTINCT teacher_id ORDER BY teacher_id) ids FROM teachers GROUP BY 1)
        SELECT h.*,a.n_enrolled n_enrolled1,b.n_enrolled n_enrolled2,
        a.first_enroll first1,b.first_enroll first2,a.p05_enroll p05_1,b.p05_enroll p05_2,
        a.median_enroll median1,b.median_enroll median2,a.p95_enroll p95_1,b.p95_enroll p95_2,
        a.last_enroll last1,b.last_enroll last2,
        (len(s1.ids)>0 AND s1.ids=s2.ids) same_school,
        (len(t1.ids)>0 AND t1.ids=t2.ids) same_teachers
        FROM high h JOIN registration_stats a ON a.course_id=h.c1 JOIN registration_stats b ON b.course_id=h.c2
        LEFT JOIN s s1 ON s1.course_id=h.c1 LEFT JOIN s s2 ON s2.course_id=h.c2
        LEFT JOIN t t1 ON t1.course_id=h.c1 LEFT JOIN t t2 ON t2.course_id=h.c2''')
    q('cohort_summary', '''SELECT count(*) candidate_pairs,
        count(*) FILTER(WHERE same_school AND same_teachers) pairs_same_school_teachers,
        count(*) FILTER(WHERE n_shared_video_ids=0) pairs_distinct_video_ids,
        count(*) FILTER(WHERE last1<first2 OR last2<first1) disjoint_full_enrollment_ranges,
        count(*) FILTER(WHERE p95_1<p05_2 OR p95_2<p05_1) disjoint_central_90pct_ranges,
        count(*) FILTER(WHERE abs(date_diff('day',median1,median2))>=90) median_gap_at_least_90_days,
        count(*) FILTER(WHERE regexp_matches(name1,'20[0-9]{2}') AND regexp_matches(name2,'20[0-9]{2}')) both_titles_with_year
        FROM cohort_pairs''')
    q('cohort_population_sensitivity', '''SELECT
        count(*) FILTER(WHERE n_enrolled1=1 OR n_enrolled2=1) pairs_with_single_enrollment_course,
        count(*) FILTER(WHERE n_enrolled1>=20 AND n_enrolled2>=20) pairs_both_at_least_20,
        count(*) FILTER(WHERE n_enrolled1>=20 AND n_enrolled2>=20 AND same_school AND same_teachers) matched_metadata_pairs_both_at_least_20,
        count(*) FILTER(WHERE n_enrolled1>=20 AND n_enrolled2>=20 AND same_school AND same_teachers AND (p95_1<p05_2 OR p95_2<p05_1)) matched_metadata_disjoint_central_90,
        count(*) FILTER(WHERE n_enrolled1>=20 AND n_enrolled2>=20 AND same_school AND same_teachers AND NOT(p95_1<p05_2 OR p95_2<p05_1)) matched_metadata_overlapping_central_90
        FROM cohort_pairs''')
    q('explicit_season_pair_examples', '''SELECT * FROM cohort_pairs
        WHERE regexp_matches(name1,'20[0-9]{2}') AND regexp_matches(name2,'20[0-9]{2}')
        ORDER BY n_shared DESC,c1,c2''')
    dest = str(OUT / 'course_pair_candidates.parquet').replace("'", "''")
    c.execute(f"COPY cohort_pairs TO '{dest}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    q('cohort_examples', '''SELECT * FROM cohort_pairs WHERE c1 IN('C_1760314','C_1760256','C_681739')
        AND c2 IN('C_696886','C_677179','C_682155') ORDER BY c1''')
    q('cohort_overlap_examples', '''SELECT * FROM cohort_pairs
        WHERE same_school AND same_teachers AND n_enrolled1>=100 AND n_enrolled2>=100
        AND NOT(p95_1<p05_2 OR p95_2<p05_1) ORDER BY n_shared DESC,c1,c2 LIMIT 5''')
    q('repeat_registration_user_sensitivity', '''SELECT count(*) candidate_user_pairs,
        count(DISTINCT r.user_id) users,
        count(*) FILTER(WHERE len(u.course_order)<=100) pairs_users_at_most_100_courses,
        count(DISTINCT r.user_id) FILTER(WHERE len(u.course_order)<=100) users_at_most_100_courses
        FROM repeated_content r JOIN raw_user u ON r.user_id=u.id''')
    q('top_repeat_registration_users', '''SELECT r.user_id,len(u.course_order) total_courses,count(*) candidate_pairs
        FROM repeated_content r JOIN raw_user u ON r.user_id=u.id GROUP BY 1,2
        ORDER BY candidate_pairs DESC,r.user_id LIMIT 5''')


def main():
    OUT.mkdir(exist_ok=True)
    data = {'source_layer': 'Notebook 01 Parquet; original JSON not present locally',
            'selected_courses': 'Official documentation describes course_order as selected courses; no separate selected_course field documented',
            'sources': ['https://github.com/THU-KEG/MOOCCubeX/blob/main/docs/user-en.md',
                        'https://github.com/THU-KEG/MOOCCubeX/blob/main/docs/course-en.md'],
            'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'queries': {}, 'metrics': {}}
    with tempfile.TemporaryDirectory(prefix='paper3-raw-runs-') as scratch:
        c = duckdb.connect()
        for s in ["TimeZone='UTC'", 'threads=2', "memory_limit='1200MB'", 'preserve_insertion_order=false',
                  "max_temp_directory_size='2GB'", f"temp_directory='{scratch}'"]:
            c.execute('SET ' + s)

        def view(name, rel):
            path = str(ROOT / rel).replace("'", "''")
            c.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")

        def q(name, sql):
            r = c.execute(sql)
            cols = [x[0] for x in r.description]
            rows = [dict(zip(cols, x)) for x in r.fetchall()]
            data['queries'][name] = sql
            data['metrics'][name] = rows
            print(name, json.dumps(rows, ensure_ascii=False, default=str), flush=True)
            (OUT / 'eda.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')
            return rows

        for name, rel in {
            'raw_user': 'input/moocubex_parquet/tables/entities/user/*.parquet',
            'raw_course': 'input/moocubex_parquet/tables/entities/course/*.parquet',
            'raw_video_map': 'input/moocubex_parquet/tables/relations/video_id_ccid/*.parquet',
            'schools': 'input/moocubex_parquet/tables/relations/course_school/*.parquet',
            'teachers': 'input/moocubex_parquet/tables/relations/course_teacher/*.parquet',
            'labeled': 'phase_features_v2/tables/labeled_user_course_splits.parquet',
        }.items():
            view(name, rel)
        q('user_schema', 'DESCRIBE raw_user')
        q('course_schema', 'DESCRIBE raw_course')
        q('raw_arrays', '''SELECT count(*) user_rows,count(DISTINCT id) distinct_user_ids,
            sum(coalesce(len(course_order),0)) course_entries,
            count(*) FILTER(WHERE len(course_order)=0 OR course_order IS NULL) users_no_courses,
            count(*) FILTER(WHERE len(list_filter(course_order,x->x IS NOT NULL))>list_unique(course_order)) users_repeated_course,
            sum(len(list_filter(course_order,x->x IS NOT NULL))-list_unique(course_order)) extra_repeated_entries,
            sum(len(course_order)-len(list_filter(course_order,x->x IS NOT NULL))) null_course_entries,
            count(*) FILTER(WHERE coalesce(len(course_order),0)<>coalesce(len(enroll_time),0)) length_mismatch,
            max(len(course_order)) max_courses_per_user FROM raw_user''')
        q('raw_duplicate_course_examples', '''SELECT id,course_order,enroll_time FROM raw_user
            WHERE len(list_filter(course_order,x->x IS NOT NULL))>list_unique(course_order) LIMIT 10''')
        q('raw_course_ids', 'SELECT count(*) n_rows,count(DISTINCT id) n_ids FROM raw_course')
        q('raw_resource_types', 'SELECT r.resource_type,count(*) n FROM raw_course CROSS JOIN unnest(resource) t(r) GROUP BY 1')
        # Raw resource_type is NULL in this copy. Infer videos from documented V_ IDs.
        c.execute("""CREATE TEMP TABLE courses AS SELECT *,
            lower(regexp_replace(trim(name),'\\s+',' ','g')) nm,
            lower(regexp_replace(trim(regexp_replace(name,'[（(][^）)]*20[0-9]{2}[^）)]*[）)]',' ','g')),'\\s+',' ','g')) base_name
            FROM raw_course""")
        c.execute("""CREATE TEMP TABLE cv AS SELECT DISTINCT id course_id,r.resource_id video_id
            FROM raw_course CROSS JOIN unnest(resource) t(r) WHERE starts_with(r.resource_id,'V_')""")
        c.execute('''CREATE TEMP TABLE vm AS SELECT video_id,min(ccid) ccid,count(DISTINCT ccid) n_ccids
            FROM raw_video_map WHERE video_id IN(SELECT video_id FROM cv) GROUP BY 1''')
        c.execute('''CREATE TEMP TABLE cc AS SELECT DISTINCT course_id,ccid FROM cv JOIN vm USING(video_id)
            WHERE n_ccids=1 AND ccid IS NOT NULL''')
        c.execute('''CREATE TEMP TABLE pairs AS SELECT a.id c1,b.id c2,a.name name1,b.name name2,
            a.nm=b.nm exact_title,a.about=b.about same_description
            FROM courses a JOIN courses b ON a.base_name=b.base_name AND a.id<b.id''')
        c.execute('''CREATE TEMP TABLE overlap AS WITH sizes AS(SELECT course_id,count(*) n FROM cc GROUP BY 1),
            shared AS(SELECT p.c1,p.c2,count(*) n_shared FROM pairs p JOIN cc a ON a.course_id=p.c1
              JOIN cc b ON b.course_id=p.c2 AND a.ccid=b.ccid GROUP BY 1,2),
            shared_ids AS(SELECT p.c1,p.c2,count(*) n_shared_ids FROM pairs p JOIN cv a ON a.course_id=p.c1
              JOIN cv b ON b.course_id=p.c2 AND a.video_id=b.video_id GROUP BY 1,2)
            SELECT p.*,a.n n1,b.n n2,coalesce(s.n_shared,0) n_shared,coalesce(i.n_shared_ids,0) n_shared_video_ids,
            coalesce(s.n_shared,0)::DOUBLE/(a.n+b.n-coalesce(s.n_shared,0)) jaccard
            FROM pairs p JOIN sizes a ON a.course_id=p.c1 JOIN sizes b ON b.course_id=p.c2
            LEFT JOIN shared s USING(c1,c2) LEFT JOIN shared_ids i USING(c1,c2)''')
        q('video_content_summary', '''SELECT (SELECT count(*) FROM cv) course_video_pairs,
            (SELECT count(*) FROM cv ANTI JOIN vm USING(video_id)) video_pairs_without_map,
            (SELECT count(*) FROM vm WHERE n_ccids>1) ambiguous_video_ids,
            (SELECT count(*) FROM cc) course_ccid_pairs,(SELECT count(*) FROM pairs) title_candidate_pairs,
            count(*) FILTER(WHERE n1>=5 AND n2>=5 AND jaccard>=0.9) high_content_overlap_pairs,
            count(*) FILTER(WHERE n1>=5 AND n2>=5 AND jaccard=1) identical_content_pairs
            FROM overlap''')
        q('high_overlap_examples', '''SELECT * FROM overlap WHERE n1>=5 AND n2>=5 AND jaccard>=0.9
            ORDER BY exact_title ASC,n_shared DESC,c1,c2 LIMIT 15''')
        c.execute('''CREATE TEMP TABLE high AS SELECT * FROM overlap WHERE n1>=5 AND n2>=5 AND jaccard>=0.9''')
        # Projection UNNEST avoids repeating each user's large input arrays.
        c.execute('''CREATE TEMP TABLE registrations AS WITH x AS(
            SELECT id user_id,unnest(list_zip(course_order,enroll_time)) entry FROM raw_user),typed AS(
            SELECT user_id,'C_'||cast(entry[1] AS VARCHAR) course_id,entry[2] enroll_time_raw,
                try_cast(entry[2] AS TIMESTAMP) enroll_time FROM x)
            SELECT * FROM typed WHERE course_id IN(SELECT c1 FROM high UNION SELECT c2 FROM high)''')
        c.execute('''CREATE TEMP TABLE repeated_content AS SELECT h.*,a.user_id,a.enroll_time t1,b.enroll_time t2
            FROM high h JOIN registrations a ON a.course_id=h.c1
            JOIN registrations b ON b.course_id=h.c2 AND b.user_id=a.user_id''')
        q('repeated_content_registrations', '''SELECT count(*) user_course_pair_candidates,
            count(DISTINCT user_id) users,count(DISTINCT(c1,c2)) course_pairs,
            count(*) FILTER(WHERE t1 IS DISTINCT FROM t2) different_enroll_times,
            count(*) FILTER(WHERE jaccard=1) identical_content_user_pairs FROM repeated_content''')
        q('repeated_content_examples', '''SELECT * FROM repeated_content
            WHERE abs(date_diff('day',t1,t2))>=90
            ORDER BY jaccard DESC,same_description DESC,n_shared DESC,user_id,c1,c2 LIMIT 10''')
        cohort_checks(c, q)
        q('course_term_mentions', '''SELECT id,name FROM raw_course
            WHERE regexp_matches(name,'20[12][0-9]|第.{0,3}[期轮]|春季|秋季|夏季|冬季') LIMIT 30''')
        q('course_repeat_text_evidence', '''SELECT id,name,about FROM raw_course
            WHERE id IN('C_697165','C_1889028')''')
        # Date coverage: metadata and actual observed events are distinct concepts.
        q('all_phase_metadata', """SELECT count(*) n,min(enroll_time) first_enroll,max(enroll_time) last_enroll
            FROM read_parquet('""" + str(ROOT / 'phase_features_v2/features/P1/*.parquet') + """')""")
        for phase in ['P1', 'P2', 'P3', 'P4']:
            view('f_' + phase, f'phase_features_v2/supervised_raw/{phase}/*.parquet')
            view('a_' + phase, f'phase_features_v2/audit/{phase}/*.parquet')
            q(phase + '_labeled_metadata', f'''SELECT count(*) n,min(enroll_time) first_enroll,
                max(enroll_time) last_enroll,min(cutoff_time) first_cutoff,max(cutoff_time) last_cutoff FROM f_{phase}''')
            q(phase + '_labeled_events', f'''SELECT count(*) n_pairs_with_activity,
                min(min_feature_event_time) first_event,max(max_feature_event_time) last_event
                FROM a_{phase} JOIN labeled USING(user_id,course_id)''')
        view('visits', 'final_quality_v2/_stage/visit_features/*.parquet')
        q('full_labeled_window', '''SELECT min(first_observed_event) first_event,max(last_observed_event) last_event,
            date_diff('day',cast(min(first_observed_event) AS DATE),cast(max(last_observed_event) AS DATE))+1 calendar_days_inclusive
            FROM visits JOIN labeled USING(user_id,course_id)''')
        q('first_labeled_event_case', '''SELECT v.* FROM visits v JOIN labeled USING(user_id,course_id)
            ORDER BY first_observed_event LIMIT 1''')
        q('last_labeled_event_case', '''SELECT v.* FROM visits v JOIN labeled USING(user_id,course_id)
            ORDER BY last_observed_event DESC LIMIT 1''')
        time_source_checks(c, q)
        c.close()
    print('Saved', OUT / 'eda.json', flush=True)


if __name__ == '__main__':
    main()
