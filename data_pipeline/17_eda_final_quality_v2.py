"""Reproducible final-quality-v2 EDA with a detailed audit of every E label.

Reads completed parquet outputs; writes separate aggregate tables, E trace tables,
figures, a Vietnamese report, and a manifest. Never changes labels or source files.
Dependencies: duckdb, numpy, matplotlib. Python >= 3.9.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import time

import duckdb

ROOT = Path(__file__).resolve().parent
METRICS = ['TS', 'RV', 'AV', 'IF', 'FV', 'FA', 'KL', 'CFA',
           'COELO', 'AFELO', 'ACELO', 'LOQV_alpha_0p1', 'LOQV_alpha_1']
ALPHAS = [('0p1', 0.1), ('1', 1.0)]
LABELS = ['E', 'G', 'A']


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def qi(value):
    return '"' + str(value).replace('"', '""') + '"'


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=str, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(con, sql):
    cur = con.execute(sql)
    names = [x[0] for x in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def label(q):
    return None if q is None else 'E' if q < 0.2 else 'G' if q < 0.4 else 'A'


def pct(n, d):
    return 100 * n / d if d else 0.0


def table(headers, rows):
    def cell(v):
        if v is None:
            return 'NULL'
        if isinstance(v, float):
            return f'{v:,.4f}'
        if isinstance(v, int):
            return f'{v:,}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(cell(x) for x in headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(cell(x) for x in row) + ' |' for row in rows])


class EDA:
    def __init__(self, root, out, memory='1500MB', threads=2):
        self.root = Path(root).resolve()
        self.source = self.root / 'final_quality_v2'
        self.out = Path(out).resolve()
        if self.out == self.source or self.source in self.out.parents:
            raise ValueError('Output must be separate from final_quality_v2')
        for sub in ['', 'tables', 'figures', '_scratch']:
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        (self.out / '_SUCCESS.json').unlink(missing_ok=True)
        source_checks = json.loads((self.source / 'validation.json').read_text())
        if not source_checks or not all(v is True for v in source_checks.values()):
            raise RuntimeError('Source validation is incomplete or contains failures')
        success = json.loads((self.source / '_SUCCESS.json').read_text())
        self.source_summary = json.loads((self.source / 'summary.json').read_text())
        if success['version'] != self.source_summary['formula_version']:
            raise RuntimeError('Source success marker and summary version disagree')
        self.con = duckdb.connect()
        for setting in [f'memory_limit={qs(memory)}', f'threads={int(threads)}',
                        "TimeZone='UTC'", 'preserve_insertion_order=false',
                        f'temp_directory={qs(self.out / "_scratch")}',
                        "max_temp_directory_size='2GB'"]:
            self.con.execute('SET ' + setting)
        self.inputs = []
        self.log = []
        self.results = {}
        self.started = time.monotonic()
        self.source_checks = source_checks
        paths = {
            'final': self.source / 'tables/user_course_final/*.parquet',
            'chapters': self.source / 'tables/chapter_quality.parquet',
            'chapter_ref': self.source / 'tables/chapter_p95_reference.parquet',
            'video_ref': self.source / 'tables/video_p95_reference.parquet',
            'sentiment': self.source / 'tables/comment_sentiment.parquet',
            'comment_links': self.source / 'tables/afelo_comment_links.parquet',
            'splits': self.source / 'tables/course_splits.parquet',
            'catalog': self.root / 'input/moocubex_model/processed/dim_lo_catalog/*.parquet',
        }
        for name, pattern in paths.items():
            files = sorted(pattern.parent.glob(pattern.name))
            if not files:
                raise FileNotFoundError(pattern)
            self.inputs.extend(files)
            self.con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({qs(pattern)})')
        for name in ['summary.json', 'validation.json', 'formula_policy.json', '_SUCCESS.json']:
            self.inputs.append(self.source / name)
        self.inputs.append(Path(__file__).resolve())
        self.inventory = [self.fingerprint(p) for p in sorted(set(self.inputs))]
        self.con.execute('''CREATE TEMP TABLE labeled AS
          SELECT * EXCLUDE(chapter_details) FROM final WHERE q_alpha_0p1 IS NOT NULL''')
        self.con.execute('''CREATE TEMP TABLE e_keys AS
          SELECT user_id,course_id FROM labeled WHERE y_alpha_0p1='E' OR y_alpha_1='E' ''')
        self.con.execute('''CREATE TEMP TABLE catalog_counts AS
          SELECT course_id,count(DISTINCT lo_key) FILTER(WHERE chapter_major>=0) n_catalog_chapters
          FROM catalog GROUP BY 1''')

    def fingerprint(self, path):
        st = path.stat()
        return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
                'sha256': sha(path)}

    def save(self, name, sql, retain=True):
        tick = time.monotonic()
        path = self.out / 'tables' / (name + '.parquet')
        self.con.execute(f'COPY ({sql}) TO {qs(path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        self.con.execute(f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({qs(path)})')
        self.con.execute(f'''COPY (SELECT * FROM {name}) TO {qs(path.with_suffix('.csv'))}
          (FORMAT CSV,HEADER true,NULL 'NULL')''')
        n = self.con.execute(f'SELECT count(*) FROM {name}').fetchone()[0]
        self.log.append({'name': name, 'rows': n, 'sql': sql,
                         'elapsed_seconds': round(time.monotonic() - tick, 3)})
        if retain:
            self.results[name] = records(self.con, f'SELECT * FROM {name}')
        print(f'{name}: {n:,} rows ({self.log[-1]["elapsed_seconds"]}s)', flush=True)

    def overview(self):
        self.save('population', '''SELECT count(*) n_eligible_pairs,
          count(*) FILTER(WHERE n_chapters_observed>0) n_observed_pairs,
          count(q_alpha_0p1) n_labeled_pairs,count(*)-count(q_alpha_0p1) n_unlabeled_pairs,
          count(DISTINCT course_id) n_courses,
          100.0*count(q_alpha_0p1)/count(*) pct_labeled FROM final''')
        self.save('split_summary', '''SELECT split,count(*) n_eligible_pairs,
          count(DISTINCT course_id) n_courses,count(q_alpha_0p1) n_labeled_pairs,
          100.0*count(q_alpha_0p1)/count(*) pct_labeled,
          avg(q_alpha_0p1) mean_q_0p1,avg(q_alpha_1) mean_q_1,
          count(*) FILTER(WHERE y_alpha_0p1='E') n_E_0p1,
          count(*) FILTER(WHERE y_alpha_1='E') n_E_1
          FROM final GROUP BY 1 ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'dev' THEN 2 ELSE 3 END''')
        self.save('course_summary', '''SELECT f.course_id,min(f.split) split,
          count(*) n_eligible_pairs,count(f.q_alpha_0p1) n_labeled_pairs,
          100.0*count(f.q_alpha_0p1)/count(*) pct_labeled,
          count(*) FILTER(WHERE f.y_alpha_0p1='E') n_E_0p1,
          count(*) FILTER(WHERE f.y_alpha_1='E') n_E_1,
          avg(f.q_alpha_0p1) mean_q_0p1,avg(f.q_alpha_1) mean_q_1
          FROM final f GROUP BY 1 ORDER BY n_labeled_pairs DESC,course_id''')
        self.save('component_masks', '''SELECT component_mask,chapter_major=-1 is_unknown,
          count(*) n_chapters,count(DISTINCT(user_id,course_id)) n_pairs,
          avg(COELO) mean_COELO,avg(AFELO) mean_AFELO,avg(ACELO) mean_ACELO,
          avg(LOQV_alpha_0p1) mean_loqv_0p1,avg(LOQV_alpha_1) mean_loqv_1,
          count(FA) n_with_FA FROM chapters GROUP BY 1,2 ORDER BY n_chapters DESC''')
        self.save('source_patterns', '''SELECT has_video,has_problem,has_social,
          count(*) n_chapters,count(*) FILTER(WHERE chapter_major=-1) n_unknown_chapters
          FROM chapters GROUP BY 1,2,3 ORDER BY n_chapters DESC''')
        self.save('label_distribution', ' UNION ALL '.join(f'''
          SELECT {qs(s)} alpha,split,y_alpha_{s} y,count(*) n_pairs,
            100.0*count(*)/sum(count(*)) OVER(PARTITION BY split) pct_labeled_in_split
          FROM labeled GROUP BY split,y_alpha_{s}''' for s, _ in ALPHAS))
        self.save('alpha_transition', '''SELECT y_alpha_0p1,y_alpha_1,count(*) n_pairs
          FROM labeled GROUP BY 1,2 ORDER BY 1,2''')
        self.save('observed_chapter_counts', '''SELECT n_chapters_observed,
          count(*) n_pairs,count(*) FILTER(WHERE y_alpha_0p1='E') n_E_0p1,
          count(*) FILTER(WHERE y_alpha_1='E') n_E_1
          FROM labeled GROUP BY 1 ORDER BY 1''')
        profiles = []
        for cohort, predicate in [('ALL_OBSERVED', 'true'), ('KNOWN_CHAPTER', 'chapter_major<>-1'),
                                  ('UNKNOWN_CHAPTER', 'chapter_major=-1')]:
            for metric in METRICS:
                v = qi(metric)
                profiles.append(f'''SELECT {qs(cohort)} cohort,{qs(metric)} metric,
                  count(*) n_rows,count({v}) n_nonnull,count(*)-count({v}) n_null,
                  count(*) FILTER(WHERE {v}=0) n_zero,count(*) FILTER(WHERE {v}=1) n_one,
                  count(*) FILTER(WHERE {v} IS NOT NULL AND NOT isfinite({v})) n_nonfinite,
                  min({v}) minimum,avg({v}) mean,stddev_pop({v}) stddev,
                  quantile_cont({v},[0.01,0.05,0.25,0.5,0.75,0.95,0.99])[1] p01,
                  quantile_cont({v},[0.01,0.05,0.25,0.5,0.75,0.95,0.99])[3] p25,
                  quantile_cont({v},[0.01,0.05,0.25,0.5,0.75,0.95,0.99])[4] median,
                  quantile_cont({v},[0.01,0.05,0.25,0.5,0.75,0.95,0.99])[5] p75,
                  quantile_cont({v},[0.01,0.05,0.25,0.5,0.75,0.95,0.99])[6] p95,
                  max({v}) maximum FROM chapters WHERE {predicate}''')
        self.save('metric_profile', ' UNION ALL '.join(profiles))
        histograms = []
        for grain, relation, metrics in [('chapter', 'chapters', METRICS),
                                         ('pair', 'labeled', ['q_alpha_0p1', 'q_alpha_1'])]:
            for m in metrics:
                histograms.append(f'''SELECT {qs(grain)} grain,{qs(m)} metric,
                  least(49,floor({qi(m)}*50))::INT bin,count(*) n_rows
                  FROM {relation} WHERE {qi(m)} IS NOT NULL GROUP BY 3''')
        self.save('histograms', ' UNION ALL '.join(histograms))
        self.save('p95_support', ' UNION ALL '.join(f'''
          SELECT {qs(metric)} metric,
            CASE WHEN {n}=0 THEN '0' WHEN {n}=1 THEN '1' WHEN {n}<=5 THEN '2-5'
              WHEN {n}<=20 THEN '6-20' ELSE '>20' END positive_support,
            count(*) n_reference_groups,sum({n}) n_positive_contributions,
            min({p}) min_p95,median({p}) median_p95,max({p}) max_p95
          FROM {rel} GROUP BY 2''' for metric, rel, n, p in [
              ('TS_video', 'video_ref', 'n_users_positive_time', 'p95_total_content_seconds'),
              ('IF', 'chapter_ref', 'n_users_if_positive', 'p95_IF_raw'),
              ('FV', 'chapter_ref', 'n_users_visits_positive', 'p95_n_visits'),
              ('FA', 'chapter_ref', 'n_users_fa_nonzero', 'p95_abs_FA_raw')]))
        correlations = []
        for grain, rel, metrics in [
                ('chapter_complete', 'chapters WHERE loqv_n_components=3', ['COELO', 'AFELO', 'ACELO']),
                ('pair_available', 'labeled', ['COELO_mean_chapter', 'AFELO_mean_chapter',
                                              'ACELO_mean_chapter', 'q_alpha_0p1', 'q_alpha_1'])]:
            for a, b in itertools.combinations(metrics, 2):
                correlations.append(f'''SELECT {qs(grain)} grain,{qs(a)} x,{qs(b)} y,
                  count(*) FILTER(WHERE {qi(a)} IS NOT NULL AND {qi(b)} IS NOT NULL) n_pairwise,
                  corr({qi(a)},{qi(b)}) pearson_r FROM {rel}''')
        self.save('correlations', ' UNION ALL '.join(correlations))

    def e_audit(self):
        self.con.execute('''CREATE TEMP TABLE counterfactual AS
          SELECT user_id,course_id,
            avg(LOQV_alpha_0p1) FILTER(WHERE chapter_major<>-1) q_known_0p1,
            avg(LOQV_alpha_1) FILTER(WHERE chapter_major<>-1) q_known_1,
            avg(LOQV_alpha_0p1) FILTER(WHERE loqv_n_components=3) q_full_0p1,
            avg(LOQV_alpha_1) FILTER(WHERE loqv_n_components=3) q_full_1,
            count(*) FILTER(WHERE chapter_major<>-1) n_known_chapters,
            count(*) FILTER(WHERE loqv_n_components=3) n_full_chapters,
            min(n_users_visits_positive) min_FV_positive_support,
            min(n_users_if_positive) FILTER(WHERE IF_raw>0) min_IF_positive_support,
            min(n_users_fa_nonzero) FILTER(WHERE FA_raw<>0) min_FA_nonzero_support,
            count(*) FILTER(WHERE n_users_visits_positive<=5) n_FV_small_support_chapters,
            count(*) FILTER(WHERE n_visits>=p95_n_visits) n_FV_saturated_chapters,
            count(*) FILTER(WHERE FA IS NOT NULL) n_FA_chapters,
            count(*) FILTER(WHERE FA_raw<>0) n_FA_nonzero_chapters,
            count(*) FILTER(WHERE FA=1) n_FA_saturated_positive_chapters,
            max(abs(LOQV_alpha_1-LOQV_alpha_0p1)) max_chapter_alpha_delta
          FROM chapters GROUP BY 1,2''')
        self.con.execute('''CREATE TEMP VIEW enriched AS SELECT l.*,c.* EXCLUDE(user_id,course_id),
          k.n_catalog_chapters,
          c.n_known_chapters::DOUBLE/nullif(k.n_catalog_chapters,0) known_catalog_coverage,
          CASE WHEN n_unknown_chapters=n_chapters_observed THEN 'UNKNOWN_ONLY'
            WHEN n_chapters_full_components=n_chapters_observed THEN 'ALL_FULL'
            WHEN n_chapters_full_components=0 THEN 'NO_FULL' ELSE 'MIXED' END evidence_cohort
          FROM labeled l JOIN counterfactual c USING(user_id,course_id)
          LEFT JOIN catalog_counts k USING(course_id)''')
        self.save('e_cohort_summary', ' UNION ALL '.join(f'''
          SELECT {qs(s)} alpha,evidence_cohort,count(*) n_E,
            count(*) FILTER(WHERE n_chapters_observed=1) n_single_chapter,
            count(*) FILTER(WHERE n_video_events_sum_observed=0 AND n_problem_events_sum_observed=0) n_social_only,
            count(*) FILTER(WHERE n_sentiment_comments_sum_observed=0) n_without_sentiment,
            count(*) FILTER(WHERE q_alpha_{s}=0) n_q_zero,
            count(*) FILTER(WHERE min_FV_positive_support<=5) n_any_FV_support_le5,
            avg(q_alpha_{s}) mean_q,median(n_visits_sum_observed) median_visits,
            median(n_chapters_observed) median_chapters,
            median(known_catalog_coverage) median_known_catalog_coverage,
            count(*) FILTER(WHERE q_known_{s}<0.2) n_E_on_known_only,
            count(*) FILTER(WHERE q_full_{s}<0.2) n_E_on_full_only,
            count(*) FILTER(WHERE q_known_{s} IS NULL) n_no_known_evidence,
            count(*) FILTER(WHERE q_full_{s} IS NULL) n_no_full_evidence,
            count(*) FILTER(WHERE q_known_{s}>=0.2) n_downgraded_on_known_only,
            count(*) FILTER(WHERE q_full_{s}>=0.2) n_downgraded_on_full_only
          FROM enriched WHERE y_alpha_{s}='E' GROUP BY 2''' for s, _ in ALPHAS))
        self.save('e_by_split', ' UNION ALL '.join(f'''
          SELECT {qs(s)} alpha,split,evidence_cohort,count(*) n_E
          FROM enriched WHERE y_alpha_{s}='E' GROUP BY 2,3''' for s, _ in ALPHAS))
        self.save('e_by_course', ' UNION ALL '.join(f'''
          SELECT {qs(s)} alpha,course_id,min(split) split,count(*) n_E,
            count(*) FILTER(WHERE evidence_cohort='UNKNOWN_ONLY') n_unknown_only,
            count(*) FILTER(WHERE evidence_cohort='ALL_FULL') n_all_full,
            avg(q_alpha_{s}) mean_q FROM enriched WHERE y_alpha_{s}='E'
          GROUP BY 2''' for s, _ in ALPHAS))
        self.save('e_vs_other_labels', ' UNION ALL '.join(f'''
          SELECT {qs(s)} alpha,y_alpha_{s} y,evidence_cohort,count(*) n_pairs,
            avg(n_chapters_observed) mean_chapters,median(n_chapters_observed) median_chapters,
            avg(known_catalog_coverage) mean_known_catalog_coverage,
            avg(COELO_mean_chapter) mean_COELO,avg(AFELO_mean_chapter) mean_AFELO,
            avg(ACELO_mean_chapter) mean_ACELO,median(n_visits_sum_observed) median_visits,
            avg(q_alpha_{s}) mean_q FROM enriched GROUP BY 2,3''' for s, _ in ALPHAS))
        self.save('e_cases', '''SELECT user_id,course_id,split,evidence_cohort,
          q_alpha_0p1,y_alpha_0p1,q_alpha_1,y_alpha_1,
          q_known_0p1,q_known_1,q_full_0p1,q_full_1,
          n_chapters_observed,n_unknown_chapters,n_chapters_full_components,n_known_chapters,
          n_catalog_chapters,known_catalog_coverage,
          COELO_mean_chapter,AFELO_mean_chapter,ACELO_mean_chapter,
          FV_mean_chapter,FA_mean_chapter,KL_mean_chapter,CFA_mean_chapter,
          n_video_events_sum_observed,n_problem_events_sum_observed,
          n_problems_attempted_sum_observed,n_problems_correct_sum_observed,
          total_problem_attempts_sum_observed,n_comments_all_sum_observed,
          n_sentiment_comments_sum_observed,n_positive_comments_sum_observed,
          n_negative_comments_sum_observed,n_neutral_comments_sum_observed,
          n_replies_authored_sum_observed,n_visits_sum_observed,
          min_FV_positive_support,min_IF_positive_support,min_FA_nonzero_support,
          n_FV_small_support_chapters,n_FV_saturated_chapters,n_FA_chapters,
          n_FA_nonzero_chapters,n_FA_saturated_positive_chapters,
          enroll_time,course_end,first_observed_event,last_observed_event
          FROM enriched WHERE y_alpha_0p1='E' OR y_alpha_1='E'
          ORDER BY evidence_cohort,q_alpha_0p1,user_id,course_id''', retain=False)
        self.save('e_chapter_details', '''SELECT c.* FROM chapters c JOIN e_keys USING(user_id,course_id)
          ORDER BY user_id,course_id,chapter_major,lo_key''', retain=False)
        self.save('e_sentiment_summary', '''SELECT e.evidence_cohort,s.predicted_label,
          count(*) n_comments,count(DISTINCT(s.user_id,s.course_id)) n_pairs,
          avg(s.confidence) mean_confidence,
          count(*) FILTER(WHERE s.confidence<0.6) n_confidence_lt_0p6,
          count(*) FILTER(WHERE s.was_truncated) n_truncated,
          count(*) FILTER(WHERE s.has_cjk) n_with_cjk,
          median(s.n_tokens) median_tokens
          FROM sentiment s JOIN e_cases e USING(user_id,course_id)
          GROUP BY 1,2 ORDER BY 1,2''')
        # Fixed representative cases: two lowest-q, two near .2, four hash-selected per cohort.
        self.save('e_review_cases', '''WITH ranked AS (SELECT *,
          row_number() OVER(PARTITION BY evidence_cohort ORDER BY least(q_alpha_0p1,q_alpha_1),user_id,course_id) low_rank,
          row_number() OVER(PARTITION BY evidence_cohort ORDER BY least(q_alpha_0p1,q_alpha_1) DESC,user_id,course_id) edge_rank,
          row_number() OVER(PARTITION BY evidence_cohort ORDER BY md5(user_id||'/'||course_id)) hash_rank
          FROM e_cases) SELECT * FROM ranked WHERE low_rank<=2 OR edge_rank<=2 OR hash_rank<=4
          ORDER BY evidence_cohort,low_rank''')
        self.save('e_review_comments', '''WITH ranked AS (SELECT s.user_id,s.course_id,
          e.evidence_cohort,s.lo_key,s.comment_id,s.event_time,s.predicted_label,
          s.p_pos,s.p_neu,s.p_neg,s.confidence,s.n_tokens,s.was_truncated,
          left(l.text,800) text_excerpt,
          row_number() OVER(PARTITION BY s.user_id,s.course_id ORDER BY s.confidence,s.comment_id) comment_rank
          FROM sentiment s JOIN e_review_cases e USING(user_id,course_id)
          JOIN comment_links l USING(comment_id))
          SELECT * FROM ranked WHERE comment_rank<=3 ORDER BY evidence_cohort,user_id,course_id,comment_rank''')
        self.save('e_threshold_margins', ' UNION ALL '.join(f'''
          SELECT {qs(s)} alpha,
            count(*) n_E,count(*) FILTER(WHERE q_alpha_{s}>=0.19) n_within_0p01,
            count(*) FILTER(WHERE q_alpha_{s}>=0.199) n_within_0p001,
            count(*) FILTER(WHERE abs(q_alpha_{s}-0.2)<1e-12) n_float_boundary_1e12,
            min(0.2-q_alpha_{s}) min_threshold_margin,
            max(q_alpha_{s}) max_E_q FROM labeled WHERE y_alpha_{s}='E' ''' for s, _ in ALPHAS))

    def peer_audit(self):
        # P95(|FA_raw|)*chapter_total equals P95(|positive-negative|).
        self.con.execute('''CREATE TEMP TABLE peer_groups AS SELECT course_id,lo_key,
          count(*) n_observed_users,count(*) FILTER(WHERE n_comments_all>0) n_comment_users,
          count(*) FILTER(WHERE FA IS NOT NULL) n_sentiment_users,
          count(*) FILTER(WHERE FA_raw<>0) n_nonzero_net_users,
          count(*) FILTER(WHERE n_positive_comments>n_negative_comments) n_positive_net_users,
          sum(n_comments_all) total_peer_comments,sum(n_visits) total_peer_visits,
          sum(n_positive_comments-n_negative_comments) total_peer_net_comments,
          median(n_comments_all) peer_median_comments,median(n_visits) peer_median_visits,
          median(n_comments_all) FILTER(WHERE n_comments_all>0) commenter_median_comments,
          quantile_cont(abs(n_positive_comments-n_negative_comments),0.95)
            FILTER(WHERE FA_raw<>0) p95_abs_net_comments_direct
          FROM chapters GROUP BY 1,2''')
        self.con.execute('''CREATE TEMP TABLE peer_ranks AS SELECT user_id,course_id,lo_key,
          cume_dist() OVER(PARTITION BY course_id,lo_key ORDER BY n_visits) visits_cdf_observed,
          cume_dist() OVER(PARTITION BY course_id,lo_key ORDER BY n_comments_all) comments_cdf_observed
          FROM chapters''')
        self.con.execute('''CREATE TEMP TABLE text_repetition AS SELECT user_id,course_id,lo_key,
          count(*) n_inferred_comment_rows,count(DISTINCT text_hash) n_distinct_inferred_texts,
          count(*) FILTER(WHERE predicted_label='positive') n_positive_rows,
          count(DISTINCT text_hash) FILTER(WHERE predicted_label='positive') n_distinct_positive_texts,
          count(*) FILTER(WHERE n_tokens<=5) n_short_token_comments,
          count(*) FILTER(WHERE confidence<0.6) n_low_confidence_comments
          FROM sentiment GROUP BY 1,2,3''')
        self.save('e_chapter_peer_comparison', '''SELECT c.user_id,c.course_id,c.lo_key,
          e.evidence_cohort,e.y_alpha_0p1,e.y_alpha_1,e.q_alpha_0p1,e.q_alpha_1,
          c.chapter_major,c.component_mask,c.n_comments_all,c.n_sentiment_comments,
          c.n_positive_comments,c.n_negative_comments,c.n_neutral_comments,
          c.n_positive_comments-c.n_negative_comments net_positive_comments,
          c.n_visits,c.n_replies_authored,c.FV,c.FA,c.FA_raw,c.AFELO,c.COELO,c.ACELO,
          c.n_comments_chapter_total,c.p95_n_visits,c.n_users_visits_positive,
          c.p95_abs_FA_raw,c.n_users_fa_nonzero,
          c.p95_abs_FA_raw*c.n_comments_chapter_total p95_abs_net_comments_from_FA,
          p.* EXCLUDE(course_id,lo_key),r.* EXCLUDE(user_id,course_id,lo_key),
          coalesce(t.n_distinct_inferred_texts,0) n_distinct_inferred_texts,
          coalesce(t.n_distinct_positive_texts,0) n_distinct_positive_texts,
          coalesce(t.n_short_token_comments,0) n_short_token_comments,
          coalesce(t.n_low_confidence_comments,0) n_low_confidence_comments,
          cs.n_eligible_pairs course_eligible_pairs,
          cs.n_eligible_pairs-p.n_observed_users course_pairs_without_this_chapter,
          p.n_observed_users-1 n_other_observed_users,
          (p.total_peer_comments-c.n_comments_all)::DOUBLE/nullif(p.n_observed_users-1,0) other_users_mean_comments,
          (p.total_peer_visits-c.n_visits)::DOUBLE/nullif(p.n_observed_users-1,0) other_users_mean_visits,
          (p.total_peer_net_comments-(c.n_positive_comments-c.n_negative_comments))::DOUBLE/
            nullif(p.n_observed_users-1,0) other_users_mean_net_comments
          FROM e_chapter_details c JOIN e_cases e USING(user_id,course_id)
          JOIN peer_groups p USING(course_id,lo_key)
          JOIN peer_ranks r USING(user_id,course_id,lo_key)
          JOIN course_summary cs USING(course_id)
          LEFT JOIN text_repetition t USING(user_id,course_id,lo_key)
          ORDER BY e.evidence_cohort,c.user_id,c.course_id,c.lo_key''', retain=False)
        self.save('e_unknown_mechanisms', '''SELECT count(*) n_E_unknown,
          median(n_comments_all) median_comments,median(n_sentiment_comments) median_inferred_comments,
          median(n_distinct_inferred_texts) median_distinct_texts,
          median(n_visits) median_visits,median(net_positive_comments) median_net_positive,
          median(peer_median_comments) median_of_peer_median_comments,
          median(peer_median_visits) median_of_peer_median_visits,
          median(other_users_mean_comments) median_other_mean_comments,
          median(other_users_mean_visits) median_other_mean_visits,
          count(*) FILTER(WHERE n_comments_all=1 AND n_visits=1) n_one_comment_one_visit,
          count(*) FILTER(WHERE n_comments_all<=1) n_at_most_one_comment,
          count(*) FILTER(WHERE n_comments_all>=6) n_at_least_six_comments,
          count(*) FILTER(WHERE n_visits>=6) n_at_least_six_visits,
          count(*) FILTER(WHERE FA IS NULL) n_no_FA,
          count(*) FILTER(WHERE FA IS NULL AND n_comments_all=0 AND n_replies_authored>0) n_reply_only,
          count(*) FILTER(WHERE FA=1) n_FA_one,count(*) FILTER(WHERE FV=1) n_FV_one,
          count(*) FILTER(WHERE FA=1 AND FV=1) n_both_one,
          count(*) FILTER(WHERE n_users_visits_positive<=5) n_FV_support_le5,
          count(*) FILTER(WHERE FA IS NOT NULL AND n_users_fa_nonzero<=5) n_FA_support_le5,
          count(*) FILTER(WHERE p95_n_visits<=1+1e-12) n_P95_visits_le1,
          count(*) FILTER(WHERE FA IS NOT NULL AND p95_abs_net_comments_direct<=1+1e-12) n_P95_net_le1,
          median(n_users_visits_positive) median_FV_support,
          median(n_users_fa_nonzero) FILTER(WHERE FA IS NOT NULL) median_FA_support,
          median(p95_n_visits) median_P95_visits,
          median(p95_abs_net_comments_direct) FILTER(WHERE FA IS NOT NULL) median_P95_abs_net,
          count(*) FILTER(WHERE n_sentiment_comments>n_distinct_inferred_texts) n_repeated_text,
          sum(n_sentiment_comments) n_sentiment_rows,sum(n_distinct_inferred_texts) n_distinct_user_texts,
          count(*) FILTER(WHERE n_visits>peer_median_visits) n_visits_above_peer_median,
          count(*) FILTER(WHERE n_comments_all>peer_median_comments) n_comments_above_peer_median,
          median(course_eligible_pairs) median_eligible_course_pairs,
          median(n_observed_users) median_observed_users_same_chapter
          FROM e_chapter_peer_comparison WHERE evidence_cohort='UNKNOWN_ONLY' ''')
        self.save('e_unknown_intensity_bins', ' UNION ALL '.join(f'''
          SELECT {qs(metric)} metric,CASE WHEN {metric}=0 THEN '0' WHEN {metric}=1 THEN '1'
            WHEN {metric}<=5 THEN '2-5' WHEN {metric}<=20 THEN '6-20' ELSE '>20' END intensity,
            count(*) n_E_unknown,avg(FV) mean_FV,avg(FA) mean_FA,avg(q_alpha_0p1) mean_q,
            median(n_users_visits_positive) median_FV_support,
            median(p95_n_visits) median_P95_visits
          FROM e_chapter_peer_comparison WHERE evidence_cohort='UNKNOWN_ONLY' GROUP BY 2'''
          for metric in ['n_comments_all','n_visits','net_positive_comments']))
        self.save('e_scale_examples', '''WITH groups AS (SELECT *,
          CASE WHEN n_comments_all=1 AND n_visits=1 THEN 'ONE_COMMENT_ONE_VISIT'
            WHEN FA IS NULL THEN 'FV_ONLY'
            WHEN n_comments_all>=6 AND n_visits>=6 THEN 'MANY_COMMENTS_AND_VISITS'
            ELSE 'OTHER' END mechanism FROM e_chapter_peer_comparison WHERE evidence_cohort='UNKNOWN_ONLY'),
          ranked AS (SELECT *,row_number() OVER(PARTITION BY mechanism ORDER BY n_observed_users DESC,
            n_comments_all DESC,user_id,course_id) example_rank FROM groups)
          SELECT * FROM ranked WHERE example_rank<=3 ORDER BY mechanism,example_rank''')
        self.save('e_peer_comparison_by_cohort', '''SELECT evidence_cohort,count(*) n_chapter_rows,
          median(n_comments_all) median_comments,median(n_visits) median_visits,
          median(peer_median_comments) median_peer_comments,median(peer_median_visits) median_peer_visits,
          count(*) FILTER(WHERE n_comments_all>peer_median_comments) n_comments_above_peer_median,
          count(*) FILTER(WHERE n_visits>peer_median_visits) n_visits_above_peer_median,
          count(*) FILTER(WHERE n_users_visits_positive<=5) n_FV_support_le5,
          count(*) FILTER(WHERE FA IS NOT NULL AND n_users_fa_nonzero<=5) n_FA_support_le5
          FROM e_chapter_peer_comparison GROUP BY 1''')
        self.save('e_float_boundary_cases', ' UNION ALL '.join(f'''
          SELECT {qs(s)} alpha,user_id,course_id,evidence_cohort,q_alpha_{s} q,
            round(q_alpha_{s},12) rounded_q_12,
            CASE WHEN round(q_alpha_{s},12)<0.2 THEN 'E' ELSE 'G' END label_after_round12
          FROM e_cases WHERE y_alpha_{s}='E' AND abs(q_alpha_{s}-0.2)<1e-12'''
          for s,_ in ALPHAS))

    def validate(self):
        con = self.con
        pop = self.results['population'][0]
        e_rows = records(con, 'SELECT * FROM e_cases')
        detail = records(con, '''SELECT user_id,course_id,COELO,AFELO,ACELO,
          LOQV_alpha_0p1,LOQV_alpha_1 FROM e_chapter_details''')
        groups = defaultdict(lambda: {'0p1': [], '1': []})
        chapter_error = 0.0
        for r in detail:
            key = (r['user_id'], r['course_id'])
            for s, alpha in ALPHAS:
                available = [(r[m], w) for m, w in [('COELO', 1), ('AFELO', alpha), ('ACELO', 1)]
                             if r[m] is not None]
                value = math.sqrt(math.fsum(w * (1 - x) ** 2 for x, w in available) /
                                  math.fsum(w for _, w in available))
                chapter_error = max(chapter_error, abs(value - r['LOQV_alpha_' + s]))
                groups[key][s].append(value)
        pair_error = 0.0
        thresholds_ok = True
        for r in e_rows:
            for s, _ in ALPHAS:
                values = groups[(r['user_id'], r['course_id'])][s]
                q = math.fsum(values) / len(values)
                pair_error = max(pair_error, abs(q - r['q_alpha_' + s]))
                thresholds_ok &= label(q) == r['y_alpha_' + s]
        self.results['formula_errors'] = {'max_chapter_error': chapter_error,
                                           'max_pair_error': pair_error}
        checks = {
            'population_matches_source': pop['n_eligible_pairs'] == self.source_summary['times']['eligible_pairs'],
            'labeled_count_matches_source': pop['n_labeled_pairs'] == self.source_summary['chapters']['observed_pairs'],
            'split_totals_reconcile': sum(r['n_eligible_pairs'] for r in self.results['split_summary']) == pop['n_eligible_pairs'],
            'mask_counts_reconcile': sum(r['n_chapters'] for r in self.results['component_masks']) == self.source_summary['chapters']['rows'],
            'metrics_finite_and_bounded': all(r['n_nonfinite'] == 0 and
                (r['minimum'] is None or r['minimum'] >= -1e-12) and
                (r['maximum'] is None or r['maximum'] <= 1+1e-12) for r in self.results['metric_profile']),
            'alpha_transition_reconciles': sum(r['n_pairs'] for r in self.results['alpha_transition']) == pop['n_labeled_pairs'],
            'all_E_cases_unique': len(e_rows) == len({(r['user_id'], r['course_id']) for r in e_rows}),
            'all_E_cases_traced': len(e_rows) == len(groups),
            'E_chapter_formula_independent_python': chapter_error < 1e-12,
            'E_pair_q_independent_python': pair_error < 1e-12,
            'E_labels_independent_python': thresholds_ok,
            'E_unknown_only_has_no_catalog_or_C_A': con.execute('''SELECT count(*)=0 FROM e_chapter_details
              WHERE chapter_major=-1 AND (COELO IS NOT NULL OR ACELO IS NOT NULL OR has_catalog)''').fetchone()[0],
            'E_unknown_only_alpha_cancels': all(abs(r['q_alpha_0p1']-r['q_alpha_1']) < 1e-12
                for r in e_rows if r['evidence_cohort'] == 'UNKNOWN_ONLY'),
            'E_problem_counts_consistent': con.execute('''SELECT count(*)=0 FROM e_chapter_details
              WHERE n_problems_correct>n_problems_attempted OR n_problems_attempted>n_problems_available
                OR invalid_correctness>0 OR invalid_attempts>0 OR correct_invalid_attempts>0''').fetchone()[0],
            'E_observed_window_consistent': con.execute('''SELECT count(*)=0 FROM e_chapter_details
              WHERE first_observed_event<enroll_time OR last_observed_event>course_end''').fetchone()[0],
            'known_catalog_coverage_bounded': con.execute('''SELECT count(*)=0 FROM enriched
              WHERE known_catalog_coverage<0 OR known_catalog_coverage>1''').fetchone()[0],
            'peer_join_preserves_all_E_chapters': con.execute('''SELECT
              (SELECT count(*) FROM e_chapter_peer_comparison)=(SELECT count(*) FROM e_chapter_details)''').fetchone()[0],
            'FA_chapter_denominator_cancels': con.execute('''SELECT coalesce(max(abs(
              p95_abs_net_comments_direct-p95_abs_net_comments_from_FA)),0)<1e-9
              FROM e_chapter_peer_comparison''').fetchone()[0],
            'FA_recomputed_from_net_counts': con.execute('''SELECT coalesce(max(abs(FA-
              CASE WHEN net_positive_comments=0 THEN 0.5 ELSE
              0.5+0.5*least(1.0,greatest(-1.0,net_positive_comments::DOUBLE/nullif(p95_abs_net_comments_direct,0))) END)),0)<1e-12
              FROM e_chapter_peer_comparison WHERE FA IS NOT NULL''').fetchone()[0],
            'FV_reference_excludes_nonobserved_zero_users': con.execute('''SELECT count(*)=0
              FROM e_chapter_peer_comparison WHERE n_users_visits_positive<>n_observed_users''').fetchone()[0],
            'FA_reference_nonzero_counts_reconcile': con.execute('''SELECT count(*)=0
              FROM e_chapter_peer_comparison WHERE n_users_fa_nonzero<>n_nonzero_net_users''').fetchone()[0],
            'unknown_E_threshold_matches_FV_FA': con.execute('''SELECT count(*)=0
              FROM e_chapter_peer_comparison WHERE evidence_cohort='UNKNOWN_ONLY'
              AND (abs(q_alpha_0p1-(1-CASE WHEN FA IS NULL THEN FV ELSE (FV+FA)/2 END))>1e-12
                OR (FA IS NULL AND FV<0.8-1e-12) OR (FA IS NOT NULL AND FV+FA<1.6-1e-12))''').fetchone()[0],
        }
        for s, _ in ALPHAS:
            expected = next(r['n'] for r in self.source_summary['label_counts'][s] if r['y'] == 'E')
            checks[f'all_E_{s}_included'] = sum(r['y_alpha_' + s] == 'E' for r in e_rows) == expected
            checks[f'E_counterfactual_partition_{s}'] = all(
                r['n_E'] == r['n_E_on_known_only']+r['n_no_known_evidence']+r['n_downgraded_on_known_only'] and
                r['n_E'] == r['n_E_on_full_only']+r['n_no_full_evidence']+r['n_downgraded_on_full_only']
                for r in self.results['e_cohort_summary'] if r['alpha'] == s)
        checks['source_files_unchanged'] = all(
            Path(r['path']).stat().st_size == r['bytes'] and
            Path(r['path']).stat().st_mtime_ns == r['mtime_ns'] for r in self.inventory)
        dump(self.out / 'validation.json', checks)
        if not all(checks.values()):
            raise AssertionError([k for k, v in checks.items() if not v])
        self.checks = checks

    def figures(self):
        os.environ.setdefault('MPLCONFIGDIR', str(self.out / '_scratch/matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
        colors = ['#146C94', '#E99B32', '#7D4C92', '#4E9F81']

        def finish(fig, name):
            fig.tight_layout()
            fig.savefig(self.out / 'figures' / (name+'.png'), dpi=160)
            fig.savefig(self.out / 'figures' / (name+'.svg'))
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        rows = self.results['split_summary']
        axes[0].bar([r['split'] for r in rows], [r['pct_labeled'] for r in rows], color=colors[:3])
        axes[0].set(title='Label coverage among eligible pairs', ylabel='% eligible pairs with a label')
        for i, r in enumerate(rows):
            axes[0].text(i, r['pct_labeled']+.05, f"{r['pct_labeled']:.2f}%", ha='center')
        for j, (s, _) in enumerate(ALPHAS):
            n = [sum(r['n_pairs'] for r in self.results['label_distribution'] if r['alpha']==s and r['y']==y) for y in LABELS]
            axes[1].bar(np.arange(3)+(j-.5)*.35, 100*np.array(n)/sum(n), width=.35, color=colors[j], label='alpha='+s.replace('p','.'))
        axes[1].set(xticks=range(3), xticklabels=LABELS, title='Labels among labeled pairs only', ylabel='% labeled pairs')
        axes[1].legend()
        finish(fig, '01_population_labels')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        core = ['COELO','AFELO','ACELO','FA','CFA']
        prof = {r['metric']: r for r in self.results['metric_profile'] if r['cohort']=='ALL_OBSERVED'}
        axes[0].barh(core, [pct(prof[m]['n_null'],prof[m]['n_rows']) for m in core], color=colors[0])
        axes[0].set(title='Missing components across observed chapters', xlabel='% chapter rows NULL', xlim=(0,100))
        masks = self.results['component_masks']
        axes[1].bar([r['component_mask'] for r in masks], [r['n_chapters'] for r in masks], color=colors[:len(masks)])
        axes[1].set(title='Component availability', ylabel='Chapter rows', xlabel='C=COELO, F=AFELO, A=ACELO')
        finish(fig, '02_component_coverage')

        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        for ax, metrics in zip(axes, [['COELO','AFELO','ACELO'], ['q_alpha_0p1','q_alpha_1']]):
            for j,m in enumerate(metrics):
                rows = [r for r in self.results['histograms'] if r['metric']==m]
                n = sum(r['n_rows'] for r in rows)
                vals = {r['bin']:pct(r['n_rows'],n) for r in rows}
                ax.step((np.arange(50)+.5)/50,[vals.get(b,0) for b in range(50)],where='mid',label=m,color=colors[j])
            ax.set(xlabel='Value',ylabel='% non-NULL rows per 0.02 bin',xlim=(0,1))
            ax.legend()
        axes[0].set_title('Chapter indices'); axes[1].set_title('Pair q: smaller is better')
        axes[1].axvline(.2,color='#555',ls='--',lw=1);axes[1].axvline(.4,color='#555',ls=':',lw=1)
        finish(fig, '03_distributions')

        fig, axes = plt.subplots(1, 2, figsize=(12,4.7))
        cohorts=['UNKNOWN_ONLY','NO_FULL','MIXED','ALL_FULL']
        for j,(s,_) in enumerate(ALPHAS):
            rows={r['evidence_cohort']:r for r in self.results['e_cohort_summary'] if r['alpha']==s}
            axes[0].barh(np.arange(4)+(j-.5)*.34,[rows.get(c,{}).get('n_E',0) for c in cohorts],height=.34,color=colors[j],label='alpha='+s.replace('p','.'))
            axes[1].bar(np.arange(4)+(j-.5)*.34,[sum(r[k] for r in rows.values()) for k in
                ['n_E','n_no_full_evidence','n_downgraded_on_full_only','n_E_on_full_only']],width=.34,color=colors[j],label='alpha='+s.replace('p','.'))
        axes[0].set(yticks=range(4),yticklabels=cohorts,title='E labels by evidence cohort',xlabel='Pairs')
        axes[1].set(xticks=range(4),xticklabels=['All E','No full\nevidence','Full-only\nq >= 0.2','Still E\nfull-only'],title='E sensitivity: retain complete chapters only',ylabel='Pairs')
        axes[0].legend();axes[1].legend()
        finish(fig, '04_E_evidence_audit')

        matrix=np.zeros((3,3),dtype=int)
        for r in self.results['alpha_transition']:
            matrix[LABELS.index(r['y_alpha_0p1']),LABELS.index(r['y_alpha_1'])]=r['n_pairs']
        fig,ax=plt.subplots(figsize=(6,5))
        ax.imshow(np.log1p(matrix),cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='white' if matrix[i,j]>20000 else 'black')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,
               xlabel='alpha = 1',ylabel='alpha = 0.1',title='Label transitions (color = log count)')
        finish(fig,'05_alpha_transition')

        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for j,(s,_) in enumerate(ALPHAS):
            rr=sorted([r for r in self.results['e_by_course'] if r['alpha']==s],key=lambda r:-r['n_E'])
            counts=np.array([r['n_E'] for r in rr])
            axes[0].plot(np.arange(1,len(rr)+1),100*np.cumsum(counts)/counts.sum(),label='alpha='+s.replace('p','.'),color=colors[j])
        axes[0].set(title='Concentration of E labels across courses',xlabel='Courses ranked by E count',ylabel='Cumulative % E pairs',ylim=(0,102));axes[0].legend()
        for j,metric in enumerate(['TS_video','IF','FV','FA']):
            rr={r['positive_support']:r for r in self.results['p95_support'] if r['metric']==metric}
            denom=sum(r['n_reference_groups'] for r in rr.values())
            axes[1].bar(np.arange(5)+(j-1.5)*.18,[pct(rr.get(b,{}).get('n_reference_groups',0),denom) for b in ['0','1','2-5','6-20','>20']],width=.18,label=metric,color=colors[j])
        axes[1].set(xticks=range(5),xticklabels=['0','1','2-5','6-20','>20'],title='P95 support across all reference groups',xlabel='Positive/nonzero users in reference',ylabel='% reference groups');axes[1].legend()
        finish(fig,'06_course_concentration_p95')

        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        bins=['0','1','2-5','6-20','>20']
        for j,metric in enumerate(['n_comments_all','n_visits']):
            rr={r['intensity']:r['n_E_unknown'] for r in self.results['e_unknown_intensity_bins'] if r['metric']==metric}
            axes[0].bar(np.arange(5)+(j-.5)*.36,[rr.get(b,0) for b in bins],width=.36,
                        label='Comments' if j==0 else 'Visits',color=colors[j])
        axes[0].set(xticks=range(5),xticklabels=bins,title='Absolute activity: E with unknown chapter only',
                    xlabel='Count per user-course',ylabel='E pairs');axes[0].legend()
        example=next(r for r in self.results['e_scale_examples'] if r['mechanism']=='ONE_COMMENT_ONE_VISIT')
        names=['FV','FA','AFELO','q']
        vals=[example['FV'],example['FA'],example['AFELO'],example['q_alpha_0p1']]
        axes[1].bar(names,vals,color=[colors[0],colors[1],colors[3],colors[2]])
        axes[1].axhline(.2,color='#555',ls=':',lw=1)
        for i,v in enumerate(vals):axes[1].text(i,v+.025,f'{v:.4f}',ha='center')
        axes[1].set(ylim=(0,1.12),ylabel='Normalized value',title='One comment + one visit can receive E')
        axes[1].text(.5,-.18,f"{example['course_id']}: {example['n_observed_users']} observed users; "
                     f"P95 visits={example['p95_n_visits']:.2f}; P95 |net comments|={example['p95_abs_net_comments_direct']:.2f}",
                     ha='center',transform=axes[1].transAxes,fontsize=9)
        finish(fig,'07_E_absolute_vs_relative')

    def report(self):
        r=self.results; pop=r['population'][0]
        lines=['# EDA final quality v2 — kiểm tra kỹ các mẫu nhãn E', '',
          'Ngày phân tích: 20/10/2026. Tất cả số liệu tính trên parquet hoàn tất; không lấy mẫu cho thống kê. '
          'Đơn vị nhãn là **user–course**; đơn vị chỉ số là **user–course–chapter**.', '',
          f"Có **{pop['n_labeled_pairs']:,}/{pop['n_eligible_pairs']:,}** cặp có nhãn "
          f"(**{pop['pct_labeled']:.4f}%**). {pop['n_unlabeled_pairs']:,} cặp không có nhãn không được gán A hoặc E.", '',
          '## Kết luận chính về E', '',
          'E đúng ngưỡng q<0,2 theo công thức hiện hành, nhưng độ đầy đủ bằng chứng khác nhau rõ rệt. '
          'Chapter -1 không có catalog nên COELO và ACELO là NULL; LOQV chỉ còn **1−AFELO**, alpha triệt tiêu. '
          'Vì vậy E trong nhóm này thể hiện AFELO cao; không đủ bằng chứng để diễn giải là học tốt trên cả ba mặt.', '']
        for s,_ in ALPHAS:
            rows=[x for x in r['e_cohort_summary'] if x['alpha']==s]
            total=sum(x['n_E'] for x in rows)
            unknown=next(x for x in rows if x['evidence_cohort']=='UNKNOWN_ONLY')
            full=next(x for x in rows if x['evidence_cohort']=='ALL_FULL')
            retain=sum(x['n_E_on_full_only'] for x in rows)
            lines += [f"- **Alpha {s.replace('p','.')}**: {total:,} mẫu E; "
              f"**{unknown['n_E']:,} ({pct(unknown['n_E'],total):.2f}%) chỉ có chapter -1**; "
              f"{full['n_E']:,} ({pct(full['n_E'],total):.2f}%) có đủ ba chỉ số ở mọi chapter quan sát. "
              f"Khi chỉ giữ chapter đủ ba chỉ số, {retain:,} mẫu E ban đầu còn E."]
        lines += ['', 'Các phép “chỉ giữ chương đã biết” và “chỉ giữ chương đủ ba chỉ số” là **phân tích độ nhạy**, '
          'không sửa nhãn gốc. Không có chương phù hợp → q=NULL, không phải chuyển thành A.', '',
          *self.peer_report(),
          '## Bằng chứng từng nhóm E', '',
          'UNKNOWN_ONLY: tất cả chapter đều -1. ALL_FULL: mọi chapter quan sát có COELO/AFELO/ACELO. '
          'NO_FULL: không có chapter đủ ba chỉ số nhưng có chapter đã biết. MIXED: phần còn lại.', '',
          table(['Alpha','Nhóm','E','1 chapter','Chỉ social','Không sentiment','q=0','Có FV support≤5','Median visits','Median catalog coverage'],
            [[x[k] for k in ['alpha','evidence_cohort','n_E','n_single_chapter','n_social_only','n_without_sentiment','n_q_zero','n_any_FV_support_le5','median_visits','median_known_catalog_coverage']]
             for x in r['e_cohort_summary']]), '',
          '“Chỉ social” nghĩa là không có event video/problem trong bảng cuối, vẫn có comment/reply. '
          'Catalog coverage = số chapter đã biết có log / số chapter catalog của course; chapter -1 không tính vào tử số. '
          'Nhãn hiện chỉ dùng chương quan sát, nên E từ một chương chưa chứng minh hoàn thành toàn khóa.', '',
          table(['Alpha','Nhóm','E gốc','E khi bỏ -1','Thiếu chương đã biết','q_known≥0,2','E đủ 3 chỉ số','Thiếu chương đủ 3','q_full≥0,2'],
            [[x[k] for k in ['alpha','evidence_cohort','n_E','n_E_on_known_only','n_no_known_evidence','n_downgraded_on_known_only',
                              'n_E_on_full_only','n_no_full_evidence','n_downgraded_on_full_only']] for x in r['e_cohort_summary']]), '',
          '![E evidence](figures/04_E_evidence_audit.png)', '',
          '## Đã kiểm tra từng mẫu E như thế nào', '',
          f"Truy vết **{self.con.execute('SELECT count(*) FROM e_cases').fetchone()[0]:,} cặp thuộc hợp của E hai alpha** "
          f"và **{self.con.execute('SELECT count(*) FROM e_chapter_details').fetchone()[0]:,} chapter** tương ứng. "
          'Tính lại khoảng cách từng chương bằng Python độc lập từ COELO/AFELO/ACELO, sau đó lấy trung bình chương và áp ngưỡng nhãn.', '',
          f"Sai số tối đa LOQV={r['formula_errors']['max_chapter_error']:.3g}; "
          f"q={r['formula_errors']['max_pair_error']:.3g}. Đã đối chiếu counts bài tập, "
          'giá trị NULL/0, mốc event, support P95 và đủ keys. Đây là kiểm tra tính nhất quán dữ liệu/công thức, '
          'không phải ground truth chất lượng học tập.', '',
          table(['Alpha','E','q≥0,19','q≥0,199','|q−0,2|<1e−12','Khoảng cách nhỏ nhất tới 0,2','q lớn nhất'],
            [[x['alpha'],x['n_E'],x['n_within_0p01'],x['n_within_0p001'],x['n_float_boundary_1e12'],
              f"{x['min_threshold_margin']:.3g}",repr(x['max_E_q'])] for x in r['e_threshold_margins']]), '',
          '**Phát hiện sát ngưỡng số thực:** có các giá trị q=0.19999999999999996 được so sánh trực tiếp '
          'với 0.2 nên nhận E, dù làm tròn 12 chữ số sẽ thành 0.2 và nhận G. '
          'Các phép như FV=0.6, FA=1 cho AFELO=0.8 là trường hợp điển hình. '
          'Bảng [e_float_boundary_cases.csv](tables/e_float_boundary_cases.csv) liệt kê các mẫu nhạy với sai số này; '
          'EDA giữ nguyên nhãn gốc, chưa áp chính sách làm tròn/tolerance.', '',
          'Bảng truy vết đầy đủ:', '',
          '- [e_cases.csv](tables/e_cases.csv): tất cả mẫu E, nhãn hai alpha, cohort, q phản thực, số chương, '
          'catalog coverage, chỉ số, hoạt động, sentiment counts, support P95 và thời gian.',
          '- [e_chapter_details.parquet](tables/e_chapter_details.parquet): toàn bộ thành phần và mẫu số theo chương cho mọi mẫu E.',
          '- [e_review_cases.csv](tables/e_review_cases.csv): mẫu đọc thủ công cố định theo nhóm, gồm q thấp, sát ngưỡng và hash chọn cố định.',
          '- [e_review_comments.csv](tables/e_review_comments.csv): tối đa 3 comment/cặp trong mẫu đọc, ưu tiên confidence thấp; '
          'có xác suất và trích đoạn tối đa 800 ký tự. Đây là mẫu chọn có chủ đích, không đại diện ngẫu nhiên toàn bộ E.', '',
          '## Sentiment của hợp mẫu E', '',
          table(['Nhóm','Sentiment','Comment','Cặp có comment','Confidence TB','Confidence<0,6','Truncated','Median tokens'],
            [[x[k] for k in ['evidence_cohort','predicted_label','n_comments','n_pairs','mean_confidence','n_confidence_lt_0p6','n_truncated','median_tokens']]
              for x in r['e_sentiment_summary']]), '',
          'FA dùng argmax sentiment của comment, không dùng confidence làm trọng số. '
          'Không có sentiment vẫn có thể có AFELO từ FV. Confidence cao không xác nhận ý nghĩa sư phạm; '
          'chưa có nhãn sentiment do người đánh giá để đo độ chính xác.', '',
          '## Quy mô, split và phân bố nhãn', '',
          table(['Split','Đủ thời gian','Khóa','Có nhãn','% có nhãn','q TB α0,1','q TB α1','E α0,1','E α1'],
            [[x[k] for k in ['split','n_eligible_pairs','n_courses','n_labeled_pairs','pct_labeled','mean_q_0p1','mean_q_1','n_E_0p1','n_E_1']] for x in r['split_summary']]), '',
          '![Population](figures/01_population_labels.png)', '',
          '## Độ phủ và phân bố chỉ số theo chương', '',
          table(['Mask','Chapter -1','Chương','User–course','COELO TB','AFELO TB','ACELO TB','Có FA'],
            [[x[k] for k in ['component_mask','is_unknown','n_chapters','n_pairs','mean_COELO','mean_AFELO','mean_ACELO','n_with_FA']] for x in r['component_masks']]), '',
          'Một cặp có thể đóng góp nhiều mask; số cặp giữa các hàng không cộng được. '
          'CFA=NULL khi không có bài đúng khác với ACELO=0 khi có catalog nhưng không có bài đúng.', '',
          table(['Chỉ số','Có giá trị','NULL','Bằng 0','Bằng 1','Mean','Median','P95'],
            [[x[k] for k in ['metric','n_nonnull','n_null','n_zero','n_one','mean','median','p95']] for x in r['metric_profile'] if x['cohort']=='ALL_OBSERVED']), '',
          '![Coverage](figures/02_component_coverage.png)', '',
          '![Distributions](figures/03_distributions.png)', '',
          '## Độ nhạy alpha và P95', '',
          table(['Nhãn α0,1','Nhãn α1','User–course'],[[x['y_alpha_0p1'],x['y_alpha_1'],x['n_pairs']] for x in r['alpha_transition']]), '',
          '![Transitions](figures/05_alpha_transition.png)', '',
          'P95 được fit trong cohort cuối kỳ theo công thức v2. Support là số user có giá trị dương '
          '(FA dùng trị tuyệt đối khác 0); nhóm singleton có P95 bằng chính giá trị đó. '
          'Không đồng nhất support nhỏ với nhãn sai; cần đọc bằng chứng từng trường hợp. '
          'Bảng e_cases ghi support tối thiểu để lọc các mẫu cần xem kỹ.', '',
          '![P95](figures/06_course_concentration_p95.png)', '',
          '## Diễn giải và sử dụng', '',
          '- Báo cáo E theo cohort bằng chứng, kèm số chương và catalog coverage; '
          'đặc biệt tách UNKNOWN_ONLY khi diễn giải kết quả học tập.',
          '- Khi đánh giá mô hình, báo cáo riêng tập đủ ba chỉ số và các phân tích độ nhạy. '
          'Việc đặt điều kiện coverage tối thiểu là thay đổi chính sách nhãn, cần quyết định riêng.',
          '- Split theo course không đảm bảo user tách biệt hoặc mọi khóa train kết thúc trước test. '
          'Độ phủ nhãn thấp làm tập có nhãn khác toàn bộ enrollment.',
          '- Tương quan chỉ số với q là quan hệ cấu thành nhãn, không phải bằng chứng mô hình dự báo. '
          'Các số cuối kỳ, chỉ số nhãn và P95 của nhãn không đưa vào X dự đoán sớm.', '',
          '## Tái lập và kiểm tra', '',
          '```bash', 'python 17_eda_final_quality_v2.py', '```', '',
          'Dependencies: duckdb, numpy, matplotlib. Notebook 17 đi kèm đọc cùng các bảng và biểu đồ. '
          'Mỗi bảng có CSV và Parquet; CSV ghi NULL rõ ràng. SQL, phiên bản thư viện, '
          'SHA256 các nguồn và cấu hình nằm trong run_manifest.json.', '',
          f"Đạt **{sum(self.checks.values())}/{len(self.checks)}** kiểm tra EDA; "
          f"nguồn v2 có {len(self.source_checks)} kiểm tra thành công. Chi tiết [validation.json](validation.json).", '']
        (self.out/'BAO_CAO_EDA_FINAL_QUALITY_V2.md').write_text('\n'.join(lines),encoding='utf-8')

    def peer_report(self):
        r=self.results; m=r['e_unknown_mechanisms'][0]; n=m['n_E_unknown']
        return ['## Vì sao các mẫu này E: comment nhiều hay scale tương đối?', '',
          '**Có cả mức hoạt động cao so với peers và hiệu ứng chuẩn hóa tương đối; không phải mọi E đều cần nhiều comment.** '
          f"Trong {n:,} mẫu UNKNOWN_ONLY E, median có **{m['median_comments']:g} comment và {m['median_visits']:g} phiên**; "
          f"median của median peers cùng course/chapter lần lượt là {m['median_of_peer_median_comments']:g} comment "
          f"và {m['median_of_peer_median_visits']:g} phiên. "
          f"{m['n_at_least_six_comments']:,} ({pct(m['n_at_least_six_comments'],n):.2f}%) có ≥6 comment; "
          f"{m['n_at_least_six_visits']:,} ({pct(m['n_at_least_six_visits'],n):.2f}%) có ≥6 phiên.", '',
          f"Tuy nhiên **{m['n_one_comment_one_visit']:,} mẫu ({pct(m['n_one_comment_one_visit'],n):.2f}%) "
          f"chỉ có 1 comment và 1 phiên mà vẫn E**. Có {m['n_no_FA']:,} mẫu không có FA, "
          f"trong đó {m['n_reply_only']:,} chỉ có reply; chúng nhận E qua FV. "
          'Số comment tuyệt đối không phải điều kiện đủ hoặc điều kiện bắt buộc của E.', '',
          '### Phép scale thực tế', '',
          'Với Δᵤ=positiveᵤ−negativeᵤ, T=tổng comment chapter và K=P95(|Δ|) trên các user có Δ khác 0:', '',
          '```text',
          'FA_raw_u = Δ_u / T',
          'P95(|FA_raw|) = P95(|Δ|) / T = K / T',
          'FA_u = 0.5 + 0.5 × clip(Δ_u / K, -1, 1)',
          'FV_u = clip(số_phiên_u / P95_positive(số_phiên), 0, 1)',
          'AFELO_u = mean_available(FV_u, FA_u)',
          'UNKNOWN_ONLY: q = 1 − AFELO; E khi AFELO > 0.8',
          '```', '',
          '**T triệt tiêu.** Điểm FA cuối không còn là thị phần comment của user trên tổng chapter. '
          'Nó so chênh lệch positive−negative với mốc tương đối của peers. '
          'Nếu K=1 thì một comment positive đủ FA=1. Nhiều comment neutral không tăng FA; '
          'comment negative giảm tử số. FV tăng theo số phiên user-resource, không phải số comment đơn thuần.', '',
          '**User khác không có comment không trực tiếp kéo P95 FA xuống:** họ có FA=NULL, '
          'không thuộc tập FA reference; user có Δ=0 cũng không thuộc tập fit P95 |Δ|. '
          'FV chỉ fit các user có phiên dương; enrollment không có log không được thêm thành 0 vào P95. '
          'Nhưng nếu các user thực sự có log đều hoạt động ít, P95 thấp và điểm cao vẫn dễ đạt. '
          'Hiện tượng này xảy ra cả trong nhóm đông người, không chỉ singleton.', '',
          f"Median số user tham chiếu FV của UNKNOWN_ONLY E là **{m['median_FV_support']:g}**; "
          f"chỉ {m['n_FV_support_le5']:,} ({pct(m['n_FV_support_le5'],n):.2f}%) có support FV≤5. "
          f"{m['n_P95_visits_le1']:,} mẫu nằm ở nhóm P95 phiên≤1. "
          f"Trong nhóm có FA, {m['n_P95_net_le1']:,} mẫu có P95 |Δ|≤1. "
          'Các nhóm này có thể chồng nhau.', '',
          f"Do clip, {m['n_FV_one']:,} mẫu có FV=1; {m['n_FA_one']:,} có FA=1; "
          f"{m['n_both_one']:,} có cả hai bằng 1. "
          'Đạt 1 nghĩa là chạm/vượt ngưỡng P95 của nhóm tham chiếu, không có nghĩa hoàn thành toàn khóa.', '',
          '![Absolute activity and normalization](figures/07_E_absolute_vs_relative.png)', '',
          table(['Đại lượng','Mức','Số UNKNOWN_ONLY E','FV TB','FA TB','q TB','Median support FV','Median P95 phiên'],
            [[x[k] for k in ['metric','intensity','n_E_unknown','mean_FV','mean_FA','mean_q',
                              'median_FV_support','median_P95_visits']] for x in r['e_unknown_intensity_bins']]), '',
          '### Ví dụ đối chiếu user khác cùng course/chapter', '',
          table(['Cơ chế','User','Course','Comment','Phiên','User chapter','User course','P95 phiên','Δ','P95 |Δ|','FV','FA','q'],
            [[x[k] for k in ['mechanism','user_id','course_id','n_comments_all','n_visits','n_observed_users',
              'course_eligible_pairs','p95_n_visits','net_positive_comments','p95_abs_net_comments_direct','FV','FA','q_alpha_0p1']]
             for x in r['e_scale_examples'] if x['example_rank']==1]), '',
          'Peers được so theo **cùng course_id và lo_key** đúng với nhóm fit P95. '
          '“User course” gồm mọi cặp đủ thời gian; “User chapter” chỉ gồm user có log ở chapter đang xét. '
          'Chapter -1 gộp các resource/comment chưa biết chapter, nên không đại diện một chương học xác định.', '',
          f"Ngoài ra **{m['n_repeated_text']:,} mẫu ({pct(m['n_repeated_text'],n):.2f}%)** có nội dung sentiment "
          'lặp lại ở các comment khác nhau. Khử trùng hiện tại chỉ loại trùng chính xác cùng user/resource/timestamp/text; '
          'cùng text ở timestamp/resource khác vẫn có thể tăng counts. '
          f"Median số text khác nhau là {m['median_distinct_texts']:g}. "
          'Các trích đoạn đọc thủ công có “OK”, “perfect”, emoji lặp lại; đây là quan sát trên mẫu đọc có chủ đích, '
          'không phải kết luận spam cho toàn bộ E. Sentiment/hoạt động cao chưa xác nhận năng lực học tập.', '',
          'Chi tiết từng E chapter và peers: '
          '[e_chapter_peer_comparison.csv](tables/e_chapter_peer_comparison.csv). '
          'Có số user course/chapter, median/mean peers, P95, counts positive/negative và số text khác nhau.', '']

    def run(self):
        self.overview()
        self.e_audit()
        self.peer_audit()
        self.validate()
        self.figures()
        self.report()
        summary = {k:v for k,v in self.results.items() if k not in ['course_summary','e_review_comments','e_review_cases']}
        dump(self.out/'summary.json',summary)
        import matplotlib
        import numpy
        dump(self.out/'run_manifest.json',{
            'analysis_version':'final-quality-v2-eda-E-audit-2026-10-20',
            'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'client_date':'2026-10-20','source_formula_version':self.source_summary['formula_version'],
            'inputs':self.inventory,'queries':self.log,'source_checks':self.source_checks,
            'libraries':{'duckdb':duckdb.__version__,'matplotlib':matplotlib.__version__,'numpy':numpy.__version__},
            'config':{'sampling':'full population; manual excerpts are deliberately selected',
                      'timezone':'UTC, no timestamp shift','counterfactuals':'diagnostics only; source labels unchanged'},
            'elapsed_seconds':round(time.monotonic()-self.started,3)})
        dump(self.out/'_SUCCESS.json',{'checks_passed':len(self.checks),
                                     'E_union_pairs':self.con.execute('SELECT count(*) FROM e_cases').fetchone()[0]})
        self.con.close()
        print('DONE:',self.out,flush=True)
        return summary


def run_eda(root=None, output_root=None, memory='1500MB', threads=2):
    root=Path(root or ROOT)
    return EDA(root,output_root or root/'eda_final_quality_v2',memory,threads).run()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--memory',default='1500MB')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    run_eda(args.root,args.output_root,args.memory,args.threads)
