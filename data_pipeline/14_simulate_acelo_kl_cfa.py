"""Simulate binary KL times conditional CFA without changing production tables.

For a correct problem, f(t) = max(0.1, 1 - 0.1 * (t - 1)).
Thus f(1)=1, f(9)=0.2, and f(t>=10)=0.1. Wrong or unattempted
problems contribute zero to ACELO. CFA itself is averaged over correct
problems only. Keep the final-window population and filters from step 06.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import duckdb


FORMULA_VERSION = "binary_kl_conditional_cfa_linear_floor_0p1_v1"
KEYS = "user_id,course_id,lo_key"


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def records(con, sql):
    cursor = con.execute(sql)
    names = [c[0] for c in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def mdtable(headers, rows):
    def display(value):
        if value is None:
            return "NULL"
        if isinstance(value, float):
            return f"{value:,.6f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(display(x) for x in row) + " |" for row in rows],
    ])


def metric_stats(con, relation, metrics, cohorts):
    result = []
    for cohort, predicate in cohorts.items():
        for metric in metrics:
            result += records(con, f"""SELECT {qs(cohort)} cohort,{qs(metric)} metric,
                count(*) n_rows,count({metric}) n_nonnull,
                count(*)-count({metric}) n_null,
                count(*) FILTER(WHERE {metric}=0) n_zero,
                count(*) FILTER(WHERE {metric}>0) n_positive,
                count(*) FILTER(WHERE abs({metric}-1)<1e-12) n_one,
                min({metric}) minimum,avg({metric}) mean,stddev_pop({metric}) stddev,
                quantile_cont({metric},0.01) p01,quantile_cont({metric},0.05) p05,
                quantile_cont({metric},0.25) p25,quantile_cont({metric},0.50) median,
                quantile_cont({metric},0.75) p75,quantile_cont({metric},0.95) p95,
                quantile_cont({metric},0.99) p99,max({metric}) maximum,
                100.0*count(*) FILTER(WHERE {metric}=0)/nullif(count({metric}),0) pct_zero_nonnull
                FROM {relation} WHERE {predicate}""")
    return result


def run(output_root=None):
    start = time.monotonic()
    base = Path(__file__).resolve().parent
    out = Path(output_root or base / "acelo_kl_cfa_simulation").resolve()
    source = base / "final_quality_tables"
    raw_stage = base / "eda_engagement_features_comparison/_stage/problem_rows"
    baseline_path = source / "tables/acelo_components.parquet"
    pairs_path = source / "_stage/active_pairs.parquet"
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Simulation must be separate from production tables.")
    for directory in [out, out / "tables", out / "figures", out / "_scratch"]:
        directory.mkdir(parents=True, exist_ok=True)
    success = out / "_SUCCESS.json"
    if success.exists():
        success.unlink()

    inputs = [baseline_path, pairs_path, source / "run_manifest.json",
              raw_stage / "_SUCCESS.json", *sorted(raw_stage.rglob("*.parquet"))]
    input_hashes = {str(p.relative_to(base)): sha256(p) for p in inputs}
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory={qs(out / '_scratch')}")
    con.execute("SET max_temp_directory_size='4GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET TimeZone='UTC'")
    con.execute(f"CREATE VIEW baseline AS SELECT * FROM read_parquet({qs(baseline_path)})")
    con.execute(f"CREATE VIEW active_pairs AS SELECT * FROM read_parquet({qs(pairs_path)})")

    # Integer tenths preserve the exact linear weights before the final division.
    # f(1)=10/10; f(9)=2/10; f(10+)=1/10.
    queries = {}
    for shard in range(32):
        path = raw_stage / f"user_shard={shard}" / "*.parquet"
        sql = f"""SELECT p.user_id,p.course_id,p.lo_key,
            count(*) n_problem_rows_recomputed,
            count(DISTINCT p.problem_id) n_distinct_problems_recomputed,
            count(*) FILTER(WHERE p.is_correct=1) n_correct_recomputed,
            count(*) FILTER(WHERE p.is_correct IS NULL OR p.is_correct NOT IN (0,1)) n_invalid_correctness,
            count(*) FILTER(WHERE p.attempts IS NULL OR NOT isfinite(p.attempts)
                OR p.attempts<=0 OR p.attempts<>floor(p.attempts)) n_invalid_attempts_recomputed,
            count(*) FILTER(WHERE p.is_correct=1 AND (p.attempts IS NULL
                OR NOT isfinite(p.attempts) OR p.attempts<=0 OR p.attempts<>floor(p.attempts))) n_correct_invalid_attempts,
            count(*) FILTER(WHERE p.is_correct=1 AND p.attempts>5) n_correct_attempts_gt5,
            count(*) FILTER(WHERE p.is_correct=1 AND p.attempts>9) n_correct_attempts_gt9,
            max(p.attempts) max_attempts_observed,
            max(p.attempts) FILTER(WHERE p.is_correct=1) max_correct_attempts_observed,
            coalesce(sum(CASE WHEN p.is_correct=1 AND p.attempts>0
                AND isfinite(p.attempts) AND p.attempts=floor(p.attempts)
                THEN greatest(1.0,11.0-p.attempts) END),0) correct_weight_tenths_sum,
            coalesce(sum(CASE WHEN p.efficiency_valid THEN p.is_correct::DOUBLE/p.attempts END),0) inverse_efficiency_sum_recomputed
            FROM read_parquet({qs(path)}) p JOIN active_pairs a USING(user_id,course_id)
            WHERE p.event_time BETWEEN a.enroll_time AND a.course_end
            GROUP BY 1,2,3"""
        if shard == 0:
            con.execute("CREATE TEMP TABLE problem_aggregates AS " + sql)
            queries["problem_aggregate_template_shard0"] = sql
        else:
            con.execute("INSERT INTO problem_aggregates " + sql)
    print("Recomputed all 32 user shards in the original final window", flush=True)

    def save(name, sql):
        target = out / "tables" / f"{name}.parquet"
        partial = target.with_suffix(".parquet.partial")
        con.execute(f"COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET,COMPRESSION ZSTD)")
        partial.replace(target)
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({qs(target)})")
        queries[name] = sql

    aggregate_fields = [r[0] for r in con.execute("DESCRIBE problem_aggregates").fetchall()
                        if r[0] not in KEYS.split(",")]
    fields = ",".join(f"coalesce(p.{field},0) {field}" for field in aggregate_fields)
    save("chapter_acelo_simulated", f"""WITH joined AS (
        SELECT b.*,b.KL KL_score_current,b.CFA CFA_current,b.ACELO ACELO_current,
            {fields}
        FROM baseline b LEFT JOIN problem_aggregates p USING({KEYS})
      ), components AS (
        SELECT *,CASE WHEN n_problems_available>0 AND n_invalid_correctness=0
            THEN n_correct_recomputed::DOUBLE/n_problems_available END KL_binary,
          CASE WHEN n_correct_recomputed>0 AND n_invalid_correctness=0 AND n_correct_invalid_attempts=0
            THEN correct_weight_tenths_sum/(10.0*n_correct_recomputed) END CFA_conditional
        FROM joined
      ), scored AS (
        SELECT *,CASE WHEN KL_binary IS NULL THEN NULL
            WHEN n_correct_recomputed=0 THEN 0.0
            WHEN CFA_conditional IS NOT NULL THEN KL_binary*CFA_conditional END ACELO_simulated,
          CASE WHEN n_problems_available>0 AND n_invalid_correctness=0 AND n_correct_invalid_attempts=0
            THEN correct_weight_tenths_sum/(10.0*n_problems_available) END ACELO_direct_check
        FROM components
      )
      SELECT *,ACELO_simulated-ACELO_current ACELO_change,
        CASE WHEN n_problems_available=0 THEN 'NO_PROBLEM_CATALOG'
             WHEN n_invalid_correctness>0 THEN 'INVALID_CORRECTNESS'
             WHEN n_problem_rows_recomputed=0 THEN 'NO_PROBLEM_LOG_ZERO_BY_CONVENTION'
             WHEN n_correct_recomputed=0 THEN 'OBSERVED_NO_CORRECT_PROBLEM'
             WHEN n_correct_invalid_attempts>0 THEN 'INVALID_ATTEMPTS_ON_CORRECT'
             ELSE 'SCORABLE_WITH_CORRECT_PROBLEMS' END simulation_status,
        {qs(FORMULA_VERSION)} simulation_formula_version
      FROM scored""")

    save("user_course_acelo_simulated", """SELECT user_id,course_id,min(split) split,
        count(*) n_chapters_observed,
        count(*) FILTER(WHERE n_problem_rows_recomputed>0) n_chapters_with_problem_log,
        count(ACELO_current) n_chapters_current_scored,
        count(ACELO_simulated) n_chapters_simulated_scored,
        avg(ACELO_current) ACELO_current_mean_observed,
        avg(ACELO_simulated) ACELO_simulated_mean_observed,
        CASE WHEN count(ACELO_current)=count(*) THEN avg(ACELO_current) END ACELO_current_all_observed,
        CASE WHEN count(ACELO_simulated)=count(*) THEN avg(ACELO_simulated) END ACELO_simulated_all_observed,
        avg(ACELO_simulated)-avg(ACELO_current) ACELO_mean_change,
        sum(n_correct_attempts_gt9) n_correct_attempts_gt9
        FROM chapter_acelo_simulated GROUP BY 1,2""")

    cohorts = {
        "ALL_OBSERVED_CHAPTERS": "true",
        "WITH_PROBLEM_LOG": "n_problem_rows_recomputed>0",
        "WITH_CORRECT_PROBLEM": "n_correct_recomputed>0",
        "NO_PROBLEM_LOG": "n_problem_rows_recomputed=0",
    }
    metrics = ["KL_binary", "CFA_conditional", "ACELO_current", "ACELO_simulated"]
    stats = metric_stats(con, "chapter_acelo_simulated", metrics, cohorts)
    user_stats = metric_stats(con, "user_course_acelo_simulated",
        ["ACELO_current_mean_observed", "ACELO_simulated_mean_observed",
         "ACELO_current_all_observed", "ACELO_simulated_all_observed"],
        {"USER_COURSE_WITH_OBSERVED_CHAPTER": "true"})

    long_selects = []
    for cohort, predicate in cohorts.items():
        for metric in ["ACELO_current", "ACELO_simulated"]:
            long_selects.append(f"SELECT {qs(cohort)} cohort,{qs(metric)} metric,{metric} score_value "
                                f"FROM chapter_acelo_simulated WHERE {predicate}")
    save("acelo_histogram", """WITH values_long AS (""" + " UNION ALL ".join(long_selects) + """),
      rounded AS (SELECT *,round(score_value,12) score_for_bin FROM values_long),
      binned AS (SELECT *,CASE WHEN score_for_bin IS NULL THEN 7 WHEN score_for_bin=0 THEN 0
        WHEN score_for_bin<0.2 THEN 1 WHEN score_for_bin<0.4 THEN 2 WHEN score_for_bin<0.6 THEN 3
        WHEN score_for_bin<0.8 THEN 4 WHEN score_for_bin<1 THEN 5 ELSE 6 END bin_order FROM rounded),
      bins(bin_order,bin_label) AS (VALUES (0,'0'),(1,'(0,0.2)'),(2,'[0.2,0.4)'),
        (3,'[0.4,0.6)'),(4,'[0.6,0.8)'),(5,'[0.8,1)'),(6,'1'),(7,'NULL')),
      dimensions AS (SELECT DISTINCT cohort,metric FROM values_long),
      counts AS (SELECT cohort,metric,bin_order,count(*) n_rows FROM binned GROUP BY 1,2,3)
      SELECT d.*,b.*,coalesce(c.n_rows,0) n_rows,
        100.0*coalesce(c.n_rows,0)/sum(coalesce(c.n_rows,0)) OVER(PARTITION BY d.cohort,d.metric) pct_all_rows
      FROM dimensions d CROSS JOIN bins b LEFT JOIN counts c USING(cohort,metric,bin_order)
      ORDER BY cohort,metric,bin_order""")
    split_stats = records(con, """SELECT split,count(*) n_rows,count(ACELO_simulated) n_scored,
        avg(ACELO_current) mean_current,avg(ACELO_simulated) mean_simulated,
        quantile_cont(ACELO_simulated,0.5) median_simulated,
        count(*) FILTER(WHERE ACELO_simulated=0) n_zero_simulated
        FROM chapter_acelo_simulated GROUP BY 1 ORDER BY split""")
    comparison = records(con, """SELECT count(*) n_paired,
        avg(ACELO_change) mean_change,quantile_cont(ACELO_change,0.5) median_change,
        min(ACELO_change) min_change,max(ACELO_change) max_change,
        count(*) FILTER(WHERE ACELO_change>1e-12) n_increased,
        count(*) FILTER(WHERE ACELO_change< -1e-12) n_decreased,
        count(*) FILTER(WHERE abs(ACELO_change)<=1e-12) n_unchanged
        FROM chapter_acelo_simulated WHERE ACELO_current IS NOT NULL AND ACELO_simulated IS NOT NULL""")[0]
    population = records(con, """SELECT count(*) n_observed_chapters,
        count(DISTINCT(user_id,course_id)) n_observed_user_courses,
        count(DISTINCT course_id) n_courses,
        count(ACELO_simulated) n_scored_simulated,count(ACELO_current) n_scored_current,
        count(*) FILTER(WHERE n_problems_available=0) n_no_problem_catalog,
        count(*) FILTER(WHERE n_problem_rows_recomputed=0) n_no_problem_log,
        count(*) FILTER(WHERE n_problem_rows_recomputed=0 AND n_problems_available>0) n_no_problem_log_scored_zero,
        count(*) FILTER(WHERE n_problem_rows_recomputed>0) n_with_problem_log,
        count(*) FILTER(WHERE n_problem_rows_recomputed>0 AND n_correct_recomputed=0) n_with_log_no_correct,
        count(*) FILTER(WHERE n_correct_recomputed>0) n_with_correct,
        sum(n_problem_rows_recomputed) n_problem_rows,sum(n_correct_recomputed) n_correct_problems,
        sum(n_correct_attempts_gt5) n_correct_attempts_gt5,
        sum(n_correct_attempts_gt9) n_correct_attempts_gt9,
        count(*) FILTER(WHERE n_correct_attempts_gt9>0) n_chapters_affected_by_floor,
        max(max_attempts_observed) max_attempts_observed,
        sum(n_invalid_correctness) n_invalid_correctness,
        sum(n_invalid_attempts_recomputed) n_invalid_attempts,
        count(*) FILTER(WHERE KL_score_current IS NULL AND KL_binary IS NOT NULL) n_KL_gained_without_max_score
        FROM chapter_acelo_simulated""")[0]

    checks = {
        "same_chapter_population_and_unique_keys": con.execute(f"SELECT count(*)=(SELECT count(*) FROM baseline) AND count(*)=count(DISTINCT({KEYS})) FROM chapter_acelo_simulated").fetchone()[0],
        "no_aggregate_outside_baseline": con.execute(f"SELECT count(*)=0 FROM problem_aggregates ANTI JOIN baseline USING({KEYS})").fetchone()[0],
        "problem_rows_unique_within_chapter": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE n_problem_rows_recomputed<>n_distinct_problems_recomputed").fetchone()[0],
        "problem_row_counts_match_step06": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE n_problem_rows_recomputed<>n_problems_attempted").fetchone()[0],
        "correct_counts_match_step06": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE n_correct_recomputed<>n_problems_correct").fetchone()[0],
        "inverse_efficiency_matches_step06": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE abs(inverse_efficiency_sum_recomputed-efficiency_sum)>1e-9").fetchone()[0],
        "no_invalid_correctness_or_attempts_in_observed_source": population["n_invalid_correctness"] == population["n_invalid_attempts"] == 0,
        "counts_within_catalog": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE n_correct_recomputed>n_problems_available OR n_problem_rows_recomputed>n_problems_available").fetchone()[0],
        "all_scores_finite_and_bounded": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE (KL_binary IS NOT NULL AND (NOT isfinite(KL_binary) OR KL_binary NOT BETWEEN 0 AND 1)) OR (CFA_conditional IS NOT NULL AND (NOT isfinite(CFA_conditional) OR CFA_conditional NOT BETWEEN 0.1-1e-12 AND 1+1e-12)) OR (ACELO_simulated IS NOT NULL AND (NOT isfinite(ACELO_simulated) OR ACELO_simulated NOT BETWEEN 0 AND 1))").fetchone()[0],
        "product_equals_direct_weighted_correct_rate": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE (ACELO_simulated IS NULL)<>(ACELO_direct_check IS NULL) OR abs(ACELO_simulated-ACELO_direct_check)>1e-12").fetchone()[0],
        "simulated_score_not_above_binary_KL": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE ACELO_simulated>KL_binary+1e-12").fetchone()[0],
        "floor_implies_score_at_least_tenth_of_KL": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE ACELO_simulated+1e-12<0.1*KL_binary").fetchone()[0],
        "no_correct_has_null_conditional_CFA": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE n_correct_recomputed=0 AND CFA_conditional IS NOT NULL").fetchone()[0],
        "no_problem_catalog_has_no_score": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE n_problems_available=0 AND (KL_binary IS NOT NULL OR ACELO_simulated IS NOT NULL)").fetchone()[0],
        "zero_correct_with_valid_catalog_has_zero_score": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE n_problems_available>0 AND n_correct_recomputed=0 AND ACELO_simulated IS DISTINCT FROM 0.0").fetchone()[0],
        "same_score_coverage_as_current": con.execute("SELECT count(*)=0 FROM chapter_acelo_simulated WHERE (ACELO_current IS NULL)<>(ACELO_simulated IS NULL)").fetchone()[0],
        "histograms_partition_populations": all(r["total"] == r["expected"] for r in records(con, "SELECT cohort,metric,sum(n_rows) total,max(CASE cohort WHEN 'ALL_OBSERVED_CHAPTERS' THEN (SELECT count(*) FROM baseline) WHEN 'WITH_PROBLEM_LOG' THEN (SELECT count(*) FROM baseline WHERE n_problems_attempted>0) WHEN 'WITH_CORRECT_PROBLEM' THEN (SELECT count(*) FROM baseline WHERE n_problems_correct>0) ELSE (SELECT count(*) FROM baseline WHERE n_problems_attempted=0) END) expected FROM acelo_histogram GROUP BY 1,2")),
    }
    # Independent hand-calculated anchors, including the agreed floor and zero case.
    anchors = con.execute("""WITH p(case_id,correct,attempts) AS (VALUES
        ('first',1,1),('ninth',1,9),('tenth',1,10),('long',1,190),('wrong',0,1),
        ('mixed',1,1),('mixed',1,2),('mixed',0,1))
        SELECT case_id,avg(CASE WHEN correct=1 THEN greatest(1,11-attempts)/10.0 ELSE 0 END)
        FROM p GROUP BY 1 ORDER BY 1""").fetchall()
    expected = {"first": 1.0, "ninth": 0.2, "tenth": 0.1, "long": 0.1, "wrong": 0.0, "mixed": 1.9/3}
    checks["hand_calculated_weight_and_zero_anchors"] = all(abs(value-expected[name])<1e-12 for name,value in anchors)
    checks["source_files_unchanged"] = input_hashes == {str(p.relative_to(base)): sha256(p) for p in inputs}
    assert all(checks.values()), checks

    summary = {
        "formula_version": FORMULA_VERSION,
        "population": population,
        "chapter_statistics": stats,
        "user_course_statistics": user_stats,
        "split_statistics": split_stats,
        "paired_comparison": comparison,
        "attempt_penalty_summary": records(con, """SELECT avg(KL_binary-ACELO_simulated) mean_KL_reduction,
            count(*) FILTER(WHERE abs(CFA_conditional-1)<1e-12) n_conditional_CFA_one,
            100.0*count(*) FILTER(WHERE abs(CFA_conditional-1)<1e-12)/count(CFA_conditional) pct_conditional_CFA_one
            FROM chapter_acelo_simulated""")[0],
        "simulation_status": records(con, "SELECT simulation_status,count(*) n_rows FROM chapter_acelo_simulated GROUP BY 1 ORDER BY n_rows DESC"),
        "histogram": records(con, "SELECT * FROM acelo_histogram"),
        "validation": checks,
    }
    dump(out / "summary.json", summary)
    dump(out / "validation.json", checks)
    for name, values in [("chapter_statistics", stats), ("user_course_statistics", user_stats), ("split_statistics", split_stats)]:
        json_path = out / f"{name}.json"
        dump(json_path, values)
        con.execute(f"COPY (SELECT * FROM read_json_auto({qs(json_path)})) TO {qs(out / 'tables' / (name+'.parquet'))} (FORMAT PARQUET,COMPRESSION ZSTD)")

    render_plot(con, out)
    report = render_report(summary)
    report_path = out / "bao_cao_gia_lap_acelo.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "as_of_local_date": "2026-10-20",
        "duckdb_version": duckdb.__version__,
        "engine_source_sha256": sha256(__file__),
        "formula_version": FORMULA_VERSION,
        "formulas": {"KL": "S/N; binary correctness over all catalog problems",
            "CFA": "sum(f(attempts) for correct problems)/S, NULL when S=0",
            "f": "max(0.1,1-0.1*(attempts-1)); f(1)=1, f(9)=0.2, f(10+)=0.1",
            "ACELO": "KL*CFA when S>0; 0 when S=0 and N>0; NULL when N=0",
            "equivalent_ACELO": "sum(f(attempts) for correct problems)/N"},
        "weight_extension_authority": "User instruction: subtract 0.1 per failed attempt; above nine attempts keep 0.1; wrong problems contribute zero.",
        "time_scope": "Same step06 active_pairs and inclusive enroll_time <= event_time <= course_end",
        "chapter_scope": "All 845238 observed final-window user-course-chapter rows; no expansion to unobserved chapters",
        "unattempted_policy": "Zero contribution with valid catalog, matching existing pipeline; absence of a log is not evidence of inability",
        "invalid_data_policy": "Invalid correctness makes KL/ACELO NULL; invalid attempts on a correct problem makes CFA/ACELO NULL; none found in this run",
        "attempts_semantics": "Aggregate user-problem attempts used as a proxy for attempts until correctness; per-attempt history is unavailable",
        "source_files_sha256": input_hashes,
        "queries": queries,
        "production_mutation": "none; no LOQV/q/y or existing component table rewritten",
        "sampling": "none",
        "histogram_numeric_policy": "Round scores to 12 decimal places only when assigning descriptive bins, to avoid floating-point boundary artifacts; stored scores and summary statistics are unrounded",
        "elapsed_seconds": round(time.monotonic()-start,3),
        "report_sha256": sha256(report_path),
    }
    dump(out / "run_manifest.json", manifest)
    dump(success, {"manifest_sha256": sha256(out / "run_manifest.json"), "report_sha256": sha256(report_path)})
    print("SIMULATION_COMPLETE", json.dumps(population, ensure_ascii=False), flush=True)
    print("PAIRED", json.dumps(comparison, ensure_ascii=False), flush=True)
    print("CHECKS", len(checks), "passed", flush=True)
    con.close()
    return summary


def render_plot(con, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frame = con.execute("SELECT ACELO_current,ACELO_simulated FROM chapter_acelo_simulated WHERE ACELO_current IS NOT NULL AND ACELO_simulated IS NOT NULL").fetchnumpy()
    old, new = frame["ACELO_current"], frame["ACELO_simulated"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    edges = np.linspace(0, 1, 21)
    axes[0].hist([np.round(old,12), np.round(new,12)], bins=edges, label=["Current ACELO", "Simulated ACELO"], color=["#9ba7b7", "#166d91"], alpha=.9)
    axes[0].set(xlabel="ACELO", ylabel="Observed chapters", title="Distribution on the same scored chapters")
    axes[0].legend(fontsize=8)
    for values, label, color in [(old,"Current ACELO","#7b8796"),(new,"Simulated ACELO","#166d91")]:
        x = np.sort(values)
        ix = np.unique(np.linspace(0, len(x)-1, 1000).astype(int))
        axes[1].plot(x[ix], (ix+1)/len(x), label=label, color=color)
    axes[1].set(xlabel="ACELO", ylabel="Cumulative fraction", title="Empirical cumulative distribution", ylim=(0,1))
    axes[1].legend(fontsize=8)
    delta = new-old
    changed = delta[np.abs(delta)>1e-12]
    axes[2].hist(changed, bins=np.linspace(-.5,.5,61), color="#166d91")
    axes[2].axvline(0, color="#a34331", linewidth=1)
    axes[2].set(xlabel="Simulated minus current ACELO", ylabel="Chapters with a changed score", title=f"Changed: {len(changed):,}; unchanged: {len(delta)-len(changed):,}")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(axis="y", style="plain")
    fig.suptitle(f"Binary KL x conditional CFA | f(1)=1, f(9)=0.2, f(10+)=0.1 | n={len(new):,}", fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(out / "figures/acelo_distribution_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(s):
    p=s["population"]
    selected=[r for r in s["chapter_statistics"] if r["cohort"]=="ALL_OBSERVED_CHAPTERS"]
    by_metric={r["metric"]:r for r in selected}
    current=by_metric["ACELO_current"];new=by_metric["ACELO_simulated"]
    penalty=s["attempt_penalty_summary"]
    hist=[r for r in s["histogram"] if r["cohort"]=="ALL_OBSERVED_CHAPTERS"]
    histogram_rows=[]
    for idx in range(8):
        old=next(r for r in hist if r["metric"]=="ACELO_current" and r["bin_order"]==idx)
        sim=next(r for r in hist if r["metric"]=="ACELO_simulated" and r["bin_order"]==idx)
        histogram_rows.append([old["bin_label"],old["n_rows"],sim["n_rows"],sim["pct_all_rows"]])
    parts=[
        "# Giả lập ACELO bằng KL nhân CFA có điều kiện",
        "Ngày 20/10/2026. Tính toàn bộ dữ liệu, không lấy mẫu. Kết quả được lưu riêng, chưa thay đổi nhãn sản xuất.",
        f"Trên {p['n_observed_chapters']:,} dòng user-course-chapter đã quan sát, có {new['n_nonnull']:,} ACELO tính được. "
        f"Mean ACELO chuyển từ {current['mean']:.6f} sang {new['mean']:.6f}; median từ {current['median']:.6f} sang {new['median']:.6f}. "
        "So sánh này thay đồng thời định nghĩa KL, CFA và phép ghép ACELO.",
        f"CFA có điều kiện đạt trung bình {by_metric['CFA_conditional']['mean']:.6f}; "
        f"{penalty['pct_conditional_CFA_one']:.2f}% chapter có problem đúng có CFA=1. "
        f"Vì vậy ACELO gần KL nhị phân: hệ số số lần thử làm giảm trung bình {penalty['mean_KL_reduction']:.6f} điểm trên cùng tập tính được.",
        "## Công thức và quy ước",
        "`N` là tổng problem trong catalog chapter. `S` là số problem có `is_correct=1` trong cửa sổ quan sát. "
        "`KL_binary=S/N`. Với mỗi problem đúng, `f(t)=max(0.1,1-0.1*(t-1))`: lần 1 bằng 1, lần 2 bằng 0.9, "
        "lần 9 bằng 0.2, từ lần 10 trở đi bằng 0.1. Problem sai hoặc chưa làm đóng góp 0.",
        "`CFA_conditional=sum(f(t) trên problem đúng)/S`. `ACELO_simulated=KL_binary*CFA_conditional`, "
        "tương đương `sum(f(t) trên problem đúng)/N`. Khi S=0 và N>0, CFA để NULL vì trung bình trên tập rỗng; ACELO bằng 0. "
        "Khi N=0, KL và ACELO để NULL.",
        "Quy tắc trên do người dùng chọn cho bản giả lập. Paper gốc dùng lần vượt qua toàn bài tự đánh giá, "
        "với 5 hệ số từ 1 đến 0.6; việc áp dụng trên từng problem và mở rộng hệ số tới sàn 0.1 là điều chỉnh.",
        "KL hiện tại trong bước 06 dùng tổng điểm đạt/tổng điểm tối đa; ACELO hiện tại là mean_non_null(KL_score,CFA_inverse_all_problems). "
        "Bản giả lập dùng đúng/sai nhị phân và không cần điểm tối đa; điểm từng phần không được sử dụng.",
        "## Phạm vi dữ liệu",
        mdtable(["Chỉ tiêu","Số lượng"],list(p.items())),
        "Dùng cùng active_pairs và điều kiện `enroll_time <= event_time <= course_end` của bước 06. "
        "Không thêm chapter chưa quan sát. Bảng user-course chỉ gồm các cặp có chapter quan sát, không mở rộng sang các cặp chưa quan sát.",
        f"Có {p['n_no_problem_log_scored_zero']:,} chapter có catalog problem nhưng không có log problem trong cửa sổ. "
        "Các dòng này nhận ACELO=0 theo quy ước hiện tại; đây không phải bằng chứng người học không có năng lực. "
        "Cần xem riêng nhóm có log problem khi diễn giải kết quả.",
        "`attempts` là trạng thái tổng hợp user-problem, không có lịch sử cho từng lần. "
        "Phép giả lập dùng attempts của dòng đúng như đại diện số lần để làm đúng; chưa xác nhận được thời điểm đúng lần đầu.",
        "## Thống kê chapter",
        mdtable(["Chỉ số","Có giá trị","NULL","Mean","Std","P25","Median","P75","P95","Bằng 0","Bằng 1"],
            [[r[k] for k in ["metric","n_nonnull","n_null","mean","stddev","p25","median","p75","p95","n_zero","n_one"]] for r in selected]),
        "## Phân bố ACELO theo khoảng",
        "Đây là các khoảng mô tả phân bố ACELO, không phải nhãn E/G/A. % dùng toàn bộ chapter quan sát làm mẫu số. "
        "Chỉ khi xếp khoảng, điểm được làm tròn tới 12 chữ số thập phân để tránh lệch biên do số thực; bảng điểm và thống kê giữ giá trị chưa làm tròn.",
        mdtable(["Khoảng","Hiện tại","Giả lập","% giả lập / toàn bộ dòng"],histogram_rows),
        "## Tách nhóm có log làm bài",
        mdtable(["Nhóm","Chỉ số","Có giá trị","Mean","Median","Bằng 0","% 0 / có giá trị"],
            [[r[k] for k in ["cohort","metric","n_nonnull","mean","median","n_zero","pct_zero_nonnull"]]
             for r in s["chapter_statistics"] if r["cohort"]!="ALL_OBSERVED_CHAPTERS" and r["metric"] in ["ACELO_current","ACELO_simulated"]]),
        "## Thay đổi trên cùng chapter",
        mdtable(["Chỉ tiêu","Giá trị"],list(s["paired_comparison"].items())),
        "## Thống kê theo split",
        mdtable(list(s["split_statistics"][0]),[list(r.values()) for r in s["split_statistics"]]),
        "## Thống kê user-course",
        "`mean_observed` là trung bình các chapter có ACELO hợp lệ. `all_observed` chỉ có giá trị nếu mọi chapter quan sát đều tính được. "
        "Không coi trung bình trên các chapter là cách tính lại KL và CFA trên toàn bộ problem của khóa; hai phép tổng hợp có trọng số khác nhau.",
        mdtable(["Chỉ số","Số cặp","Có giá trị","Mean","Median","P25","P75","Bằng 0"],
            [[r[k] for k in ["metric","n_rows","n_nonnull","mean","median","p25","p75","n_zero"]] for r in s["user_course_statistics"]]),
        "## Biểu đồ",
        "![Phân bố ACELO](figures/acelo_distribution_comparison.png)",
        "## Kiểm chứng và tái lập",
        f"Đạt {sum(bool(v) for v in s['validation'].values())}/{len(s['validation'])} kiểm tra. "
        "Đối chiếu từng chapter với bước 06 về số problem, số problem đúng và tổng hiệu quả nghịch đảo; "
        "kiểm tra khóa duy nhất, miền giá trị, đồng nhất phép nhân với tỷ lệ đúng có trọng số, các trường hợp biên và hash đầu vào.",
        "Chạy lại: `python 14_simulate_acelo_kl_cfa.py` trong working_13-10, dùng môi trường có DuckDB, NumPy và Matplotlib. "
        "Chi tiết nằm trong summary.json, validation.json và run_manifest.json. "
        "Bảng kết quả nằm ở tables/chapter_acelo_simulated.parquet và tables/user_course_acelo_simulated.parquet. "
        "Chưa tính lại LOQV, q hoặc y.",
    ]
    return "\n\n".join(parts)+"\n"


if __name__ == "__main__":
    run()
