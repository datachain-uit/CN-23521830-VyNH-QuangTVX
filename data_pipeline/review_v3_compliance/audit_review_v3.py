"""Read-only comparison evidence for Review 2026-10-12 versus Data V3.

Writes only this audit directory. Does not rebuild datasets, refit anything,
train a model, or treat diagnostics on the labeled cohort as full-course truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run():
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1200MB'")
    con.execute("SET TimeZone='UTC'")
    evidence = {
        "audit_date": "2026-10-21",
        "duckdb_version": duckdb.__version__,
        "scope": "Read-only data checks; no training or protocol changes; course identity checks limited to course_id per user instruction",
        "diagnostic_population": "Only retained labeled user-course pairs; not all enrollments",
        "source_sha256": {},
        "stored_validation": {},
    }

    def rows(sql):
        result = con.execute(sql)
        return [dict(zip([d[0] for d in result.description], r)) for r in result.fetchall()]

    def view(name, path):
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({quote(ROOT / path)})")

    for relative in [
        "Review-2026-10-12.docx", "DATA_CARD_V3.md",
        "15_build_final_quality_v2.py", "16_build_phase_features_v2.py",
        "18_build_final_quality_v3.py", "19_build_phase_features_v3.py",
        "final_quality_v3/run_manifest.json", "final_quality_v3/formula_policy.json",
        "phase_features_v3/run_manifest.json",
        "phase_features_v3/course_random_split/run_manifest.json",
    ]:
        evidence["source_sha256"][relative] = digest(ROOT / relative)
    for relative in ["final_quality_v3", "phase_features_v3", "phase_features_v3/course_random_split"]:
        checks = json.loads((ROOT / relative / "validation.json").read_text())
        success = json.loads((ROOT / relative / "_SUCCESS.json").read_text())
        manifest = ROOT / relative / "run_manifest.json"
        evidence["stored_validation"][relative] = {
            "passed": sum(v is True for v in checks.values()), "total": len(checks),
            "success_manifest_hash_matches": success.get("manifest_sha256") == digest(manifest),
        }

    view("full_final", "final_quality_v3/tables/user_course_final/*.parquet")
    con.execute("""CREATE TEMP TABLE labeled AS SELECT user_id,course_id,
        enroll_time,course_start,course_end,cutoff_P1,cutoff_P2,cutoff_P3,cutoff_P4,
        q_alpha_0p1,y_alpha_0p1,q_alpha_1,y_alpha_1
        FROM full_final WHERE y_alpha_0p1 IS NOT NULL""")
    evidence["population"] = rows("""SELECT count(*) n_final,
        count(y_alpha_0p1) n_labeled, count(*)-count(y_alpha_0p1) n_unlabeled,
        count(DISTINCT course_id) n_courses, count(DISTINCT user_id) n_users
        FROM full_final""")[0]
    evidence["label_checks"] = rows("""SELECT count(*) n_labeled,
        count(DISTINCT(user_id,course_id)) n_unique_pairs,
        count(DISTINCT course_id) n_courses,count(DISTINCT user_id) n_users,
        count(*) FILTER(WHERE user_id IS NULL OR course_id IS NULL) n_missing_keys
        FROM labeled""")[0]
    evidence["thresholds"] = {}
    for suffix in ["0p1", "1"]:
        evidence["thresholds"][suffix] = rows(f"""SELECT count(*) FILTER(WHERE
            y_alpha_{suffix} IS DISTINCT FROM CASE WHEN q_alpha_{suffix}<0.2 THEN 'E'
            WHEN q_alpha_{suffix}<0.4 THEN 'G' WHEN q_alpha_{suffix}>=0.4 THEN 'A' END
            ) n_mismatch FROM full_final""")[0]
    evidence["within_course_label_variation"] = rows("""SELECT
        count(*) FILTER(WHERE n0>1) courses_multiple_labels_alpha_0p1,
        count(*) FILTER(WHERE n1>1) courses_multiple_labels_alpha_1
        FROM (SELECT course_id,count(DISTINCT y_alpha_0p1) n0,
        count(DISTINCT y_alpha_1) n1 FROM labeled GROUP BY 1)""")[0]
    view("chapter", "final_quality_v3/tables/chapter_quality.parquet")
    evidence["chapter_aggregation"] = rows("""SELECT
        max(abs(f.q_alpha_0p1-c.q0)) max_error_alpha_0p1,
        max(abs(f.q_alpha_1-c.q1)) max_error_alpha_1
        FROM labeled f JOIN (SELECT user_id,course_id,avg(LOQV_alpha_0p1) q0,
        avg(LOQV_alpha_1) q1 FROM chapter GROUP BY 1,2) c USING(user_id,course_id)""")[0]
    evidence["chapter_components"] = rows("""SELECT loqv_n_components,count(*) n
        FROM chapter GROUP BY 1 ORDER BY 1""")
    evidence["formula_comparison"] = rows("""SELECT
        count(*) FILTER(WHERE loqv_n_components=3) chapters_full_three,
        max(abs(LOQV_alpha_1-sqrt((pow(1-COELO,2)+pow(1-AFELO,2)+pow(1-ACELO,2))/3)))
            alpha_1_max_error_full_three,
        count(*) FILTER(WHERE loqv_n_components<3) chapters_with_missing_dimension
        FROM chapter""")[0]

    evidence["variants"] = {}
    for variant, folder in [
        ("user_course_stratified", "phase_features_v3"),
        ("course_random", "phase_features_v3/course_random_split"),
    ]:
        detail = {}
        evidence["variants"][variant] = detail
        view("assignment", folder + "/tables/labeled_user_course_splits.parquet")
        con.execute("""CREATE OR REPLACE TEMP VIEW cohort AS SELECT f.*,s.split
            FROM labeled f JOIN assignment s USING(user_id,course_id)""")
        detail["split_counts"] = rows("""SELECT split,count(*) n_pairs,
            count(DISTINCT course_id) n_courses,count(DISTINCT user_id) n_users
            FROM cohort GROUP BY 1 ORDER BY 1""")
        detail["overlap"] = rows("""SELECT
            (SELECT count(*) FROM (SELECT course_id FROM cohort GROUP BY 1 HAVING count(DISTINCT split)>1))
                courses_in_multiple_splits,
            (SELECT count(*) FROM (SELECT user_id FROM cohort GROUP BY 1 HAVING count(DISTINCT split)>1))
                users_in_multiple_splits""")[0]
        detail["test_user_overlap"] = rows("""SELECT count(*) n_test_enrollments,
            count(*) FILTER(WHERE t.user_id IN (SELECT user_id FROM cohort WHERE split='train')) n_old,
            count(*) FILTER(WHERE t.user_id NOT IN (SELECT user_id FROM cohort WHERE split='train')) n_new
            FROM cohort t WHERE split='test'""")[0]
        detail["time_overlap_diagnostic"] = rows("""WITH first_test AS (
            SELECT min(course_start) first_start FROM cohort WHERE split='test'),
            courses AS (SELECT DISTINCT course_id,course_end FROM cohort WHERE split='train')
            SELECT min(first_start) earliest_test_observed_start,count(*) train_courses,
            count(*) FILTER(WHERE course_end>first_start) train_courses_ending_after_first_test_start
            FROM courses CROSS JOIN first_test""")[0]
        config = json.loads((ROOT / folder / "feature_columns.json").read_text())
        numeric = config["raw_numeric_features"]
        features = config["model_feature_columns"]
        detail["feature_contract"] = {
            "numeric": len(numeric), "categorical": len(config["categorical_features"]),
            "forbidden_in_X": sorted(set(features) & {
                "course_id", "q", "y", "COELO", "AFELO", "ACELO",
                "course_end", "cutoff_time", "split", "q_alpha_0p1", "y_alpha_0p1",
                "q_alpha_1", "y_alpha_1",
            }),
            "categorical_in_distance": sorted(set(config["categorical_features"]) & set(config["distance_feature_columns"])),
        }
        view("supervised", folder + "/supervised_raw/*/*.parquet")
        detail["snapshot_contract"] = rows("""SELECT count(*) n_bad_pairs FROM (
            SELECT user_id,course_id FROM supervised GROUP BY 1,2 HAVING count(*)<>4
            OR count(DISTINCT phase)<>4 OR count(DISTINCT split)<>1
            OR count(DISTINCT q_alpha_0p1)<>1 OR count(DISTINCT q_alpha_1)<>1
            OR count(DISTINCT y_alpha_0p1)<>1 OR count(DISTINCT y_alpha_1)<>1)""")[0]
        detail["cutoff_fraction_check"] = rows("""SELECT phase,max(abs(epoch_us(s.cutoff_time)-
            (epoch_us(s.enroll_time)+(epoch_us(f.course_end)-epoch_us(s.enroll_time))*
             CASE s.phase WHEN 'P1' THEN 1 WHEN 'P2' THEN 1 WHEN 'P3' THEN 3 ELSE 9 END //
             CASE s.phase WHEN 'P1' THEN 4 WHEN 'P2' THEN 2 WHEN 'P3' THEN 4 ELSE 10 END)))
             max_error_microseconds FROM supervised s JOIN labeled f USING(user_id,course_id)
             GROUP BY 1 ORDER BY 1""")
        view("ready", folder + "/model_ready/*/*.parquet")
        detail["ready_phase_indicator"] = rows("""SELECT phase,min(phase_fraction) min_value,
            max(phase_fraction) max_value FROM ready GROUP BY 1 ORDER BY 1""")
        view("audits", folder + "/audit/*/*.parquet")
        detail["event_window_check"] = rows("""SELECT count(*) n_audit_rows,
            count(*) FILTER(WHERE n_out_of_window_events<>0 OR min_feature_event_time<enroll_time
                OR max_feature_event_time>cutoff_time+INTERVAL '1 microsecond') n_bad_audit_rows
            FROM audits""")[0]
        detail["missing_numeric_counts_by_phase_split"] = rows("SELECT phase,split,count(*) n," +
            ",".join(f"count(*) FILTER(WHERE {x} IS NULL) {x}" for x in numeric) +
            " FROM supervised GROUP BY 1,2 ORDER BY 1,2")
        detail["labeled_course_diagnostics"] = {}
        for suffix in ["0p1", "1"]:
            con.execute(f"""CREATE OR REPLACE TEMP VIEW diag_course AS SELECT split,course_id,
                count(*) n,avg(q_alpha_{suffix}) q_labeled,
                avg((y_alpha_{suffix}='E')::DOUBLE) pi_e,
                avg((y_alpha_{suffix}='G')::DOUBLE) pi_g,
                avg((y_alpha_{suffix}='A')::DOUBLE) pi_a
                FROM cohort GROUP BY 1,2""")
            stats = rows("""SELECT split,count(*) n_courses,min(n) min_n,
                quantile_cont(n,0.25) p25_n,median(n) median_n,quantile_cont(n,0.75) p75_n,
                max(n) max_n,count(*) FILTER(WHERE n<30) n_below_30,
                count(*) FILTER(WHERE n>=30) n_at_least_30,
                count(*) FILTER(WHERE q_labeled<0.2) courses_E,
                count(*) FILTER(WHERE q_labeled>=0.2 AND q_labeled<0.4) courses_G,
                count(*) FILTER(WHERE q_labeled>=0.4) courses_A,
                count(*) FILTER(WHERE n>=30 AND q_labeled<0.2) scored_courses_E,
                count(*) FILTER(WHERE n>=30 AND q_labeled>=0.2 AND q_labeled<0.4) scored_courses_G,
                count(*) FILTER(WHERE n>=30 AND q_labeled>=0.4) scored_courses_A
                FROM diag_course GROUP BY 1 ORDER BY 1""")
            means = rows(f"""SELECT y_alpha_{suffix} y,count(*) n,avg(q_alpha_{suffix}) mu
                FROM cohort WHERE split='train' GROUP BY 1 ORDER BY 1""")
            mu = {r["y"]: r["mu"] for r in means}
            floor = rows(f"""SELECT count(*) n_dev_courses_scored,
                avg(abs(pi_e*{mu['E']!r}+pi_g*{mu['G']!r}+pi_a*{mu['A']!r}-q_labeled))
                oracle_floor_labeled_only FROM diag_course WHERE split='dev' AND n>=30""")[0]
            detail["labeled_course_diagnostics"][suffix] = {
                "status": "Supplemental audit diagnostics; absent from V3 data card and not certified full-course Qc",
                "size_and_class_counts": stats, "train_class_means": means, "oracle_floor_dev_nmin30": floor,
            }
    evidence["warnings"] = [
        "Stored builder validation is not equivalent to Review section 7.11 checks 1-12.",
        "Diagnostics computed in this audit do not establish completion in the original pipeline/data card.",
        "Old V1/V2 audits or protocol constants must not be reported as V3 completion.",
    ]
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({
        "population": evidence["population"], "label_checks": evidence["label_checks"],
        "thresholds": evidence["thresholds"], "within_course": evidence["within_course_label_variation"],
        "chapter_aggregation": evidence["chapter_aggregation"],
        "formula_comparison": evidence["formula_comparison"],
        "stored_validation": evidence["stored_validation"],
        "variants": {k: {x: v[x] for x in ["overlap", "test_user_overlap", "time_overlap_diagnostic",
            "feature_contract", "snapshot_contract", "event_window_check", "labeled_course_diagnostics"]}
            for k, v in evidence["variants"].items()},
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    run()
