"""Attach course-level teacher subgroup metadata without rewriting feature matrices."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil

import duckdb

ROOT = Path(__file__).resolve().parent
VERSION = "teacher-subgroup-train-reference-v1"
POLICY_FILES = ["preprocessing.json", "feature_columns.json", "learner_feature_columns.json",
                "selector_feature_columns.json", "audit/qa23_by_phase.json", "audit/held_out_columns.json"]
META_COLUMNS = ["teacher_subgroup", "n_teachers", "n_seen_teachers", "n_new_teachers"]
PATHS = {"metadata": "metadata/teacher_subgroup.parquet",
         "mapping": "metadata/course_teacher_mapping.parquet",
         "reference": "metadata/train_teacher_reference.json",
         "spec": "metadata/teacher_subgroup_spec.json",
         "audit": "audit/teacher_subgroup_validation.json"}
spec = importlib.util.spec_from_file_location("teacher_subgroup_base", ROOT/"25_build_phase_features_v4_2.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_index(bundle):
    success = read_json(bundle/"_SUCCESS.json")
    if (base.sha256(bundle/"run_manifest.json") != success["manifest_sha256"] or
            base.sha256(bundle/"SHA256SUMS.txt") != success["checksums_sha256"]):
        raise AssertionError("Existing bundle completion hashes do not match")
    for line in (bundle/"SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if base.sha256(bundle/name) != digest:
            raise AssertionError("Existing artifact hash mismatch: "+name)


def derive_teacher_subgroups(con):
    """Read courses/source_course_teacher views; retain relation IDs without profile joins."""
    con.execute("""CREATE OR REPLACE TEMP TABLE course_teacher_edges AS
        SELECT DISTINCT c.course_id,trim(cast(t.teacher_id AS VARCHAR)) teacher_id
        FROM source_course_teacher t JOIN courses c
          ON trim(cast(t.course_id AS VARCHAR))=c.course_id
        WHERE t.teacher_id IS NOT NULL AND trim(cast(t.teacher_id AS VARCHAR))<>''""")
    con.execute("""CREATE OR REPLACE TEMP TABLE train_teacher_ids AS
        SELECT DISTINCT teacher_id FROM course_teacher_edges JOIN courses USING(course_id)
        WHERE split='train'""")
    con.execute("""CREATE OR REPLACE TEMP TABLE teacher_groups AS
        WITH counts AS (
            SELECT c.course_id,c.course_family_id,c.split,
                count(e.teacher_id)::BIGINT n_teachers,
                count(r.teacher_id)::BIGINT n_seen_teachers,
                count(e.teacher_id) FILTER(WHERE r.teacher_id IS NULL)::BIGINT n_new_teachers
            FROM courses c LEFT JOIN course_teacher_edges e USING(course_id)
            LEFT JOIN train_teacher_ids r USING(teacher_id) GROUP BY 1,2,3)
        SELECT course_id,course_family_id,split,
            CASE WHEN n_teachers=0 THEN 'missing'
                 WHEN n_new_teachers=0 THEN 'seen'
                 WHEN n_seen_teachers=0 THEN 'new' ELSE 'mixed' END teacher_subgroup,
            n_teachers,n_seen_teachers,n_new_teachers FROM counts""")


def export_parquet(con, path, query):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+".partial")
    con.execute(f"COPY ({query}) TO {base.qs(partial)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    partial.replace(path)


def build(root=ROOT, bundle=None):
    root = Path(root).resolve()
    bundle = Path(bundle or root/"phase_features_v4_2").resolve()
    validate_index(bundle)
    manifest = read_json(bundle/"run_manifest.json")
    config = read_json(bundle/"feature_columns.json")
    protected = {p:base.sha256(bundle/p) for p in manifest["raw_paths"]+manifest["ready_paths"]+POLICY_FILES}
    source_dir = root/"input/moocubex_parquet/tables/relations/course_teacher"
    sources = sorted(source_dir.glob("*.parquet"))
    if not sources:
        raise FileNotFoundError(source_dir)
    source_hashes = {p.relative_to(root).as_posix():base.sha256(p) for p in sources}
    raw_paths = "["+",".join(base.qs(bundle/p) for p in manifest["raw_paths"])+"]"
    mapping_paths = "["+",".join(base.qs(p) for p in sources)+"]"
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1000MB'")
    checks = {}

    def check(name, passed):
        checks[name] = {"status":"PASS" if passed else "FAIL"}
        if not passed:
            raise AssertionError(name)

    try:
        con.execute(f"CREATE VIEW snapshots AS SELECT * FROM read_parquet({raw_paths})")
        con.execute("CREATE TEMP TABLE courses AS SELECT DISTINCT course_id,course_family_id,split FROM snapshots")
        con.execute(f"CREATE VIEW source_course_teacher AS SELECT * FROM read_parquet({mapping_paths})")
        check("course_has_one_family_and_split", con.execute(
            "SELECT count(*)=count(DISTINCT course_id) FROM courses").fetchone()[0])
        check("train_reference_available", con.execute("SELECT count(*)>0 FROM courses WHERE split='train'").fetchone()[0])
        derive_teacher_subgroups(con)
        check("one_metadata_row_per_course", con.execute("SELECT count(*)=count(DISTINCT course_id) AND count(*)=(SELECT count(*) FROM courses) FROM teacher_groups").fetchone()[0])
        check("counts_consistent", con.execute("SELECT count(*)=0 FROM teacher_groups WHERE n_teachers<>n_seen_teachers+n_new_teachers OR least(n_teachers,n_seen_teachers,n_new_teachers)<0").fetchone()[0])
        check("train_is_seen_or_missing", con.execute("SELECT count(*)=0 FROM teacher_groups WHERE split='train' AND teacher_subgroup NOT IN ('seen','missing')").fetchone()[0])
        check("no_teacher_metadata_in_X", not set(META_COLUMNS) & set(config['learner']+config['selector']))
        check("all_snapshots_match_metadata", con.execute("SELECT count(*)=0 FROM snapshots s LEFT JOIN teacher_groups t USING(course_id) WHERE t.course_id IS NULL OR s.split<>t.split OR s.course_family_id<>t.course_family_id").fetchone()[0])
        teacher_ids = [r[0] for r in con.execute("SELECT teacher_id FROM train_teacher_ids ORDER BY teacher_id").fetchall()]
        reference = {"version":VERSION,"reference_split":"train","scope":"All distinct courses in the frozen train cohort; includes P1-P4 and FULL_A without multiplicity",
                     "n_train_courses":con.execute("SELECT count(*) FROM courses WHERE split='train'").fetchone()[0],
                     "n_teacher_ids":len(teacher_ids),"teacher_ids_sha256":base.canonical_hash(teacher_ids),"teacher_ids":teacher_ids}
        prior_manifest = manifest.get("teacher_subgroup")
        old_reference = read_json(bundle/prior_manifest['reference_path']) if prior_manifest else reference
        check("frozen_train_teacher_reference", reference==old_reference)
        records = lambda query: con.execute(query).df().to_dict(orient="records")
        summary = records("""SELECT t.split,t.teacher_subgroup,count(DISTINCT t.course_id)::BIGINT courses,
            count(DISTINCT s.enrollment_id)::BIGINT enrollments,count(*)::BIGINT snapshots_including_full_a
            FROM teacher_groups t JOIN snapshots s USING(course_id) GROUP BY 1,2 ORDER BY 1,2""")
        source_summary = records("""SELECT count(*)::BIGINT raw_mapping_rows,
            count(*) FILTER(WHERE teacher_id IS NULL OR trim(cast(teacher_id AS VARCHAR))='')::BIGINT invalid_teacher_rows
            FROM source_course_teacher t JOIN courses c ON trim(cast(t.course_id AS VARCHAR))=c.course_id""")[0]
        source_summary['distinct_valid_edges'] = con.execute("SELECT count(*) FROM course_teacher_edges").fetchone()[0]
        contract = {"version":VERSION,"dataset_version":manifest['dataset_version'],
                    "grain":"One row per course_id; subgroup constant across all phases and enrollments of a course",
                    "join_key":"course_id","metadata_columns":META_COLUMNS,"excluded_from_X":True,
                    "reference_split":"train","reference_ids_sha256":reference['teacher_ids_sha256'],
                    "source":"input/moocubex_parquet/tables/relations/course_teacher/*.parquet",
                    "source_sha256":source_hashes,
                    "normalization":"Cast course_id/teacher_id to string, trim surrounding spaces; discard NULL/empty teacher_id; deduplicate (course_id,teacher_id); restrict to bundle courses",
                    "profile_requirement":"None; teacher relation IDs remain valid when entities/teacher has no matching profile",
                    "rules":{"missing":"n_teachers == 0",
                             "seen":"n_teachers > 0 and n_new_teachers == 0",
                             "new":"n_teachers > 0 and n_seen_teachers == 0",
                             "mixed":"n_seen_teachers > 0 and n_new_teachers > 0"},
                    "count_definitions":{"n_teachers":"Distinct usable teacher IDs for the course",
                                         "n_seen_teachers":"Intersection with the frozen train teacher ID reference",
                                         "n_new_teachers":"Course teacher IDs outside the frozen train reference"},
                    "missing_meaning":"No usable teacher ID in the relation mapping; does not establish that the course has no teachers",
                    "temporal_scope":"Final relationship snapshot at collection; no per-edge as-of history; retrospective analysis metadata, not an online feature",
                    "interpretation":"A relation to train teachers, independent of learner subgroup and categorical codes; no target values used",
                    "review_basis":["Review-2026-10-26.docx / Y31","Review-2026-10-27.docx / Y31"],
                    "model_inputs_changed":False}
        for p,h in protected.items():
            check('unchanged/'+p, base.sha256(bundle/p)==h)
        for p,h in source_hashes.items():
            check('source_unchanged/'+p, base.sha256(root/p)==h)
        report = {"version":VERSION,"status":"PASS","checks":checks,"checks_passed":len(checks),
                  "summary":summary,"source_summary":source_summary,"reference_ids_sha256":reference['teacher_ids_sha256'],
                  "model_input_sha256":protected,"model_inputs_changed":False}
        # Remove the completion marker only after derivation and validation succeed.
        history = bundle/"provenance/pre_teacher_subgroup"
        history.mkdir(parents=True, exist_ok=True)
        for name in ['run_manifest.json','_SUCCESS.json','SHA256SUMS.txt','README.md','load_cq_v4_2.py']:
            if not (history/name).exists():
                shutil.copyfile(bundle/name, history/name)
        success = read_json(bundle/'_SUCCESS.json')
        (bundle/'_SUCCESS.json').unlink()
        export_parquet(con, bundle/PATHS['metadata'], 'SELECT * FROM teacher_groups ORDER BY course_id')
        export_parquet(con, bundle/PATHS['mapping'], 'SELECT * FROM course_teacher_edges ORDER BY course_id,teacher_id')
        base.dump(bundle/PATHS['reference'], reference)
        base.dump(bundle/PATHS['spec'], contract)
        base.dump(bundle/PATHS['audit'], report)
        (bundle/'load_cq_v4_2.py').write_text(base.LOADER)
        readme = re.sub(r'\n## Teacher subgroup\n.*?(?=\n## |\Z)', '', (bundle/'README.md').read_text(), flags=re.S)
        readme += ('\n## Teacher subgroup\n\nOfficial course metadata: `metadata/teacher_subgroup.parquet`; '
                   'rules and train-only reference are in `metadata/teacher_subgroup_spec.json` and '
                   '`metadata/train_teacher_reference.json`. Load with `teacher_subgroups()` or '
                   '`load(..., include_teacher_subgroup=True)`; keep these metadata outside X.\n')
        (bundle/'README.md').write_text(readme)
        provenance = bundle/'provenance/teacher_subgroup'
        provenance.mkdir(parents=True, exist_ok=True)
        for source in [Path(__file__), ROOT/'25_build_phase_features_v4_2.py', ROOT/'26_verify_phase_features_v4_2.py']:
            shutil.copyfile(source, provenance/source.name)
        manifest['teacher_subgroup'] = {"version":VERSION,"join_key":"course_id","metadata_columns":META_COLUMNS,
                                        "excluded_from_X":True,"reference_split":"train",
                                        "reference_ids_sha256":reference['teacher_ids_sha256'],"model_inputs_changed":False}
        for key,path in PATHS.items():
            manifest['teacher_subgroup'][key+'_path'] = path
            manifest['teacher_subgroup'][key+'_sha256'] = base.sha256(bundle/path)
        base.dump(bundle/'run_manifest.json', manifest)
        if any(base.sha256(bundle/p)!=h for p,h in protected.items()):
            raise AssertionError('A model input changed during metadata installation')
        files = [p for p in sorted(bundle.rglob('*')) if p.is_file() and
                 not any(s in p.relative_to(bundle).parts for s in ['_stage','_scratch','__pycache__']) and
                 p not in [bundle/'SHA256SUMS.txt',bundle/'_SUCCESS.json'] and not p.name.endswith('.partial')]
        (bundle/'SHA256SUMS.txt').write_text(''.join(f'{base.sha256(p)}  {p.relative_to(bundle).as_posix()}\n' for p in files))
        success.update({'manifest_sha256':base.sha256(bundle/'run_manifest.json'),
                        'checksums_sha256':base.sha256(bundle/'SHA256SUMS.txt'),
                        'teacher_subgroup_version':VERSION,'model_inputs_changed':False})
        base.dump(bundle/'_SUCCESS.json', success)
        print(json.dumps({'status':'PASS','checks_passed':len(checks),'summary':summary,
                          'train_teacher_ids':len(teacher_ids),'protected_model_artifacts':len(protected)}, ensure_ascii=False, indent=2))
        return report
    finally:
        con.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    build(args.root, args.bundle)
