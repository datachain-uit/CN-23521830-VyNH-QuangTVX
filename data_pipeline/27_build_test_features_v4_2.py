"""Append CD3 test P1-P4 using the frozen V4.2 train preprocessing and QA23 policy.

This entry point explicitly reads the V4 test snapshots. It only builds and
validates features; it never fits preprocessing, runs QA23, or trains a model.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from cq_tempo_features import CQRawEarlyViews, TempoV1Preprocessor

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cq_v4_2_common_builder", ROOT/"25_build_phase_features_v4_2.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

FROZEN_POLICY = ["preprocessing.json", "feature_columns.json", "learner_feature_columns.json",
                 "selector_feature_columns.json", "audit/qa23_by_phase.json", "audit/held_out_columns.json"]


def restore_frozen_views(prep):
    """Restore JSON state without invoking either fit method."""
    views = CQRawEarlyViews()
    views.cols = copy.deepcopy(prep["columns"])
    views.v1 = TempoV1Preprocessor()
    views.v1.columns = list(views.cols["selector"])
    views.v1.log_columns = list(prep["log_columns"])
    views.v1.center = dict(prep["center"])
    views.v1.scale = dict(prep["scale"])
    views.maps = copy.deepcopy(prep["category_maps"])
    if set(views.v1.center) != set(views.v1.scale) or not set(views.v1.center) <= set(views.v1.columns):
        raise ValueError("Invalid frozen V1 statistics")
    if any(not np.isfinite(v) or v <= 0 for v in views.v1.scale.values()):
        raise ValueError("Invalid frozen V1 scale")
    if any(not np.isfinite(v) for v in views.v1.center.values()):
        raise ValueError("Invalid frozen V1 center")
    return views


def validate_bundle_index(bundle):
    success = json.loads((bundle/"_SUCCESS.json").read_text())
    if base.sha256(bundle/"run_manifest.json") != success["manifest_sha256"]:
        raise AssertionError("Existing manifest hash mismatch")
    if base.sha256(bundle/"SHA256SUMS.txt") != success["checksums_sha256"]:
        raise AssertionError("Existing checksum index hash mismatch")
    for line in (bundle/"SHA256SUMS.txt").read_text().splitlines():
        digest,name = line.split("  ",1)
        if base.sha256(bundle/name) != digest:
            raise AssertionError("Existing artifact hash mismatch: "+name)


def install_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix+".partial")
    shutil.copyfile(source, partial)
    partial.replace(target)


def build_test(root=ROOT, bundle=None, threads=2, memory="1200MB", batch_size=1500, tests_log=None):
    root = Path(root).resolve()
    bundle = Path(bundle or root/"phase_features_v4_2").resolve()
    validate_bundle_index(bundle)
    manifest = json.loads((bundle/"run_manifest.json").read_text())
    config = json.loads((bundle/"feature_columns.json").read_text())
    prep = json.loads((bundle/"preprocessing.json").read_text())
    if prep["fit_split"] != "train" or prep["fit_population"] != "TRAIN_POOLED_P1_P4":
        raise ValueError("A frozen train-only V4.2 preprocessor is required")
    if base.sha256(ROOT/"cq_tempo_features.py") != manifest["helper_sha256"]:
        raise ValueError("The CD3 helper differs from the helper used for the frozen train bundle")
    if prep["columns"] != {k:config[k] for k in ("learner","selector")}:
        raise ValueError("Frozen preprocessing and feature order disagree")
    existing_raw = [p for p in manifest["raw_paths"] if "/test_phase" not in p]
    existing_ready = [p for p in manifest["ready_paths"] if "/test_phase" not in p]
    if len(existing_raw) != 6 or len(existing_ready) != 6:
        raise ValueError("Expected the six frozen train/dev/FULL_A tables")
    protected = FROZEN_POLICY+existing_raw+existing_ready
    frozen_hashes = {p:base.sha256(bundle/p) for p in protected}
    work = base.Builder(root, bundle/"_stage/test_extension", threads, memory, batch_size,
                        snapshot_splits=("test",))
    started = time.monotonic()
    try:
        print("Building test only; restoring frozen train V1 and category maps", flush=True)
        work.prepare()
        work.build_raw()
        work.audit_events()
        con = work.con
        paths = "["+",".join(base.qs(bundle/p) for p in existing_raw)+"]"
        con.execute("CREATE VIEW frozen_existing AS SELECT row_id,enrollment_id,course_id,course_family_id,split FROM read_parquet("+paths+")")
        for key in ("row_id","enrollment_id","course_id","course_family_id"):
            work.sqlcheck("test_disjoint_"+key, f"SELECT count(*)=0 FROM raw_all t JOIN frozen_existing b USING({key})")
        work.sqlcheck("test_has_only_four_early_phases", "SELECT count(*)=0 FROM raw_all WHERE split<>'test' OR phase NOT IN ('P1','P2','P3','P4')")
        views = restore_frozen_views(prep)
        for name,path in work.outputs:
            frame = con.execute(f"SELECT * FROM read_parquet({base.qs(path)}) ORDER BY row_id").df()
            matrices = views.transform(frame)
            learner,selector = matrices["learner"],matrices["selector"]
            work.check(name+"_feature_order", list(learner.columns)==config["learner"] and list(selector.columns)==config["selector"])
            work.check(name+"_ready_finite", np.isfinite(learner.to_numpy()).all())
            work.check(name+"_shared_train_v1", learner[config["selector"]].equals(selector))
            work.check(name+"_phase_fraction_unscaled", learner.phase_fraction.equals(frame.phase_fraction.astype(float)))
            result = pd.concat([frame[base.META+base.TARGETS],learner],axis=1)
            ready = work.out/"model_ready"/(name+".parquet")
            work.frame_write(ready,result)
            con.execute(f"CREATE OR REPLACE VIEW exported_ready AS SELECT * FROM read_parquet({base.qs(ready)})")
            work.sqlcheck(name+"_ready_row_count", f"SELECT count(*)={len(frame)} AND count(*)=count(DISTINCT row_id) FROM exported_ready")
            print(name+": frozen train transform complete", flush=True)
        for path,digest in frozen_hashes.items():
            work.check("frozen_unchanged/"+path, base.sha256(bundle/path)==digest)
        for path,record in work.sources.items():
            p = Path(path)
            work.check("source_unchanged_"+base.canonical_hash(path)[:12],
                       p.stat().st_size==record["bytes"] and p.stat().st_mtime_ns==record["mtime_ns"])
            record["sha256"] = base.sha256(p)
        work.check("no_test_fit_or_qa23", prep==json.loads((bundle/"preprocessing.json").read_text())
                   and all(base.sha256(bundle/p)==frozen_hashes[p] for p in FROZEN_POLICY))
        counts = base.json_records(con.execute("SELECT phase,split,count(*) n_rows,count(DISTINCT enrollment_id) enrollments FROM raw_all GROUP BY 1,2 ORDER BY 1,2").df())
        report = {"status":"PASS", "checks_passed":len(work.checks), "counts":counts,
                  "test_used_for_fit":False, "test_used_for_qa23":False, "model_training_run":False,
                  "frozen_artifacts_sha256":frozen_hashes, "checks":work.checks}
        base.dump(work.out/"audit/test_validation.json",report)
        extension = {"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),
                     "authorization":"Explicit user request to build test CD3 features",
                     "preprocessing_sha256":frozen_hashes["preprocessing.json"],
                     "feature_columns_sha256":frozen_hashes["feature_columns.json"],
                     "qa23_sha256":frozen_hashes["audit/qa23_by_phase.json"],
                     "helper_sha256":manifest["helper_sha256"], "builder_sha256":base.sha256(__file__),
                     "common_builder_sha256":base.sha256(ROOT/"25_build_phase_features_v4_2.py"),
                     "input_signature":work.signature, "checks_passed":len(work.checks),
                     "seconds":time.monotonic()-started, "threads":threads,"memory":memory,"batch_size":batch_size}
        # Keep the original dev-only provenance, and leave its data/fit artifacts
        # byte-for-byte intact. Installation starts only after all test checks pass.
        history = bundle/"provenance/pre_test_extension"
        history.mkdir(parents=True,exist_ok=True)
        for name in ("run_manifest.json","summary.json","SHA256SUMS.txt","_SUCCESS.json","audit/validation.json"):
            target = history/name
            if not target.exists():
                install_file(bundle/name,target)
        (bundle/"_SUCCESS.json").unlink(missing_ok=True)
        for name,path in work.outputs:
            install_file(path,bundle/"raw_features"/(name+".parquet"))
            install_file(work.out/"model_ready"/(name+".parquet"),bundle/"model_ready"/(name+".parquet"))
        for path in (work.out/"audit").iterdir():
            install_file(path,bundle/"audit/test"/path.name)
        base.dump(bundle/"audit/test_extension_validation.json",report)
        base.dump(bundle/"provenance/test_source_manifest.json",work.sources)
        base.dump(bundle/"provenance/test_extension_manifest.json",extension)
        provenance = bundle/"provenance/test_extension"
        for name in (Path(__file__).name,"25_build_phase_features_v4_2.py","26_verify_phase_features_v4_2.py"):
            install_file(ROOT/name,provenance/name)
        if tests_log is not None:
            install_file(Path(tests_log),provenance/"tests.log")
        validation = json.loads((bundle/"audit/validation.json").read_text())
        validation = {k:v for k,v in validation.items() if not k.startswith("test_extension/")}
        validation.update({"test_extension/"+k:v for k,v in work.checks.items()})
        base.dump(bundle/"audit/validation.json",validation)
        manifest.update({"test_data_read":True,"test_in_bundle":True,"test_used_for_fit":False,
                         "test_used_for_qa23":False,"test_extension":extension})
        manifest["raw_paths"] = existing_raw+["raw_features/"+n+".parquet" for n,_ in work.outputs]
        manifest["ready_paths"] = existing_ready+["model_ready/"+n+".parquet" for n,_ in work.outputs]
        base.dump(bundle/"run_manifest.json",manifest)
        summary = json.loads((bundle/"summary.json").read_text())
        summary["counts"] = [r for r in summary["counts"] if r["split"]!="test"]+counts
        summary.update({"test_extension_seconds":time.monotonic()-started,"checks_passed":len(validation),"test_in_bundle":True})
        base.dump(bundle/"summary.json",summary)
        (bundle/"load_cq_v4_2.py").write_text(base.LOADER,encoding="utf-8")
        readme = (bundle/"README.md").read_text()
        readme = readme.split("\n## Test extension\n")[0]
        readme = readme.replace("Built from frozen V4 train/dev/FULL_A.","Built from frozen V4 train/dev/test/FULL_A.")
        readme = readme.replace("FULL_A is transform-only.","dev, test and FULL_A are transform-only.")
        readme = readme.replace("it rejects test.","test loading requires a completed test extension and an explicit phase.")
        readme += "\n## Test extension\n\n"
        readme += "Test P1–P4 was appended from the original V4 test spine by explicit user request.\n"
        readme += "raw_features/pool_b/test_phase1.parquet through test_phase4.parquet contain raw test snapshots.\n"
        readme += "model_ready/pool_b/test_phase1.parquet through test_phase4.parquet contain the frozen train transform.\n"
        readme += "Use load(split='test', phase='P1', view='learner') or view='selector'; raw=True returns raw features.\n"
        readme += "No test data were used for fitting, QA23, feature selection or model training.\n"
        readme += "Original train/dev/FULL_A files, preprocessing, feature order and QA23 reports retain their original hashes.\n"
        readme += "audit/test_extension_validation.json records these checks; provenance/pre_test_extension retains the original dev-only manifests.\n"
        readme += "Reproduce the append step: python 27_build_test_features_v4_2.py; verify with python 26_verify_phase_features_v4_2.py.\n"
        (bundle/"README.md").write_text(readme,encoding="utf-8")
        for path,digest in frozen_hashes.items():
            if base.sha256(bundle/path)!=digest:
                raise AssertionError("Protected artifact changed during installation: "+path)
        paths = [p for p in sorted(bundle.rglob("*")) if p.is_file() and not any(
            s in p.relative_to(bundle).parts for s in ("_stage","_scratch","__pycache__"))
                 and p not in (bundle/"SHA256SUMS.txt",bundle/"_SUCCESS.json") and not p.name.endswith(".partial")]
        (bundle/"SHA256SUMS.txt").write_text("".join(f"{base.sha256(p)}  {p.relative_to(bundle).as_posix()}\n" for p in paths))
        base.dump(bundle/"_SUCCESS.json", {"dataset_version":base.VERSION,"checks_passed":len(validation),
                  "test_checks_passed":len(work.checks),"manifest_sha256":base.sha256(bundle/"run_manifest.json"),
                  "checksums_sha256":base.sha256(bundle/"SHA256SUMS.txt")})
        print(f"SUCCESS: test P1-P4 appended; {len(work.checks)} checks; {bundle}",flush=True)
        return report
    finally:
        work.con.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--bundle",type=Path,default=ROOT/"phase_features_v4_2")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--memory",default="1200MB")
    parser.add_argument("--batch-size",type=int,default=1500)
    parser.add_argument("--tests-log",type=Path)
    args=parser.parse_args()
    if args.batch_size<1:
        parser.error("batch-size must be positive")
    build_test(args.root,args.bundle,args.threads,args.memory,args.batch_size,args.tests_log)
