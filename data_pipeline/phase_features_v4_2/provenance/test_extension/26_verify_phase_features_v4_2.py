"""Independently verify a dev-only or complete V4.2 CD3 bundle from its manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import duckdb


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):
            h.update(b)
    return h.hexdigest()


def qs(value):
    return "'"+str(value).replace("'","''")+"'"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def verify(root):
    root=Path(root).resolve()
    checks={}

    def check(name,passed):
        checks[name]=bool(passed)
        if not passed:
            raise AssertionError(name)

    success=json.loads((root/"_SUCCESS.json").read_text())
    manifest=json.loads((root/"run_manifest.json").read_text())
    config=json.loads((root/"feature_columns.json").read_text())
    prep=json.loads((root/"preprocessing.json").read_text())
    check("success_manifest_hash",sha256(root/"run_manifest.json")==success["manifest_sha256"])
    check("success_checksum_index_hash",sha256(root/"SHA256SUMS.txt")==success["checksums_sha256"])
    for line in (root/"SHA256SUMS.txt").read_text().splitlines():
        digest,name=line.split("  ",1)
        check("checksum/"+name,sha256(root/name)==digest)
    check("builder_validation_passed",all(v["status"]=="PASS" for v in json.loads((root/"audit/validation.json").read_text()).values()))
    has_test=manifest.get("test_in_bundle",False)
    check("test_scope_matches_manifest",manifest["test_data_read"] is has_test)
    names=["pool_b/train.parquet"]+[f"pool_b/dev_phase{i}.parquet" for i in range(1,5)]+["fulltrain_a/train.parquet"]
    if has_test:
        names += [f"pool_b/test_phase{i}.parquet" for i in range(1,5)]
        check("test_not_used_for_fit_or_qa23",manifest["test_used_for_fit"] is False and manifest["test_used_for_qa23"] is False)
        extension=json.loads((root/"audit/test_extension_validation.json").read_text())
        check("test_extension_passed",extension["status"]=="PASS" and all(x["status"]=="PASS" for x in extension["checks"].values()))
        for name,digest in extension["frozen_artifacts_sha256"].items():
            check("frozen_train_bundle/"+name,sha256(root/name)==digest)
        check("frozen_train_preprocessing",sha256(root/"preprocessing.json")==manifest["test_extension"]["preprocessing_sha256"])
        check("frozen_train_qa23",sha256(root/"audit/qa23_by_phase.json")==manifest["test_extension"]["qa23_sha256"])
    check("complete_raw_and_ready_inventory",manifest["raw_paths"]==["raw_features/"+n for n in names]
          and manifest["ready_paths"]==["model_ready/"+n for n in names])
    check("raw_inventory",len(config["raw_feature_columns"])==85 and len(config["mask_columns"])==3)
    check("view_relation",config["learner"]==config["selector"]+["gender","phase_fraction"])
    check("no_metadata_targets_or_masks_in_X",not(set(config["learner"]) & set(config["metadata_columns"]+config["target_columns"]+config["mask_columns"])))
    for view in ["learner","selector"]:
        f=json.loads((root/(view+"_feature_columns.json")).read_text())
        check(view+"_feature_hash",f["columns"]==config[view] and canonical_hash(f["columns"])==f["columns_sha256"])
    check("held_out_union_matches_views",len(config["selector"])==74-len(prep["held_out_columns"]))
    con=duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1000MB'")
    counts=[]
    try:
        for raw,ready in zip(manifest["raw_paths"],manifest["ready_paths"]):
            con.execute(f"CREATE OR REPLACE VIEW r AS SELECT * FROM read_parquet({qs(root/raw)})")
            con.execute(f"CREATE OR REPLACE VIEW m AS SELECT * FROM read_parquet({qs(root/ready)})")
            expected=config["metadata_columns"]+config["target_columns"]+config["learner"]
            check(ready+"/schema",[r[0] for r in con.execute("DESCRIBE m").fetchall()]==expected)
            check(raw+"/unique_snapshot_keys",con.execute("SELECT count(*)=count(DISTINCT row_id) AND count(*)=count(DISTINCT(enrollment_id,phase)) FROM r").fetchone()[0])
            metadata=config["metadata_columns"]+config["target_columns"]
            differences=" OR ".join("r."+c+" IS DISTINCT FROM m."+c for c in metadata)
            check(ready+"/alignment",con.execute("SELECT count(*)=0 FROM r FULL JOIN m USING(row_id) WHERE r.row_id IS NULL OR m.row_id IS NULL OR "+differences).fetchone()[0])
            check(ready+"/finite",con.execute("SELECT count(*)=0 FROM m WHERE "+" OR ".join(c+" IS NULL OR NOT isfinite("+c+")" for c in config["learner"])).fetchone()[0])
            expressions=[]
            for c in config["selector"]:
                if c=="most_common_hour_sin":
                    source="sin(2*pi()*r.most_common_hour/24)"
                elif c=="most_common_hour_cos":
                    source="cos(2*pi()*r.most_common_hour/24)"
                else:
                    source="r."+c
                if c in prep["log_columns"]:
                    source=f"CASE WHEN {source}>=0 THEN ln(1+{source}) ELSE {source} END"
                if c in prep["center"]:
                    expected=f"coalesce((({source})-({prep['center'][c]!r}))/({prep['scale'][c]!r}),0)"
                else:
                    expected="0.0"
                expressions.append(f"abs(m.{c}-({expected}))>1e-8")
            check(ready+"/raw_to_V1",con.execute("SELECT count(*)=0 FROM r JOIN m USING(row_id) WHERE "+" OR ".join(expressions)).fetchone()[0])
            check(ready+"/unscaled_phase_fraction",con.execute("SELECT count(*)=0 FROM r JOIN m USING(row_id) WHERE r.phase_fraction<>m.phase_fraction").fetchone()[0])
            cases=" ".join(f"WHEN {('r.gender IS NULL' if k=='<MISSING>' else 'cast(r.gender AS VARCHAR)='+qs(k))} THEN {v}" for k,v in prep["category_maps"]["gender"].items())
            check(ready+"/train_gender_encoding",con.execute("SELECT count(*)=0 FROM r JOIN m USING(row_id) WHERE m.gender<>CASE "+cases+" ELSE 0 END").fetchone()[0])
            counts.extend(con.execute("SELECT phase,split,count(*) FROM r GROUP BY 1,2 ORDER BY 1,2").fetchall())
            if "/test_phase" in raw:
                phase="P"+Path(raw).stem[-1]
                check(raw+"/test_only",con.execute("SELECT count(*)=0 FROM r WHERE split<>'test' OR phase<>"+qs(phase)).fetchone()[0])
                source=root.parent/"phase_features_v4/supervised_raw"/phase/"test.parquet"
                if source.exists():
                    con.execute(f"CREATE OR REPLACE VIEW test_source AS SELECT * FROM read_parquet({qs(source)})")
                    diff=" OR ".join("r."+c+" IS DISTINCT FROM s."+c for c in metadata)
                    check(raw+"/frozen_v4_metadata_targets",con.execute("SELECT count(*)=0 FROM r FULL JOIN test_source s USING(row_id) WHERE r.row_id IS NULL OR s.row_id IS NULL OR "+diff).fetchone()[0])
        allpaths="["+",".join(qs(root/p) for p in manifest["raw_paths"])+"]"
        con.execute("CREATE VIEW all_exported AS SELECT row_id,enrollment_id,course_id,course_family_id,phase,split FROM read_parquet("+allpaths+")")
        check("global_snapshot_keys_unique",con.execute("SELECT count(*)=count(DISTINCT row_id) FROM all_exported").fetchone()[0])
        for key in ("enrollment_id","course_id","course_family_id"):
            check("split_disjoint/"+key,con.execute(f"SELECT count(*)=0 FROM (SELECT {key} FROM all_exported GROUP BY 1 HAVING count(DISTINCT split)>1)").fetchone()[0])
        check("every_enrollment_four_early_phases",con.execute("SELECT count(*)=0 FROM (SELECT enrollment_id FROM all_exported WHERE phase<>'FULL_A' GROUP BY 1 HAVING count(*)<>4 OR count(DISTINCT phase)<>4)").fetchone()[0])
        train=root/"raw_features/pool_b/train.parquet"
        ids=[r[0] for r in con.execute(f"SELECT row_id FROM read_parquet({qs(train)}) ORDER BY row_id").fetchall()]
        check("exact_train_fit_population",len(ids)==prep["fit_rows"] and canonical_hash(ids)==prep["fit_row_ids_sha256"])
        con.execute(f"CREATE VIEW train_raw AS SELECT * FROM read_parquet({qs(train)})")
        check("static_fit_cohort_size",con.execute("SELECT count(DISTINCT enrollment_id) FROM train_raw").fetchone()[0]==prep["static_fit_rows"])
        # Recompute train quantiles in SQL, independently of the pandas helper.
        static={"resource_count","ex_counts","teacher_count","teacher_avg_bio_length_chars","teacher_org_count",
                "school_count","school_avg_bio_length_chars","school_avg_motto_length_chars"}
        for c in config["selector"]:
            expr="sin(2*pi()*most_common_hour/24)" if c.endswith("hour_sin") else "cos(2*pi()*most_common_hour/24)" if c.endswith("hour_cos") else c
            if c in prep["log_columns"]:
                expr=f"CASE WHEN {expr}>=0 THEN ln(1+{expr}) ELSE {expr} END"
            source="(SELECT DISTINCT enrollment_id,"+c+" FROM train_raw)" if c in static else "train_raw"
            q=con.execute(f"SELECT quantile_cont({expr},[.25,.5,.75]) FROM {source}").fetchone()[0]
            if q is None or q[0] is None:
                check("train_quantiles/"+c,c not in prep["center"])
            else:
                scale=q[2]-q[0]
                if scale<2.220446049250313e-15:
                    scale=1.0
                check("train_quantiles/"+c,abs(q[1]-prep["center"][c])<1e-10 and abs(scale-prep["scale"][c])<1e-10)
        spec=importlib.util.spec_from_file_location("cq_v4_2_delivered_loader",root/"load_cq_v4_2.py")
        loader=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        try:
            loader.load("test")
        except ValueError:
            check("loader_test_requires_phase" if has_test else "loader_test_locked",True)
        else:
            check("loader_test_requires_phase" if has_test else "loader_test_locked",False)
        if has_test:
            for i in range(1,5):
                rel=loader.load("test",phase=f"P{i}",view="selector",connection=con)
                check(f"loader_test_P{i}_columns",rel.columns==config["metadata_columns"]+config["selector"]+["q","y"])
                check(f"loader_test_P{i}_rows",rel.count("*").fetchone()[0]==sum(n for p,s,n in counts if p==f"P{i}" and s=="test"))
        check("loader_selector_columns",loader.load(view="selector",connection=con).columns==config["metadata_columns"]+config["selector"]+["q","y"])
    finally:
        con.close()
    return {"status":"PASS","checks_passed":len(checks),"counts":counts,"checks":checks}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    pipeline=Path(__file__).resolve().parent
    parser.add_argument("--root",type=Path,default=pipeline/"phase_features_v4_2")
    parser.add_argument("--report",type=Path,default=pipeline/"outputs/phase_features_v4_2_validation.json")
    args=parser.parse_args()
    result=verify(args.root)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+"\n")
    print(f"PASS: {result['checks_passed']} independent checks; {args.report}")
