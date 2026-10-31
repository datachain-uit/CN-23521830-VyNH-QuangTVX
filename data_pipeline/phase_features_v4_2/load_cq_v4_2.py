"""Read local CD3 V4.2 raw/ready data; test requires a completed test extension."""
from pathlib import Path
import json
import duckdb

ROOT = Path(__file__).resolve().parent

def feature_columns(view="learner", root=None):
    if view not in ("learner", "selector"):
        raise ValueError("view must be learner or selector")
    return json.loads((Path(root or ROOT)/(view+"_feature_columns.json")).read_text())["columns"]

def teacher_subgroups(root=None, connection=None):
    root = Path(root or ROOT)
    contract = json.loads((root/"run_manifest.json").read_text()).get("teacher_subgroup")
    if not contract:
        raise ValueError("Teacher subgroup metadata is not built; run 29_build_teacher_subgroup_v4_2.py")
    con = connection or duckdb.connect()
    return con.read_parquet(str(root/contract["metadata_path"]))

def load(split="train", phase=None, view="learner", raw=False, root=None, connection=None,
         include_teacher_subgroup=False):
    root = Path(root or ROOT)
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    if split == "test" and not json.loads((root/"run_manifest.json").read_text()).get("test_in_bundle",False):
        raise ValueError("Test is not built in this bundle; test remains locked")
    if phase == "FULL_A":
        if split != "train":
            raise ValueError("FULL_A is train only")
        name = "fulltrain_a/train.parquet"
    elif split in ("dev", "test"):
        if phase not in ("P1", "P2", "P3", "P4"):
            raise ValueError("Specify "+split+" phase P1–P4")
        name = "pool_b/"+split+"_phase"+phase[1]+".parquet"
    else:
        if phase is not None and phase not in ("P1", "P2", "P3", "P4"):
            raise ValueError("Unknown phase")
        name = "pool_b/train.parquet"
    path = root/("raw_features" if raw else "model_ready")/name
    con = connection or duckdb.connect()
    rel = con.read_parquet(str(path))
    if phase is not None:
        rel = rel.filter("phase='"+phase+"'")
    extra = []
    if include_teacher_subgroup:
        extra = ["teacher_subgroup", "n_teachers", "n_seen_teachers", "n_new_teachers"]
        metadata = teacher_subgroups(root, con).project("course_id,"+",".join(extra))
        rel = rel.join(metadata, "course_id", how="left")
    if raw:
        return rel
    config = json.loads((root/"feature_columns.json").read_text())
    cols = config["metadata_columns"] + feature_columns(view, root) + extra
    return rel.project(",".join('"'+c+'"' for c in cols)+",q_alpha_1 AS q,y_alpha_1 AS y")
