"""Read CQ V4 using fixed feature order; never fit preprocessing."""
from pathlib import Path
import json
_CONNECTION = None

def _root(root):
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parent

def feature_columns(view="learner", root=None):
    if view not in {"learner", "selector"}:
        raise ValueError("view must be learner or selector")
    return json.loads((_root(root) / (view + "_feature_columns.json")).read_text())["columns"]

def load_phase(phase="P1", split="train", ready=True, view="learner", root=None, connection=None):
    global _CONNECTION
    import duckdb
    root = _root(root)
    if phase not in {"P1", "P2", "P3", "P4", None} or split not in {"train", "dev", "test"}:
        raise ValueError("invalid phase/split")
    runtime = (root / "bundle_manifest.json").exists()
    if runtime:
        if split == "test":
            raise PermissionError("This dev-only bundle has no test access")
        if not ready:
            raise ValueError("The runtime bundle provides ready data; raw is in the private phase root")
        if split == "dev" and phase is None:
            raise ValueError("Select a dev phase")
        path = root / "pool_b" / ("train.parquet" if split == "train" else "dev_phase" + phase[-1] + ".parquet")
    else:
        base = "model_ready" if ready else "supervised_raw"
        path = root / base / (phase or "*") / (split + ".parquet")
    if connection is None:
        if _CONNECTION is None:
            _CONNECTION = duckdb.connect()
        connection = _CONNECTION
    columns = feature_columns(view, root)
    metadata = ["row_id","enrollment_id","user_id","course_id","course_family_id","phase","split","enroll_time","cutoff_time","course_end"]
    select = ",".join('"' + x + '"' for x in metadata + columns)
    relation = connection.sql("SELECT " + select + ",q_alpha_1 q,y_alpha_1 y FROM read_parquet('" + str(path).replace("'","''") + "')")
    return relation.filter("phase='" + phase + "'") if phase else relation

def load_fullA(ready=True, view="learner", root=None, connection=None):
    global _CONNECTION
    import duckdb
    root = _root(root)
    runtime = (root / "bundle_manifest.json").exists()
    if runtime and not ready:
        raise ValueError("Runtime bundle has ready FullTrain-A only")
    path = root / "fulltrain_a" / ("train.parquet" if runtime else "train_" + ("ready" if ready else "raw") + ".parquet")
    if connection is None:
        if _CONNECTION is None:
            _CONNECTION = duckdb.connect()
        connection = _CONNECTION
    columns = feature_columns(view, root)
    meta = ["row_id","enrollment_id","user_id","course_id","course_family_id","phase","split","enroll_time","cutoff_time","course_end"]
    return connection.sql("SELECT " + ",".join('"'+x+'"' for x in meta+columns) + ",q_alpha_1 q,y_alpha_1 y FROM read_parquet('" + str(path).replace("'","''") + "')")
