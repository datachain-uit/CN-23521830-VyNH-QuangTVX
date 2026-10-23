"""Adapter dev-only cho CQ V4 theo Review 22/10/2026.

Đầu vào bắt buộc dùng ``course_family_id``, split family-temporal đã khóa và
không chứa bất kỳ test parquet nào. Adapter không fit lại preprocessing; nó chỉ
kiểm tra hai feature view do nhóm dữ liệu bàn giao, tạo mảng runtime và xác nhận
FullTrain-A lambda=1 có đúng cohort train của Pool B.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...artifacts import atomic_json, sha256_file, stable_hash
from ..registry import get_dataset_spec
from .base import DatasetAdapter


PHASES = ("P1", "P2", "P3", "P4")
LABEL_MAPPING = {"E": 0, "G": 1, "A": 2}


class CourseQualityV4Adapter(DatasetAdapter):
    """Chuẩn hóa gói CQ V4 thành contract runtime nhưng giữ test khóa cứng."""

    dataset_id = "course_quality_v4"

    def prepare(
        self,
        input_dir: str | Path,
        processed_root: str | Path,
        *,
        overwrite: bool = False,
        validate_expected_rows: bool = True,
    ) -> Path:
        del validate_expected_rows
        source = Path(input_dir).resolve()
        output = Path(processed_root).resolve() / self.dataset_id
        manifest_path = output / "preprocessing_manifest.json"
        if manifest_path.exists() and not overwrite:
            return manifest_path
        if output.exists() and overwrite:
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

        required = (
            "validation.json", "splits.json", "run_manifest.json",
            "selector_feature_columns.json", "learner_feature_columns.json",
            "preprocessing.json", "bundle_manifest.json", "SHA256SUMS.txt",
            "family_splits.parquet", "course_splits.parquet",
        )
        missing = [name for name in required if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"CQ V4 thiếu artifact bắt buộc: {missing}")
        _verify_sha256s(source)
        test_files = sorted(
            path.relative_to(source).as_posix()
            for path in source.rglob("*")
            if path.is_file() and "test" in path.name.lower()
            and path.suffix.lower() in {".parquet", ".npy", ".csv"}
        )
        if test_files:
            raise ValueError(f"CQ V4 dev bundle không được chứa test data: {test_files}")

        validation = _read_json(source / "validation.json")
        if validation.get("dataset_version") not in {None, "CQ V4", "cq_v4"}:
            raise ValueError("validation.json không khai báo dataset_version CQ V4")
        bundle_manifest = _read_json(source / "bundle_manifest.json")
        test_excluded = validation.get(
            "test_excluded", bundle_manifest.get("test_excluded"),
        )
        if test_excluded is not True:
            raise ValueError("CQ V4 cần validation.test_excluded=true")
        splits = _read_json(source / "splits.json")
        group_unit = splits.get("group_unit")
        if group_unit is None and _has_family_assignment(splits):
            group_unit = "course_family_id"
        if str(group_unit) != "course_family_id":
            raise ValueError("CQ V4 splits.json phải dùng group_unit=course_family_id")
        if int(splits.get("seed", 42)) != 42:
            raise ValueError("CQ V4 train/dev split phải dùng seed 42")

        selector_columns = _column_list(source / "selector_feature_columns.json")
        learner_columns = _column_list(source / "learner_feature_columns.json")
        if not selector_columns or not set(selector_columns) <= set(learner_columns):
            raise ValueError("Feature view CQ V4 rỗng hoặc selector không thuộc learner view")
        forbidden = {
            "user_id", "course_id", "course_family_id", "phase", "split",
            "y_alpha_1", "y_alpha_0p1",
        }
        leaked = sorted(forbidden & (set(selector_columns) | set(learner_columns)))
        if leaked:
            raise ValueError(f"CQ V4 feature view làm rò ID/split/target: {leaked}")

        pool = source / "pool_b"
        train = pd.read_parquet(pool / "train.parquet")
        _require_columns(train, {
            "user_id", "course_id", "course_family_id", "phase", "y_alpha_1",
            *selector_columns, *learner_columns,
        }, "pool_b/train")
        if train["course_family_id"].isna().any():
            raise ValueError("CQ V4 train thiếu course_family_id")
        phase_values = train["phase"].astype(str).str.upper()
        unknown_phases = sorted(set(phase_values) - set(PHASES))
        if unknown_phases:
            raise ValueError(f"CQ V4 train có phase lạ: {unknown_phases}")
        X_train = _finite_matrix(train, learner_columns, "train learner")
        X_selector = _finite_matrix(train, selector_columns, "train selector")
        y_train = _labels(train["y_alpha_1"], "train")
        row_ids_train = _row_ids(train)
        phase_train = phase_values.map({phase: i for i, phase in enumerate(PHASES)}).to_numpy(np.int8)
        family_train = _group_ids(train["course_family_id"])
        course_train = _group_ids(train["course_id"])
        _save(output, "X_train", X_train)
        _save(output, "X_selector_train", X_selector)
        _save(output, "y_train", y_train)
        _save(output, "row_ids_train", row_ids_train)
        _save(output, "phase_train", phase_train)
        _save(output, "course_family_groups_train", family_train)
        _save(output, "course_groups_train", course_train)

        rows_by_split_phase: dict[str, dict[str, int]] = {"train": {}}
        rows_by_split_phase["train"] = {
            phase: int(np.count_nonzero(phase_values == phase)) for phase in PHASES
        }
        dev_families: set[int] = set()
        canonical_rows_saved = True
        for position, phase in enumerate(PHASES, start=1):
            frame = pd.read_parquet(pool / f"dev_phase{position}.parquet")
            _require_columns(frame, {
                "user_id", "course_id", "course_family_id", "y_alpha_1", *learner_columns,
            }, f"dev_phase{position}")
            X = _finite_matrix(frame, learner_columns, f"dev {phase}")
            y = _labels(frame["y_alpha_1"], f"dev {phase}")
            rows = _row_ids(frame, phase=phase)
            if "row_id" in frame:
                _save(output, f"source_row_ids_dev_phase{position}", frame.row_id.astype(str).to_numpy(dtype=str))
            else:
                canonical_rows_saved = False
            families = _group_ids(frame["course_family_id"])
            courses = _group_ids(frame["course_id"])
            dev_families.update(int(value) for value in np.unique(families))
            for stem, value in (
                (f"X_dev_phase{position}", X),
                (f"y_dev_phase{position}", y),
                (f"row_ids_dev_phase{position}", rows),
                (f"course_family_groups_dev_phase{position}", families),
                (f"course_groups_dev_phase{position}", courses),
            ):
                _save(output, stem, value)
            rows_by_split_phase.setdefault("dev", {})[phase] = int(len(frame))
        for stem in ("X", "y", "row_ids"):
            shutil.copy2(output / f"{stem}_dev_phase4.npy", output / f"{stem}_dev.npy")

        train_families = set(int(value) for value in np.unique(family_train))
        if train_families & dev_families:
            raise ValueError("CQ V4 course_family_id overlap giữa train và dev")

        full_path = source / "fulltrain_a" / "train.parquet"
        if not full_path.is_file():
            raise FileNotFoundError("CQ V4 thiếu fulltrain_a/train.parquet lambda=1")
        full = pd.read_parquet(full_path)
        _require_columns(full, {
            "user_id", "course_id", "course_family_id", "y_alpha_1", *learner_columns,
        }, "fulltrain_a/train")
        train_cohort = set(map(tuple, train[["user_id", "course_id"]].drop_duplicates().to_numpy()))
        full_cohort = set(map(tuple, full[["user_id", "course_id"]].drop_duplicates().to_numpy()))
        if train_cohort != full_cohort:
            raise ValueError(
                "FullTrain-A không cùng cohort train Pool B: "
                f"missing={len(train_cohort-full_cohort)}, extra={len(full_cohort-train_cohort)}"
            )
        _save(output, "X_fulltrain_a_train", _finite_matrix(full, learner_columns, "FullTrain-A"))
        _save(output, "y_fulltrain_a_train", _labels(full["y_alpha_1"], "FullTrain-A"))
        _save(output, "row_ids_fulltrain_a_train", _row_ids(full, phase="lambda1"))
        _save(output, "course_groups_fulltrain_a_train", _group_ids(full["course_id"]))
        _save(
            output,
            "course_family_groups_fulltrain_a_train",
            _group_ids(full["course_family_id"]),
        )

        contract = {
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "dataset_version": "CQ V4",
            "target_column": "y_alpha_1",
            "group_unit": "course_family_id",
            "split_policy": get_dataset_spec(self.dataset_id).split_policy,
            "selector_columns": selector_columns,
            "learner_columns": learner_columns,
            "preprocessing_scope": "fit_on_pool_b_train_v4_by_data_builder",
            "adapter_refits_preprocessing": False,
            "fulltrain_a_contract": "lambda_1_same_train_cohort",
            "test_locked": True,
            "test_data_packaged": False,
            "canonical_row_ids_saved": canonical_rows_saved,
        }
        contract["contract_hash"] = stable_hash(contract)
        atomic_json(output / "feature_contract.json", contract)
        atomic_json(output / "schema.json", contract)
        atomic_json(output / "split_manifest.json", {
            "dataset_id": self.dataset_id,
            "dataset_version": "CQ V4",
            "group_unit": "course_family_id",
            "split_seed": 42,
            "rows_by_split_phase": rows_by_split_phase,
            "train_dev_family_overlap": 0,
            "test_locked": True,
            "test_data_packaged": False,
            "splits_source_sha256": sha256_file(source / "splits.json"),
        })
        source_files = sorted(path for path in source.rglob("*") if path.is_file())
        atomic_json(manifest_path, {
            "status": "ready_for_m0_v4_dev",
            "dataset_version": "CQ V4",
            "adapter": f"{type(self).__module__}.{type(self).__qualname__}",
            "feature_contract_hash": contract["contract_hash"],
            "source_files_sha256": {
                path.relative_to(source).as_posix(): sha256_file(path) for path in source_files
            },
        })
        return manifest_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _column_list(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(value) for value in payload]
    for key in ("columns", "feature_columns", "selector_columns", "learner_columns"):
        if key in payload:
            return [str(value) for value in payload[key]]
    raise ValueError(f"Không đọc được danh sách feature từ {path.name}")


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} thiếu cột: {missing}")


def _finite_matrix(frame: pd.DataFrame, columns: list[str], name: str) -> np.ndarray:
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} có NaN/Inf; adapter V4 không tự fit/impute")
    return matrix


def _labels(series: pd.Series, name: str) -> np.ndarray:
    values = series.astype(str).str.upper()
    unknown = sorted(set(values) - set(LABEL_MAPPING))
    if unknown:
        raise ValueError(f"{name} có label ngoài E/G/A: {unknown}")
    return values.map(LABEL_MAPPING).to_numpy(np.int64)


def _row_ids(frame: pd.DataFrame, phase: str | None = None) -> np.ndarray:
    if "row_id" in frame:
        identity = frame["row_id"].astype(str)
    else:
        identity = frame[["user_id", "course_id"]].copy()
        identity["phase"] = frame["phase"].astype(str) if "phase" in frame else str(phase)
    values = pd.util.hash_pandas_object(identity, index=False).to_numpy(np.uint64)
    if len(np.unique(values)) != len(values):
        raise ValueError("CQ V4 có row_id trùng trong một bảng")
    return values


def _group_ids(series: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(series.astype(str), index=False).to_numpy(np.uint64)


def _has_family_assignment(splits: dict[str, Any]) -> bool:
    assignment = splits.get("course_assignment")
    if not isinstance(assignment, dict) or not assignment:
        return False
    return all(
        isinstance(value, dict) and value.get("course_family_id") not in {None, ""}
        for value in assignment.values()
    )


def _verify_sha256s(source: Path) -> None:
    failures: list[str] = []
    checked = 0
    for raw_line in (source / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            failures.append(f"malformed:{line}")
            continue
        path = source / relative.lstrip("*")
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        checked += 1
        if sha256_file(path) != expected.lower():
            failures.append(f"mismatch:{relative}")
    if checked == 0 or failures:
        raise ValueError(
            "CQ V4 SHA256SUMS không hợp lệ: "
            f"checked={checked}, failures={failures}"
        )


def _save(output: Path, stem: str, value: np.ndarray) -> None:
    np.save(output / f"{stem}.npy", value)
