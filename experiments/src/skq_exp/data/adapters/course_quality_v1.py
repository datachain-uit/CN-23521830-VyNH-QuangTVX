"""Adapter CQ v1 enrollment-phase cho Pool B và bốn phase đánh giá.

V3 được đọc từ ``course_random_split/supervised_raw`` rồi fit một phép tiền xử
lý chung trên toàn Pool B train.  Nhờ vậy ``phase_fraction`` còn phân biệt P1--P4
thay vì bị triệt tiêu bởi scaler riêng phase.  Gói M0 dev có thể cố ý không chứa
test; adapter và loader chỉ cho phép train/dev trong trường hợp đó.  Mọi artifact
vẫn development-only cho tới khi review chấp thuận split và xác nhận
``course_id`` chính là group/root unit.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ...artifacts import atomic_json, sha256_file, stable_hash
from ..registry import get_dataset_spec
from .base import DatasetAdapter


PHASES = ("P1", "P2", "P3", "P4")
SPLITS = ("train", "dev", "test")
LABEL_MAPPING = {"E": 0, "G": 1, "A": 2}


class _CourseQualityV1Base(DatasetAdapter):
    target_column: str
    target_q_column: str

    def prepare(
        self,
        input_dir: str | Path,
        processed_root: str | Path,
        *,
        overwrite: bool = False,
        validate_expected_rows: bool = True,
    ) -> Path:
        source = Path(input_dir).resolve()
        output = Path(processed_root).resolve() / self.dataset_id
        manifest_path = output / "preprocessing_manifest.json"
        if manifest_path.exists() and not overwrite:
            return manifest_path
        if output.exists() and overwrite:
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

        required_metadata = (
            "_SUCCESS.json", "validation.json", "feature_columns.json",
            "preprocessing.json", "run_manifest.json",
        )
        missing = [name for name in required_metadata if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"CQ v1 thiếu metadata nguồn: {missing}")
        validation = json.loads((source / "validation.json").read_text(encoding="utf-8"))
        failed_checks = sorted(key for key, value in validation.items() if value is False)
        if failed_checks:
            raise ValueError(f"CQ v1 validation nguồn có gate fail: {failed_checks}")

        feature_spec = json.loads(
            (source / "feature_columns.json").read_text(encoding="utf-8")
        )
        source_version = str(feature_spec.get("version", "phase-features-v2"))
        selector_columns = [
            column for column in feature_spec["distance_feature_columns"]
            if column != "phase_fraction"
        ]
        learner_columns = list(feature_spec.get("model_feature_columns", ()))
        if not learner_columns:
            learner_columns = [*feature_spec["distance_feature_columns"], *feature_spec["categorical_features"]]
        if len(selector_columns) != 56:
            raise ValueError(f"CQ v1 cần 56 feature selector sau lọc, nhận {len(selector_columns)}")
        if not set(selector_columns) <= set(learner_columns):
            raise ValueError("Mọi feature selector phải nằm trong learner view")
        missing_learner_ids = sorted(
            set(feature_spec["categorical_features"]) - set(learner_columns)
        )
        if missing_learner_ids:
            raise ValueError(f"CQ v1 learner view thiếu cột mã QĐ3: {missing_learner_ids}")

        raw_root = source / "supervised_raw"
        use_pool_preprocessing = all(
            (raw_root / phase / f"{split}.parquet").is_file()
            for phase in PHASES for split in ("train", "dev")
        )
        data_root = raw_root if use_pool_preprocessing else source / "model_ready"
        test_paths = [data_root / phase / "test.parquet" for phase in PHASES]
        if any(path.is_file() for path in test_paths) and not all(
            path.is_file() for path in test_paths
        ):
            raise FileNotFoundError("CQ v1 test phải đủ cả P1--P4 hoặc không đóng gói")
        test_data_packaged = all(path.is_file() for path in test_paths)
        available_splits = (
            SPLITS if test_data_packaged else ("train", "dev")
        )

        numeric_columns = list(
            feature_spec.get("raw_numeric_features", feature_spec["distance_feature_columns"])
        )
        categorical_columns = list(feature_spec["categorical_features"])
        pool_medians: pd.Series | None = None
        pool_means: pd.Series | None = None
        pool_scales: pd.Series | None = None
        if use_pool_preprocessing:
            numeric_parts = [
                pd.read_parquet(
                    raw_root / phase / "train.parquet", columns=numeric_columns,
                ).apply(pd.to_numeric, errors="coerce")
                for phase in PHASES
            ]
            pooled_numeric = pd.concat(numeric_parts, ignore_index=True)
            pool_medians = pooled_numeric.median(axis=0, skipna=True)
            if pool_medians.isna().any():
                bad = pool_medians.index[pool_medians.isna()].tolist()
                raise ValueError(f"CQ v1 Pool B có feature train toàn NULL: {bad}")
            pooled_imputed = pooled_numeric.fillna(pool_medians)
            pool_means = pooled_imputed.mean(axis=0)
            pool_scales = pooled_imputed.std(axis=0, ddof=0)
            pool_scales = pool_scales.mask(pool_scales <= 1e-12, 1.0)
            del numeric_parts, pooled_numeric, pooled_imputed

        shapes: dict[str, dict[str, list[int]]] = {
            split: {} for split in available_splits
        }
        class_counts: dict[str, dict[str, dict[str, int]]] = {
            split: {} for split in available_splits
        }
        split_groups: dict[str, set[int]] = {}
        train_parts: list[np.ndarray] = []
        selector_train_parts: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []
        train_row_ids: list[np.ndarray] = []
        train_phases: list[np.ndarray] = []

        for phase_index, phase in enumerate(PHASES, start=1):
            for split in available_splits:
                path = data_root / phase / f"{split}.parquet"
                if not path.is_file():
                    raise FileNotFoundError(f"CQ v1 thiếu {path}")
                columns = [
                    "user_id", "course_id", "phase", "split",
                    self.target_column, self.target_q_column,
                    *dict.fromkeys([*learner_columns, *selector_columns]),
                ]
                frame = pd.read_parquet(path, columns=columns)
                self._validate_frame(frame, split=split, phase=phase)
                if use_pool_preprocessing:
                    assert pool_medians is not None
                    assert pool_means is not None
                    assert pool_scales is not None
                    numeric = frame[numeric_columns].apply(
                        pd.to_numeric, errors="coerce"
                    )
                    numeric = (numeric.fillna(pool_medians) - pool_means) / pool_scales
                    transformed = numeric
                    for column in categorical_columns:
                        transformed[column] = pd.to_numeric(
                            frame[column], errors="coerce"
                        )
                    X = transformed[learner_columns].to_numpy(
                        dtype=np.float32, copy=True,
                    )
                    X_selector = transformed[selector_columns].to_numpy(
                        dtype=np.float32, copy=True,
                    )
                else:
                    X = frame[learner_columns].to_numpy(dtype=np.float32, copy=True)
                    X_selector = frame[selector_columns].to_numpy(
                        dtype=np.float32, copy=True,
                    )
                if not np.isfinite(X).all() or not np.isfinite(X_selector).all():
                    raise ValueError(f"CQ v1 có NaN/Inf trong learner/selector view: {phase}/{split}")
                y = self._encode_labels(frame[self.target_column], phase, split)
                row_ids = pd.util.hash_pandas_object(
                    frame[["user_id", "course_id", "phase"]], index=False,
                ).to_numpy(dtype=np.uint64)
                course_groups = pd.util.hash_pandas_object(
                    frame[["course_id"]], index=False,
                ).to_numpy(dtype=np.uint64)
                split_groups.setdefault(split, set()).update(
                    int(value) for value in np.unique(course_groups)
                )
                shapes[split][phase] = [int(X.shape[0]), int(X.shape[1])]
                class_counts[split][phase] = self._class_counts(y)

                if split == "train":
                    train_parts.append(X)
                    selector_train_parts.append(X_selector)
                    train_labels.append(y)
                    train_row_ids.append(row_ids)
                    train_phases.append(
                        np.full(len(y), phase_index - 1, dtype=np.int8)
                    )
                else:
                    prefix = f"{split}_phase{phase_index}"
                    np.save(output / f"X_{prefix}.npy", X)
                    np.save(output / f"y_{prefix}.npy", y)
                    np.save(output / f"row_ids_{prefix}.npy", row_ids)
                    np.save(output / f"course_groups_{prefix}.npy", course_groups)

        X_train = np.concatenate(train_parts, axis=0)
        X_selector_train = np.concatenate(selector_train_parts, axis=0)
        y_train = np.concatenate(train_labels)
        row_ids_train = np.concatenate(train_row_ids)
        phase_train = np.concatenate(train_phases)
        if len(np.unique(row_ids_train)) != len(row_ids_train):
            raise ValueError("CQ v1 Pool B có row_id trùng")
        np.save(output / "X_train.npy", X_train)
        np.save(output / "X_selector_train.npy", X_selector_train)
        np.save(output / "y_train.npy", y_train)
        np.save(output / "row_ids_train.npy", row_ids_train)
        np.save(output / "phase_train.npy", phase_train)

        # Alias tương thích runner: split canonical là P4; CQ development vẫn
        # đánh giá đủ dev_phase1..4. Test không được tạo nếu gói M0 đã khóa nó.
        for split in tuple(item for item in ("dev", "test") if item in available_splits):
            for stem in ("X", "y", "row_ids"):
                shutil.copy2(
                    output / f"{stem}_{split}_phase4.npy",
                    output / f"{stem}_{split}.npy",
                )

        expected = get_dataset_spec(self.dataset_id)
        if validate_expected_rows:
            phase_rows = {
                split: {shapes[split][phase][0] for phase in PHASES}
                for split in available_splits
            }
            if any(len(values) != 1 for values in phase_rows.values()):
                raise ValueError(f"CQ v1 số dòng không đồng nhất giữa phase: {phase_rows}")
            if test_data_packaged:
                unique_total = sum(next(iter(values)) for values in phase_rows.values())
                if unique_total != expected.expected_rows:
                    raise ValueError(
                        f"CQ v1 có {unique_total:,} enrollment, cần {expected.expected_rows:,}"
                    )
            summary_path = source / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if int(summary.get("labeled_pairs", expected.expected_rows)) != expected.expected_rows:
                    raise ValueError("CQ v1 summary không khớp expected_rows đã đăng ký")
                expected_by_split = {
                    str(item["split"]): int(item["n"])
                    for item in summary.get("labeled_splits", [])
                }
                mismatched = {
                    split: (next(iter(phase_rows[split])), expected_by_split.get(split))
                    for split in available_splits
                    if split in expected_by_split
                    and next(iter(phase_rows[split])) != expected_by_split[split]
                }
                if mismatched:
                    raise ValueError(f"CQ v1 rows không khớp summary: {mismatched}")

        overlaps = {
            f"{left}_{right}": len(split_groups[left] & split_groups[right])
            for position, left in enumerate(available_splits)
            for right in available_splits[position + 1:]
        }
        if any(overlaps.values()):
            raise ValueError(f"CQ v1 course_id overlap giữa split: {overlaps}")

        contract = {
            "schema_version": 2,
            "numeric_feature_columns": list(feature_spec["distance_feature_columns"]),
            "selector_columns": selector_columns,
            "learner_columns": learner_columns,
            "excluded_from_X": [
                "user_id", "course_id", "phase", "split", "enroll_time",
                "cutoff_time",
                *feature_spec["target_columns"],
            ],
            "target_column": self.target_column,
            "label_mapping": LABEL_MAPPING,
            "source_preprocessing_scope": feature_spec.get("preprocessing_scope"),
            "adapter_preprocessing_scope": (
                "joint Pool B train across P1-P4; applied unchanged to dev/test"
                if use_pool_preprocessing
                else "source train-only per-phase preprocessing"
            ),
            "source_data_version": source_version,
            "group_unit": "course_id_pending_root_confirmation",
            "required_confirmatory_group_unit": "course_root_id",
            "confirmatory_eligible": False,
            "deviation": (
                "V3 split is random course-disjoint, not temporal/SGKF; "
                "course_id equivalence to course_root_id awaits written approval."
            ),
        }
        contract["contract_hash"] = stable_hash(contract)
        atomic_json(output / "feature_contract.json", contract)
        atomic_json(output / "schema.json", contract)
        atomic_json(output / "split_manifest.json", {
            "dataset_id": self.dataset_id,
            "split_policy": expected.split_policy,
            "pool_b_rows": int(len(y_train)),
            "rows_by_split_phase": shapes,
            "class_counts": class_counts,
            "course_id_overlap": overlaps,
            "group_unit": "course_id_pending_root_confirmation",
            "test_locked": True,
            "test_data_packaged": test_data_packaged,
            "confirmatory_eligible": False,
        })
        preprocessor_payload: dict[str, Any] = {
            "source": (
                "phase_features_v3/course_random_split/supervised_raw"
                if use_pool_preprocessing
                else "phase_features_v2/model_ready"
            ),
            "learner_feature_columns": learner_columns,
            "selector_feature_columns": selector_columns,
            "target_column": self.target_column,
            "label_mapping": LABEL_MAPPING,
            "transform": (
                "median/mean/std fitted once on concatenated P1-P4 train"
                if use_pool_preprocessing
                else "source train-only per-phase preprocessing; no refit in adapter"
            ),
        }
        if use_pool_preprocessing:
            assert pool_medians is not None
            assert pool_means is not None
            assert pool_scales is not None
            preprocessor_payload.update({
                "numeric_feature_columns": numeric_columns,
                "medians": pool_medians.to_dict(),
                "means": pool_means.to_dict(),
                "scales": pool_scales.to_dict(),
            })
        joblib.dump(preprocessor_payload, output / "preprocessor.joblib")

        source_files = [source / name for name in required_metadata]
        source_files.extend(
            data_root / phase / f"{split}.parquet"
            for phase in PHASES for split in available_splits
        )
        files_before_manifest = sorted(path for path in output.iterdir() if path.is_file())
        atomic_json(manifest_path, {
            "status": "prepared_development_only",
            "dataset": asdict(expected),
            "adapter": f"{type(self).__module__}.{type(self).__qualname__}",
            "fit_scope": preprocessor_payload["transform"],
            "target_policy": "alpha=1 primary" if self.target_column == "y_alpha_1" else "alpha=0.1 sensitivity",
            "test_locked": True,
            "test_data_packaged": test_data_packaged,
            "confirmatory_eligible": False,
            "blockers": [
                "course_id equivalence to course_root_id awaits written approval",
                "V3 random course split is not the registered temporal/SGKF protocol",
                "FullTrain-A full-behavior table is not supplied",
            ],
            "raw_files_sha256": {
                path.relative_to(source).as_posix(): sha256_file(path)
                for path in source_files
            },
            "artifacts_sha256": {
                path.name: sha256_file(path) for path in files_before_manifest
            },
        })
        return manifest_path

    def _validate_frame(self, frame: pd.DataFrame, *, split: str, phase: str) -> None:
        if set(frame["split"].astype(str).unique()) != {split}:
            raise ValueError(f"CQ v1 split sai trong {phase}/{split}")
        if set(frame["phase"].astype(str).unique()) != {phase}:
            raise ValueError(f"CQ v1 phase sai trong {phase}/{split}")
        if frame[["user_id", "course_id", self.target_column]].isna().any().any():
            raise ValueError(f"CQ v1 thiếu key/target trong {phase}/{split}")
        if frame.duplicated(["user_id", "course_id", "phase"]).any():
            raise ValueError(f"CQ v1 trùng enrollment-phase trong {phase}/{split}")

    @staticmethod
    def _encode_labels(values: pd.Series, phase: str, split: str) -> np.ndarray:
        unknown = sorted(set(values.astype(str)) - set(LABEL_MAPPING))
        if unknown:
            raise ValueError(f"CQ v1 có label lạ {unknown} trong {phase}/{split}")
        return values.astype(str).map(LABEL_MAPPING).to_numpy(dtype=np.int64)

    @staticmethod
    def _class_counts(y: np.ndarray) -> dict[str, int]:
        labels, counts = np.unique(y, return_counts=True)
        return {str(int(label)): int(count) for label, count in zip(labels, counts)}


class CourseQualityV1Adapter(_CourseQualityV1Base):
    """Nhãn chính theo đóng góp AFELO alpha=1."""

    dataset_id = "course_quality_v1"
    target_column = "y_alpha_1"
    target_q_column = "q_alpha_1"


class CourseQualityV1Afelo01Adapter(_CourseQualityV1Base):
    """Sensitivity label theo đóng góp AFELO alpha=0.1."""

    dataset_id = "course_quality_v1_afelo0p1"
    target_column = "y_alpha_0p1"
    target_q_column = "q_alpha_0p1"
