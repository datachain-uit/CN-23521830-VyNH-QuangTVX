"""Khung dùng lại cho dataset CSV đã có train/dev/test từ bên ngoài.

Subclass chỉ khai báo layout file và cách chuẩn hóa một DataFrame. Khung này lo
kiểm tra file, row count, lưu NPY, row ID, temporal test alias, audit group,
preprocessor, schema và hash provenance. Nhờ vậy adapter mới không sao chép luồng
I/O nhạy cảm với leakage hoặc quên một manifest bắt buộc.
"""

from __future__ import annotations

import shutil
from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ...artifacts import atomic_json, sha256_file
from ..registry import get_dataset_spec
from .base import DatasetAdapter


@dataclass(frozen=True)
class PresplitCSVLayout:
    train_file: str
    dev_file: str
    test_files: tuple[str, ...]
    canonical_test_index: int = -1

    @property
    def all_files(self) -> tuple[str, ...]:
        return (self.train_file, self.dev_file, *self.test_files)


class PresplitCSVAdapter(DatasetAdapter):
    """Template method cho snapshot CSV có split được cung cấp sẵn."""

    layout: PresplitCSVLayout
    expected_rows_by_file: dict[str, int]
    expected_processed_features: int | None = None
    group_columns: tuple[str, ...] = ()
    target_column: str
    preprocessing_contract: str
    evidence_role: str = "development"
    fit_scope: str = "preprocessor fitted on train only"

    def prepare(
        self,
        input_dir: str | Path,
        processed_root: str | Path,
        *,
        overwrite: bool = False,
        validate_expected_rows: bool = True,
    ) -> Path:
        spec = get_dataset_spec(self.dataset_id)
        source = Path(input_dir).resolve()
        output = Path(processed_root).resolve() / self.dataset_id
        manifest_path = output / "preprocessing_manifest.json"
        if manifest_path.exists() and not overwrite:
            return manifest_path

        paths = {name: source / name for name in self.layout.all_files}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Thiếu file của {self.dataset_id}: {missing}")
        output.mkdir(parents=True, exist_ok=True)

        state = self.new_state()
        train_raw = self.read_frame(paths[self.layout.train_file])
        self.validate_row_count(
            self.layout.train_file, len(train_raw), validate_expected_rows,
        )
        train_groups = self.group_keys(train_raw)
        raw_train_columns = train_raw.columns.tolist()
        X_train, y_train, feature_names = self.transform_frame(
            train_raw, state, fit=True, feature_names=None,
        )
        del train_raw
        expected_features = self.expected_processed_features or spec.expected_features
        if validate_expected_rows and X_train.shape[1] != expected_features:
            raise ValueError(
                f"{self.dataset_id} có {X_train.shape[1]} feature sau xử lý, "
                f"cần đúng {expected_features}"
            )
        self.save_split(output, "train", X_train, y_train, row_offset=0)
        train_shape = [int(value) for value in X_train.shape]
        del X_train, y_train

        dev_raw = self.read_frame(paths[self.layout.dev_file])
        self.validate_row_count(self.layout.dev_file, len(dev_raw), validate_expected_rows)
        dev_groups = self.group_keys(dev_raw)
        X_dev, y_dev, _ = self.transform_frame(
            dev_raw, state, fit=False, feature_names=feature_names,
        )
        del dev_raw
        self.save_split(output, "dev", X_dev, y_dev, row_offset=train_shape[0])
        dev_shape = [int(value) for value in X_dev.shape]
        del X_dev, y_dev

        test_shapes: dict[str, list[int]] = {}
        test_groups: dict[int, set[int]] = {}
        test_labels: dict[int, dict[int, str]] = {}
        canonical = self._canonical_test_position()
        for position, filename in enumerate(self.layout.test_files, start=1):
            raw = self.read_frame(paths[filename])
            self.validate_row_count(filename, len(raw), validate_expected_rows)
            test_groups[position] = self.group_keys(raw)
            test_labels[position] = self.group_labels(raw)
            X_test, y_test, _ = self.transform_frame(
                raw, state, fit=False, feature_names=feature_names,
            )
            del raw
            np.save(output / f"X_test_phase{position}.npy", X_test)
            np.save(output / f"y_test_phase{position}.npy", y_test)
            test_shapes[str(position)] = [int(value) for value in X_test.shape]
            if position == canonical:
                np.save(
                    output / "row_ids_test.npy",
                    np.arange(len(y_test), dtype=np.int64)
                    + train_shape[0] + dev_shape[0],
                )
            del X_test, y_test

        self._write_canonical_test_alias(output, canonical)
        joblib.dump(
            self.preprocessor_payload(state, feature_names),
            output / "preprocessor.joblib",
        )

        split_manifest = self.build_split_manifest(
            train_shape=train_shape,
            dev_shape=dev_shape,
            test_shapes=test_shapes,
            train_groups=train_groups,
            dev_groups=dev_groups,
            test_groups=test_groups,
            test_labels=test_labels,
            canonical_test_position=canonical,
        )
        atomic_json(output / "split_manifest.json", split_manifest)
        atomic_json(
            output / "schema.json",
            self.schema_payload(raw_train_columns, feature_names, state),
        )
        files_before_manifest = sorted(path for path in output.iterdir() if path.is_file())
        atomic_json(
            manifest_path,
            {
                "status": "prepared",
                "dataset": asdict(spec),
                "adapter": {
                    "class": f"{type(self).__module__}.{type(self).__qualname__}",
                    "preprocessing_contract": self.preprocessing_contract,
                },
                "fit_scope": self.fit_scope,
                "evidence_role": self.evidence_role,
                "raw_files_sha256": {
                    name: sha256_file(path) for name, path in sorted(paths.items())
                },
                "artifacts_sha256": {
                    path.name: sha256_file(path) for path in files_before_manifest
                },
            },
        )
        return manifest_path

    def read_frame(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def new_state(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def transform_frame(
        self,
        frame: pd.DataFrame,
        state: dict[str, Any],
        *,
        fit: bool,
        feature_names: list[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Fit/transform train hoặc transform dev/test mà không nhìn ngược split."""

    @abstractmethod
    def preprocessor_payload(
        self,
        state: dict[str, Any],
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Trả state đủ để tái hiện preprocessing."""

    @abstractmethod
    def schema_payload(
        self,
        raw_train_columns: list[str],
        feature_names: list[str],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Trả schema có ý nghĩa riêng của dataset."""

    def validate_row_count(self, filename: str, actual: int, enabled: bool) -> None:
        expected = self.expected_rows_by_file.get(filename)
        if enabled and expected is not None and actual != expected:
            raise ValueError(f"{filename} có {actual:,} dòng, cần đúng {expected:,}")

    def group_keys(self, frame: pd.DataFrame) -> set[int]:
        if not self.group_columns:
            return set()
        missing = [column for column in self.group_columns if column not in frame.columns]
        if missing:
            raise KeyError(f"{self.dataset_id} thiếu group columns: {missing}")
        values = pd.util.hash_pandas_object(
            frame[list(self.group_columns)], index=False,
        ).to_numpy(dtype=np.uint64)
        return {int(value) for value in np.unique(values)}

    def group_labels(self, frame: pd.DataFrame) -> dict[int, str]:
        if not self.group_columns:
            return {}
        keys = pd.util.hash_pandas_object(
            frame[list(self.group_columns)], index=False,
        ).to_numpy(dtype=np.uint64)
        labels = frame[self.target_column].astype(str).to_numpy()
        result: dict[int, str] = {}
        for key, label in zip(keys, labels):
            result.setdefault(int(key), str(label))
        return result

    def build_split_manifest(
        self,
        *,
        train_shape: list[int],
        dev_shape: list[int],
        test_shapes: dict[str, list[int]],
        train_groups: set[int],
        dev_groups: set[int],
        test_groups: dict[int, set[int]],
        test_labels: dict[int, dict[int, str]],
        canonical_test_position: int,
    ) -> dict[str, Any]:
        first_test = test_groups[1]
        overlap = {
            "train_dev": len(train_groups & dev_groups),
            "train_test": len(train_groups & first_test),
            "dev_test": len(dev_groups & first_test),
        } if self.group_columns else {}
        aligned = all(
            test_groups[position] == first_test
            and test_labels[position] == test_labels[1]
            for position in range(2, len(self.layout.test_files) + 1)
        )
        gate = bool(self.group_columns) and all(
            value == 0 for value in overlap.values()
        ) and aligned
        rows = {"train": train_shape[0], "dev": dev_shape[0]}
        rows.update({f"test_phase_{key}": shape[0] for key, shape in test_shapes.items()})
        return {
            "dataset_id": self.dataset_id,
            "split_seed": None,
            "split_policy": get_dataset_spec(self.dataset_id).split_policy,
            "rows": rows,
            "canonical_test_alias": f"test_phase_{canonical_test_position}",
            "group_definition": list(self.group_columns),
            "group_overlap": overlap,
            "test_snapshots_aligned": aligned,
            "paper_evidence_gate_passed": gate,
            "warning": self.split_warning(overlap, aligned),
        }

    def split_warning(self, overlap: dict[str, int], aligned: bool) -> str | None:
        if not self.group_columns:
            return (
                "Adapter chưa khai báo group_columns; chưa đủ audit để mở "
                "paper evidence gate."
            )
        if overlap and any(value > 0 for value in overlap.values()):
            return (
                "Run chỉ dùng để integration/debug khi group overlap khác 0; "
                "không dùng metric này làm bằng chứng confirmatory."
            )
        if not aligned:
            return "Các test snapshot không cùng group/nhãn; cần audit trước khi đánh giá."
        return None

    @staticmethod
    def save_split(
        output: Path,
        split: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        row_offset: int,
    ) -> None:
        np.save(output / f"X_{split}.npy", X)
        np.save(output / f"y_{split}.npy", y)
        np.save(
            output / f"row_ids_{split}.npy",
            np.arange(len(y), dtype=np.int64) + int(row_offset),
        )

    def _canonical_test_position(self) -> int:
        count = len(self.layout.test_files)
        position = self.layout.canonical_test_index
        if position < 0:
            position = count + position
        if not 0 <= position < count:
            raise ValueError("canonical_test_index nằm ngoài test_files")
        return position + 1

    @staticmethod
    def _write_canonical_test_alias(output: Path, position: int) -> None:
        # Copy thay vì symlink để archive Kaggle tự chứa đủ dữ liệu.
        shutil.copyfile(output / f"X_test_phase{position}.npy", output / "X_test.npy")
        shutil.copyfile(output / f"y_test_phase{position}.npy", output / "y_test.npy")


class TrainOnlyOrdinalCSVAdapter(PresplitCSVAdapter):
    """Preprocessor số + categorical ordinal fit train, dùng lại cho nhiều CSV."""

    label_mapping: dict[Any, int]
    drop_columns: frozenset[str] = frozenset()

    def new_state(self) -> dict[str, Any]:
        return {"category_maps": {}, "numeric_fill_values": {}}

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Hook dataset-specific chạy trước khi bỏ cột và encode."""
        return frame

    def transform_frame(
        self,
        frame: pd.DataFrame,
        state: dict[str, Any],
        *,
        fit: bool,
        feature_names: list[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        frame = self.normalize_frame(frame)
        frame = frame.drop(columns=[
            column for column in self.drop_columns if column in frame.columns
        ])
        if self.target_column not in frame.columns:
            raise KeyError(f"Thiếu cột nhãn {self.target_column}")
        raw_labels = frame.pop(self.target_column)
        labels = raw_labels.map(self.label_mapping)
        if labels.isna().any():
            unknown = sorted(set(raw_labels.loc[labels.isna()].astype(str)))
            raise ValueError(f"{self.target_column} ngoài mapping: {unknown}")

        for column in frame.select_dtypes(include=["object"]).columns:
            values = frame[column].astype(str)
            if fit:
                categories = sorted(values.unique().tolist())
                state["category_maps"][column] = {
                    value: index for index, value in enumerate(categories)
                }
            mapping = state["category_maps"].get(column)
            if mapping is None:
                raise ValueError(
                    f"Cột categorical '{column}' không tồn tại trong train schema"
                )
            frame[column] = values.map(mapping).fillna(-1).astype(np.int64)

        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if fit:
            state["numeric_fill_values"] = {
                column: self._median_or_zero(frame[column]) for column in frame.columns
            }
            feature_names = frame.columns.tolist()
        if feature_names is None:
            raise ValueError("Thiếu feature_names đã fit từ train")
        missing = [column for column in feature_names if column not in frame.columns]
        if missing:
            raise ValueError(f"Split thiếu feature đã có trong train: {missing}")
        frame = frame[feature_names]
        for column in feature_names:
            frame[column] = frame[column].fillna(state["numeric_fill_values"][column])
        return (
            frame.to_numpy(dtype=np.float32, copy=True),
            labels.to_numpy(dtype=np.int64, copy=True),
            feature_names,
        )

    def preprocessor_payload(
        self,
        state: dict[str, Any],
        feature_names: list[str],
    ) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "label_mapping": self.label_mapping,
            "feature_names": feature_names,
            **state,
        }

    def schema_payload(
        self,
        raw_train_columns: list[str],
        feature_names: list[str],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        del state
        return {
            "dataset_id": self.dataset_id,
            "raw_train_columns": raw_train_columns,
            "processed_columns": feature_names,
            "label_values": [str(value) for value in self.label_mapping],
            "label_mapping": self.label_mapping,
            "dropped_columns": sorted(self.drop_columns),
            "preprocessing_contract": self.preprocessing_contract,
        }

    @staticmethod
    def _median_or_zero(series: pd.Series) -> float:
        median = series.median()
        return 0.0 if pd.isna(median) else float(median)
