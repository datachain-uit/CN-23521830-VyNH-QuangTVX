"""File này tạo đường dẫn và lưu artifact theo cùng một quy tắc.

Mọi kết quả đều đi qua các hàm ở đây để tránh mỗi script tự đặt tên khác nhau.
JSON được ghi an toàn qua file tạm; mảng lớn được lưu riêng để manifest luôn nhỏ
và dễ đọc.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ArtifactStorageError(RuntimeError):
    """Báo dừng an toàn khi artifact hoặc dung lượng đĩa vượt ngưỡng."""


def stable_hash(payload: Any) -> str:
    if is_dataclass(payload):
        payload = asdict(payload)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=output.name, suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class ArtifactLayout:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def run_dir(
        self, experiment_id: str, protocol_id: str, stage_id: str,
        dataset_id: str, method_id: str, budget_ratio: float,
        selector_seed: int, learner_id: str, model_seed: int,
    ) -> Path:
        budget = f"b{budget_ratio:.4f}".replace(".", "p")
        return (
            self.root / "experiments" / experiment_id / protocol_id / stage_id /
            dataset_id / method_id / budget / f"ss{selector_seed}" /
            learner_id / f"ms{model_seed}"
        )

    def method_artifact_dir(
        self, experiment_id: str, protocol_id: str, stage_id: str,
        dataset_id: str, method_id: str, budget_ratio: float, selector_seed: int,
    ) -> Path:
        """Một selector/generator chỉ có một payload, không lặp theo learner."""
        budget = f"b{budget_ratio:.4f}".replace(".", "p")
        return (
            self.root / "method_artifacts" / experiment_id / protocol_id / stage_id /
            dataset_id / method_id / budget / f"ss{selector_seed}"
        )

    def save_method_artifact(
        self,
        artifact_dir: Path,
        result: Any,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        """Lưu selection hoặc synthetic payload đúng một lần và trả reference nhỏ."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_kind = "synthetic" if hasattr(result, "X") else "selection"
        payload_paths: list[Path]
        if output_kind == "synthetic":
            _atomic_save_npz(
                artifact_dir / "generated_train.npz",
                X=np.asarray(result.X, dtype=np.float32),
                y=np.asarray(result.y),
                sample_weights=np.asarray(result.weights, dtype=np.float64),
            )
            payload_paths = [artifact_dir / "generated_train.npz"]
        else:
            _atomic_save_npy(
                artifact_dir / "selected_indices.npy",
                np.asarray(result.indices, dtype=np.int64),
            )
            _atomic_save_npy(
                artifact_dir / "sample_weights.npy",
                np.asarray(result.weights, dtype=np.float64),
            )
            payload_paths = [
                artifact_dir / "selected_indices.npy",
                artifact_dir / "sample_weights.npy",
            ]
        atomic_json(artifact_dir / "diagnostics.json", result.diagnostics)
        atomic_json(artifact_dir / "timings.json", result.timings)
        payload_paths.extend([artifact_dir / "diagnostics.json", artifact_dir / "timings.json"])
        payload_sha256 = {path.name: sha256_file(path) for path in payload_paths}
        artifact_bytes = int(sum(path.stat().st_size for path in payload_paths))
        artifact_manifest = {
            "schema_version": 3,
            "output_kind": output_kind,
            "identity": identity,
            "identity_hash": stable_hash(identity),
            "result": result.to_dict(),
            "payload_sha256": payload_sha256,
            "artifact_bytes": artifact_bytes,
        }
        atomic_json(artifact_dir / "artifact_manifest.json", artifact_manifest)
        return {
            "relative_path": artifact_dir.relative_to(self.root).as_posix(),
            "output_kind": output_kind,
            "identity_hash": artifact_manifest["identity_hash"],
            "artifact_bytes": artifact_bytes,
            "payload_sha256": payload_sha256,
        }

    def load_method_artifact(
        self,
        artifact_dir: Path,
        *,
        expected_identity: dict[str, Any],
    ) -> Any | None:
        """Nạp payload đã khóa; hash lệch thì không tái sử dụng âm thầm."""
        manifest_path = artifact_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity_hash") != stable_hash(expected_identity):
            return None
        for filename, expected_hash in manifest.get("payload_sha256", {}).items():
            path = artifact_dir / filename
            if not path.exists() or sha256_file(path) != expected_hash:
                raise ArtifactStorageError(f"Artifact hỏng hoặc đổi nội dung: {path}")
        raw = manifest["result"]
        if manifest.get("output_kind") == "synthetic":
            from .methods.synthetic.result import GeneratedDatasetResult

            with np.load(artifact_dir / "generated_train.npz") as saved:
                return GeneratedDatasetResult(
                    X=saved["X"], y=saved["y"], weights=saved["sample_weights"],
                    requested_rows=int(raw["requested_rows"]),
                    realized_rows=int(raw["realized_rows"]),
                    generator_id=str(raw["method_id"]),
                    output_type=str(raw.get("output_type", "synthetic_raw")),
                    diagnostics={**raw.get("diagnostics", {}), "reused_from": str(artifact_dir)},
                    timings=raw.get("timings", {}), status=str(raw["status"]),
                )
        from .methods.result import SelectionResult

        return SelectionResult(
            indices=np.load(artifact_dir / "selected_indices.npy"),
            weights=np.load(artifact_dir / "sample_weights.npy"),
            requested_rows=int(raw["requested_rows"]),
            realized_rows=int(raw["realized_rows"]),
            budget_mode=str(raw["budget_mode"]), method_id=str(raw["method_id"]),
            diagnostics={**raw.get("diagnostics", {}), "reused_from": str(artifact_dir)},
            timings=raw.get("timings", {}), status=str(raw["status"]),
        )

    def save_run_reference(
        self,
        run_dir: Path,
        result: Any,
        manifest: dict[str, Any],
        artifact_ref: dict[str, Any],
    ) -> None:
        """Learner run chỉ lưu manifest/ref; payload lớn ở method_artifacts."""
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "method_artifact_ref.json", artifact_ref)
        atomic_json(
            run_dir / "manifest.json",
            {**manifest, **result.to_dict(), "method_artifact_ref": artifact_ref},
        )

    def save_full_reference(self, run_dir: Path, result: Any, manifest: dict[str, Any]) -> None:
        """C00 chỉ là evaluator reference nên không tạo selected_indices giả."""
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "selected_indices.npy").unlink(missing_ok=True)
        (run_dir / "sample_weights.npy").unlink(missing_ok=True)
        (run_dir / "generated_train.npz").unlink(missing_ok=True)
        atomic_json(run_dir / "manifest.json", {**manifest, **result.to_dict(), "output_kind": "full_reference"})
        atomic_json(run_dir / "diagnostics.json", {"uses_all_train_rows": True})
        atomic_json(run_dir / "timings.json", result.timings)

    def experiment_size_bytes(self, experiment_id: str) -> int:
        """Tính dung lượng riêng của một experiment, bỏ archive đã đóng gói."""
        roots = [
            self.root / "experiments" / experiment_id,
            self.root / "method_artifacts" / experiment_id,
            self.root / "structures" / experiment_id,
            self.root / experiment_id,
        ]
        return int(sum(
            path.stat().st_size
            for base in roots if base.exists()
            for path in base.rglob("*") if path.is_file()
        ))

    def enforce_storage_budget(
        self,
        experiment_id: str,
        *,
        max_artifact_gb: float,
        min_free_disk_gb: float,
        reserve_bytes: int = 0,
    ) -> None:
        """Dừng trước khi ghi nếu vượt quota hoặc ổ đĩa sắp đầy."""
        self.root.mkdir(parents=True, exist_ok=True)
        used = self.experiment_size_bytes(experiment_id)
        quota = int(max_artifact_gb * 1024 ** 3)
        if used + reserve_bytes > quota:
            raise ArtifactStorageError(
                f"Artifact experiment sẽ vượt {max_artifact_gb:.2f} GiB "
                f"(đang dùng {used / 1024**3:.3f} GiB)"
            )
        free = shutil.disk_usage(self.root).free
        minimum = int(min_free_disk_gb * 1024 ** 3)
        if free - reserve_bytes < minimum:
            raise ArtifactStorageError(
                f"Ổ đĩa chỉ còn {free / 1024**3:.2f} GiB; cần chừa "
                f"ít nhất {min_free_disk_gb:.2f} GiB"
            )


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temp_name, "wb") as stream:
            np.save(stream, values)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temp_name, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Không thể ghi JSON cho kiểu {type(value).__name__}")

