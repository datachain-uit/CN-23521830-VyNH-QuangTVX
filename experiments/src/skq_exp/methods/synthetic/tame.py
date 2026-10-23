"""Adapter TAME gọi trực tiếp hàm tổng hợp dữ liệu từ source chính thức.

Trạng thái: official-source adapter có độ bám cao ở bước synthesis; file này nạp
``tame_synthesize`` từ repo TAME đã khóa commit, không chép hoặc viết lại vòng tối
ưu. Tuy nhiên, đây không phải tái lập nguyên vẹn toàn bộ paper vì preprocessing,
budget và downstream learner được đặt theo protocol chung của SKQ. Cơ chế TAME:
sinh các điểm train mới bằng distribution matching trong không gian embedding.
Loại đầu ra: dữ liệu tổng hợp theo số mẫu mỗi lớp, không phải subset dòng thật.

TAME được đưa vào bài làm baseline synthetic/dataset-condensation có source chính
thức để so sánh với phương pháp nén bằng chọn dòng của SKQ. Adapter chỉ chuyển
tensor sang ``GeneratedDatasetResult`` và ghi commit/provenance. Thiếu PyTorch,
CUDA bắt buộc hoặc vượt ``max_rows`` phải ghi BLOCKED/predicted_timeout; không có
fallback mang tên TAME.
"""

from __future__ import annotations

import importlib
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..native.common import verify_locked_repo
from .result import GeneratedDatasetResult


class TAMEOfficialGenerator:
    method_id = "s_tame_official"

    def __init__(self, repo_root: Path, seed: int, options: dict[str, Any] | None = None):
        self.repo_root = Path(repo_root)
        self.seed = int(seed)
        self.options = dict(options or {})

    def generate(self, X_train: np.ndarray, y_train: np.ndarray, budget_ratio: float, **kwargs):
        requested = max(1, int(round(len(y_train) * budget_ratio)))
        repo = self.repo_root / "tame"
        if not (repo / "synth" / "tame_synth.py").exists():
            return GeneratedDatasetResult.failure(
                self.method_id, "blocked", requested, f"Thiếu repo TAME: {repo}"
            )
        try:
            commit = verify_locked_repo(repo)
        except RuntimeError as error:
            return GeneratedDatasetResult.failure(self.method_id, "blocked", requested, str(error))
        max_rows = int(self.options.get("max_rows", 2000))
        if requested > max_rows:
            return GeneratedDatasetResult.failure(
                self.method_id, "predicted_timeout", requested,
                f"TAME budget {requested} vượt resource gate {max_rows}; tăng method_options.max_rows sau smoke GPU",
            )
        labels = np.unique(y_train)
        if not np.array_equal(labels, np.arange(len(labels))):
            return GeneratedDatasetResult.failure(
                self.method_id, "failed", requested,
                "TAME upstream yêu cầu nhãn liên tục bắt đầu từ 0",
            )
        ipc = requested // len(labels)
        if ipc < 1:
            return GeneratedDatasetResult.failure(
                self.method_id, "budget_infeasible", requested,
                "Budget tổng nhỏ hơn số lớp; không thể tạo ít nhất một điểm/lớp",
            )
        device = str(self.options.get("device", "auto"))
        try:
            import torch
        except ModuleNotFoundError as error:
            return GeneratedDatasetResult.failure(
                self.method_id, "blocked", requested, f"TAME cần PyTorch: {error}"
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            return GeneratedDatasetResult.failure(
                self.method_id, "blocked", requested, "TAME config yêu cầu CUDA nhưng runtime không có GPU"
            )
        started = time.perf_counter()
        try:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            tame_synthesize = _load_tame_synthesize(repo)
            config = {
                "device": device,
                "ipc": int(ipc),
                "dm_iters": int(self.options.get("dm_iters", 100)),
                "dm_lr": float(self.options.get("dm_lr", 0.5)),
                "dm_batch_real": int(self.options.get("dm_batch_real", 128)),
                "dm_embed_hidden": int(self.options.get("dm_embed_hidden", 128)),
                "dm_embed_dim": int(self.options.get("dm_embed_dim", 64)),
                "dm_embedder_type": str(self.options.get("dm_embedder_type", "ln_res_l")),
                "dm_embedder_size": str(self.options.get("dm_embedder_size", "small")),
                "grad_clip": float(self.options.get("grad_clip", 10.0)),
                "moment_eps": float(self.options.get("moment_eps", 1e-6)),
                "cov_weight": float(self.options.get("cov_weight", 1.0)),
                "dm_views": int(self.options.get("dm_views", 1)),
                "init_seed": self.seed,
            }
            data = {
                # Processed arrays có thể là read-only memmap. Tạo bản sao writable
                # để PyTorch không cảnh báo/ghi ngoài ý muốn; giá trị không thay đổi.
                "X_train": torch.as_tensor(
                    np.array(X_train, dtype=np.float32, copy=True), dtype=torch.float32
                ),
                "y_train": torch.as_tensor(
                    np.array(y_train, dtype=np.int64, copy=True), dtype=torch.long
                ),
                "input_dim": int(X_train.shape[1]),
                "num_classes": int(len(labels)),
            }
            X_syn, y_syn = tame_synthesize(data, config)
            X_syn = X_syn.detach().cpu().numpy().astype(np.float32, copy=False)
            y_syn = y_syn.detach().cpu().numpy().astype(np.int64, copy=False)
        except (ModuleNotFoundError, ImportError) as error:
            return GeneratedDatasetResult.failure(
                self.method_id, "blocked", requested, f"Dependency TAME chưa đủ: {error}"
            )
        except Exception as error:
            return GeneratedDatasetResult.failure(
                self.method_id, "failed", requested, f"TAME upstream trả lỗi, không dùng fallback: {error!r}"
            )
        if (
            X_syn.ndim != 2 or y_syn.ndim != 1 or len(X_syn) != len(y_syn)
            or X_syn.shape[1] != X_train.shape[1] or len(X_syn) == 0
            or not np.isfinite(X_syn).all()
            or set(np.unique(y_syn).tolist()) != set(labels.tolist())
        ):
            return GeneratedDatasetResult.failure(
                self.method_id, "failed", requested,
                "TAME output vi phạm shape/finite/label coverage contract",
            )
        elapsed = time.perf_counter() - started
        return GeneratedDatasetResult(
            X=X_syn,
            y=y_syn,
            weights=np.ones(len(y_syn), dtype=np.float64),
            requested_rows=requested,
            realized_rows=len(y_syn),
            generator_id=self.method_id,
            diagnostics={
                "upstream_repo": "https://github.com/eduard-b/TAME.git",
                "upstream_commit": commit,
                "fidelity": "official tame_synthesize function executed through an adapter",
                "execution_mode": "upstream_source_function",
                "budget_contract": "TAME IPC per class; realized = floor(requested/classes)*classes",
                "ipc": int(ipc),
                "device": device,
                "embedder": config["dm_embedder_type"],
                "storage_bytes_float32": int(X_syn.nbytes + y_syn.nbytes),
            },
            timings={"generate": elapsed, "total": elapsed},
        )


def _load_tame_synthesize(repo: Path):
    """Nạp module upstream cùng package ``models`` ở đúng vị trí mà TAME mong đợi."""
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    module = importlib.import_module("synth.tame_synth")
    return module.tame_synthesize
