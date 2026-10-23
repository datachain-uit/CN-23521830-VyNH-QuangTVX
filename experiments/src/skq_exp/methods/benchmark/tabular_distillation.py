"""Adapter Gonzalez và Leverage từ repo benchmark Tabular Data Distillation.

Trạng thái: official-benchmark-source adapter, không phải source gốc của paper
Gonzalez/Leverage và vì vậy không gắn nhãn native-paper. Adapter chạy trực tiếp
hai hàm upstream đã khóa commit; chỉ thêm seed, inject ``PCA`` bị thiếu và khôi
phục index cho Gonzalez. Cơ chế: Gonzalez là farthest-first/k-center; Leverage
chọn theo leverage score sau phép chiếu PCA. Loại đầu ra: tập con dòng thật đúng
exact budget, trọng số bằng nhau.

Hai baseline được đưa vào bài để đại diện cho phương pháp hình học cổ điển và
phương pháp dựa trên độ ảnh hưởng tuyến tính. Repo upstream còn import SDV không
liên quan, nên file này chỉ compile đúng hai ``FunctionDef`` cần thiết; không đổi
công thức chọn mẫu nhưng luôn ghi rõ compatibility boundary trong provenance.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from ..base import BaseSelector
from ..contracts import exact_budget_size
from ..result import SelectionResult
from ..native.common import verify_locked_repo


def _load_upstream_functions(repo: Path) -> dict[str, object]:
    """Chỉ compile hai hàm upstream, bỏ qua các import SDV không liên quan."""
    source_path = repo / "tabular_distillation_method.py"
    if not source_path.exists():
        raise FileNotFoundError(f"Thiếu source upstream: {source_path}")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"distill_coreset", "distill_coreset_leverage_scores"}
    }
    missing = {"distill_coreset", "distill_coreset_leverage_scores"} - set(wanted)
    if missing:
        raise RuntimeError(f"Source upstream thiếu hàm: {sorted(missing)}")
    namespace = {"np": np, "PCA": PCA}
    compiled = {}
    for name, node in wanted.items():
        module = ast.Module(body=[node], type_ignores=[])
        code = compile(ast.fix_missing_locations(module), str(source_path), "exec")
        exec(code, namespace, namespace)
        compiled[name] = namespace[name]
    return compiled


def _recover_indices(X: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Khôi phục vị trí dòng Gonzalez đã chọn mà không đổi nội dung output."""
    X = np.asarray(X)
    selected = np.asarray(selected)
    buckets: dict[bytes, list[int]] = {}
    for index, row in enumerate(X):
        buckets.setdefault(np.ascontiguousarray(row).tobytes(), []).append(index)
    recovered: list[int] = []
    for row in selected:
        key = np.ascontiguousarray(row).tobytes()
        candidates = buckets.get(key, [])
        if not candidates:
            raise RuntimeError("Không khôi phục được index từ output Gonzalez upstream")
        recovered.append(candidates.pop(0))
    if len(set(recovered)) != len(recovered):
        raise RuntimeError("Output Gonzalez upstream chứa dòng trùng không thể ánh xạ duy nhất")
    return np.asarray(recovered, dtype=np.int64)


class _TabularDistillationSelector(BaseSelector):
    """Khởi tạo chung cho hai hàm lấy từ cùng một repo benchmark."""

    def __init__(self, repo_root: Path, seed: int):
        super().__init__(seed)
        self.repo = Path(repo_root) / "tabular_data_distillation"


class GonzalezBenchmarkSelector(_TabularDistillationSelector):
    method_id = "n_gcoreset_benchmark"

    def select(self, X_train, y_train, budget_ratio, **kwargs):
        requested = exact_budget_size(len(y_train), budget_ratio)
        started = time.perf_counter()
        X = np.asarray(X_train)
        try:
            upstream_commit = verify_locked_repo(self.repo)
        except RuntimeError as error:
            return SelectionResult.failure(self.method_id, "blocked", requested, str(error))
        functions = _load_upstream_functions(self.repo)
        np.random.seed(self.seed)
        selected_X, _selected_y = functions["distill_coreset"](
            X, np.asarray(y_train), requested * 100 / len(X),
        )
        indices = _recover_indices(X, selected_X)
        elapsed = time.perf_counter() - started
        return SelectionResult(
            indices=np.asarray(indices), weights=np.ones(requested), requested_rows=requested,
            realized_rows=requested, budget_mode="exact_total", method_id=self.method_id,
            diagnostics={
                "upstream_repo": "https://github.com/corneliuflorea/Tabular-Data-Distillation.git",
                "upstream_commit": upstream_commit,
                "fidelity": "official upstream function executed through a narrow compatibility adapter",
                "execution_mode": "upstream_source_function",
                "compatibility_patch": "seed injection and output-row-to-index recovery",
                "upstream_function": "distill_coreset",
            }, timings={"select": elapsed, "total": elapsed},
        )


class LeverageBenchmarkSelector(_TabularDistillationSelector):
    method_id = "n_leverage_benchmark"

    def select(self, X_train, y_train, budget_ratio, **kwargs):
        requested = exact_budget_size(len(y_train), budget_ratio)
        started = time.perf_counter()
        X = np.asarray(X_train)
        try:
            upstream_commit = verify_locked_repo(self.repo)
        except RuntimeError as error:
            return SelectionResult.failure(self.method_id, "blocked", requested, str(error))
        functions = _load_upstream_functions(self.repo)
        np.random.seed(self.seed)
        _selected_X, _selected_y, indices = functions["distill_coreset_leverage_scores"](
            X, np.asarray(y_train), requested * 100 / len(X),
        )
        indices = np.asarray(indices, dtype=np.int64)
        elapsed = time.perf_counter() - started
        return SelectionResult(
            indices=indices, weights=np.ones(requested), requested_rows=requested,
            realized_rows=requested, budget_mode="exact_total", method_id=self.method_id,
            diagnostics={
                "upstream_repo": "https://github.com/corneliuflorea/Tabular-Data-Distillation.git",
                "upstream_commit": upstream_commit,
                "fidelity": "official upstream function executed through a narrow compatibility adapter",
                "execution_mode": "upstream_source_function",
                "compatibility_patch": "injected missing sklearn PCA name and explicit seed",
                "pca_components": int(min(20, X.shape[1])),
                "upstream_function": "distill_coreset_leverage_scores",
            }, timings={"select": elapsed, "total": elapsed},
        )
