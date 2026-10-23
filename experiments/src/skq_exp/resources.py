"""File này ước lượng chi phí trước khi chạy và theo dõi tài nguyên trong selector.

Bộ nhớ dự kiến vẫn có thể chặn run bằng `predicted_oom`. Số phép tính chỉ là chỉ
báo chẩn đoán, không được dùng làm gate loại run theo QĐ16/O8/O15 của Review
12–13. Với code SKQ nội bộ, guard kiểm tra thời gian và RSS thật giữa các
batch/vòng lặp để dừng an toàn bằng trạng thái `timeout` hoặc `oom` riêng biệt.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import psutil


class ResourceLimitError(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class CostEstimate:
    rff_bytes: int
    working_bytes: int
    estimated_operations: float


def estimate_selection_cost(
    method_id: str,
    n_rows: int,
    n_features: int,
    budget_rows: int,
    *,
    rff_components: int = 256,
    n_classes: int = 2,
    structured_work_rows: float | None = None,
    streaming_rff_rows: int | None = None,
) -> CostEstimate:
    rff_storage_rows = n_rows if streaming_rff_rows is None else min(
        n_rows, max(1, int(streaming_rff_rows))
    )
    rff_bytes = int(rff_storage_rows * rff_components * 4)
    base_bytes = int(n_rows * n_features * 4)
    operations = float(n_rows * n_features)
    if method_id == "d05_equal_group_weight":
        # D05 chỉ dùng lại index của source method rồi tính lại trọng số theo group.
        rff_bytes = 0
    elif method_id.startswith(("p", "d02", "d04", "skq_pc", "x_coretab_mc_cell")):
        operations += float(n_rows * n_features * rff_components)
        # Khi chưa có structure, giữ estimate global bảo thủ. Runner có thể hoãn
        # hạng herding bằng structured_work_rows=0; selector sẽ kiểm lại sau khi
        # biết chính xác quota và candidate count của từng group.
        if structured_work_rows is None:
            structured_work_rows = budget_rows * n_rows / max(1, n_classes)
        operations += float(structured_work_rows * rff_components)
    if "craig" in method_id:
        pairwise_bytes = int(n_rows * n_rows * 8)
        gradient_bytes = int(n_rows * n_features * max(2, n_classes) * 8)
        operations += float(budget_rows * n_rows * max(2, n_classes))
        base_bytes += gradient_bytes + pairwise_bytes
    if "gcoreset" in method_id:
        operations += float(budget_rows * n_rows * n_features)
    if method_id.startswith("s_kip"):
        # KIP-TDBench tạo kernel target-support với target batch lớn gấp 10 lần
        # synthetic budget. Nhân thêm 6 cho kernel, gradient và trạng thái tối ưu;
        # đây là ước lượng RAM có đơn vị, không phải trần budget tùy ý.
        operations += float(100 * max(1, budget_rows) ** 2 * max(1, n_features))
        base_bytes += int(10 * max(1, budget_rows) ** 2 * 8 * 6)
    if method_id.startswith(("s_mtt", "s_datm")):
        operations += float(100 * n_rows * n_features * max(2, n_classes))
    if method_id.startswith("s_gm"):
        # GM tính gradient trên real và synthetic theo lớp qua nhiều epoch.
        # Hệ số 128 đại diện hidden width của profile pilot mặc định.
        operations += float(100 * n_rows * n_features * 128)
        base_bytes += int((n_rows + budget_rows) * 128 * max(2, n_classes) * 8)
    if method_id in {
        "p04_skq_lrq_sq", "p05_skq_lrq_mq",
        "p08_skq_gonzalez_lrq_sq", "p09_skq_gonzalez_lrq_mq",
        "skq_pc_lrq_sq",
    }:
        operations += float(5 * n_rows * n_features * (3 if method_id.endswith("mq") else 1))
    working_bytes = int((base_bytes + rff_bytes) * 1.8)
    return CostEstimate(rff_bytes, working_bytes, operations)


def enforce_preflight(estimate: CostEstimate, policy: dict[str, Any] | None) -> None:
    policy = policy or {}
    max_ram_gb = float(policy.get("max_ram_gb", 8.0))
    available = psutil.virtual_memory().available
    configured = int(max_ram_gb * 1024 ** 3)
    allowed = min(configured, int(available * 0.75))
    if estimate.working_bytes > allowed:
        raise ResourceLimitError(
            "predicted_oom",
            f"Ước lượng {estimate.working_bytes / 1024**3:.2f} GiB vượt mức an toàn "
            f"{allowed / 1024**3:.2f} GiB",
        )


def operation_diagnostics(
    estimate: CostEstimate,
    policy: dict[str, Any] | None,
) -> dict[str, float | bool | str]:
    """Ghi estimate phép tính để audit; kết quả này tuyệt đối không phải gate."""
    policy = policy or {}
    reference = float(policy.get("max_estimated_operations", 50_000_000_000))
    ratio = estimate.estimated_operations / reference if reference > 0 else float("inf")
    return {
        "estimated_operations": float(estimate.estimated_operations),
        "legacy_reference_operations": reference,
        "ratio_to_legacy_reference": float(ratio),
        "exceeds_legacy_reference": bool(estimate.estimated_operations > reference),
        "decision_role": "diagnostic_only",
    }


class ResourceGuard:
    def __init__(self, policy: dict[str, Any] | None):
        policy = policy or {}
        self.started = time.monotonic()
        self.timeout_seconds = float(policy.get("timeout_seconds", 3600))
        self.max_rss_bytes = int(float(policy.get("max_ram_gb", 8.0)) * 1024 ** 3)
        self.process = psutil.Process(os.getpid())

    def check(self, stage: str = "run") -> None:
        elapsed = time.monotonic() - self.started
        if elapsed > self.timeout_seconds:
            raise ResourceLimitError(
                "timeout", f"Dừng ở {stage}: vượt {self.timeout_seconds:.0f} giây"
            )
        rss = self.process.memory_info().rss
        if rss > self.max_rss_bytes:
            raise ResourceLimitError(
                "oom", f"Dừng ở {stage}: RSS {rss / 1024**3:.2f} GiB vượt giới hạn"
            )
