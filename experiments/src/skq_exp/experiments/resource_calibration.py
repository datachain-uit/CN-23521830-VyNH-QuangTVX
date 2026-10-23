"""Fit calibration thời gian/RSS từ phép đo thật, không dùng operation gate.

Protocol mặc định yêu cầu 50k/200k/800k dòng, ba replicate mỗi mức. Mô hình
log-log trả cả point estimate và upper prediction bound. Kết quả chỉ quyết định
chạy trực tiếp, yêu cầu shard hoặc đo thêm; TIMEOUT chỉ được ghi từ phép chạy thật.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


CALIBRATION_ROWS = (50_000, 200_000, 800_000)
REPLICATES_PER_SIZE = 3
SELECTOR_CAP_SECONDS = 3 * 60 * 60


@dataclass(frozen=True)
class DeviceMetadata:
    cpu: str
    gpu: str | None
    ram_bytes: int
    vram_bytes: int | None
    max_threads: int
    environment_id: str
    provider: str = "unknown"
    gpu_count: int = 0
    accelerator_policy: str = "unspecified"


@dataclass(frozen=True)
class CalibrationObservation:
    selector_id: str
    rows: int
    budget_rows: int
    replicate: int
    warmup_seconds: float
    steady_seconds: float
    peak_rss_bytes: int
    peak_vram_bytes: int | None
    device: DeviceMetadata
    status: str = "success"

    def validate(self) -> None:
        if self.rows <= 0 or self.budget_rows <= 0:
            raise ValueError("rows và budget_rows phải dương")
        if self.replicate < 1:
            raise ValueError("replicate đánh số từ 1")
        if self.warmup_seconds < 0 or self.steady_seconds <= 0:
            raise ValueError("thời gian calibration không hợp lệ")
        if self.peak_rss_bytes <= 0:
            raise ValueError("peak_rss_bytes phải dương")
        if self.device.gpu_count < 0:
            raise ValueError("gpu_count không được âm")
        if self.device.gpu_count > 0 and not self.device.gpu:
            raise ValueError("Có GPU được cấp nhưng thiếu tên GPU")


def fit_scaling_model(
    observations: tuple[CalibrationObservation, ...],
    *,
    target_rows: int,
) -> dict[str, Any]:
    """Fit p90 theo cỡ mẫu; không tự loại outlier hay bịa điểm thiếu."""
    if target_rows <= 0:
        raise ValueError("target_rows phải dương")
    for observation in observations:
        observation.validate()
    selectors = {item.selector_id for item in observations}
    devices = {item.device.environment_id for item in observations}
    if len(selectors) != 1 or len(devices) != 1:
        raise ValueError("Một fit chỉ được chứa một selector và một environment_id")

    successful = [item for item in observations if item.status == "success"]
    counts = {
        rows: sum(item.rows == rows for item in successful)
        for rows in CALIBRATION_ROWS
    }
    complete = all(counts[rows] >= REPLICATES_PER_SIZE for rows in CALIBRATION_ROWS)
    evidence = {
        "schema_version": 1,
        "selector_id": next(iter(selectors), None),
        "environment_id": next(iter(devices), None),
        "required_rows": list(CALIBRATION_ROWS),
        "required_replicates": REPLICATES_PER_SIZE,
        "replicate_counts": {str(key): value for key, value in counts.items()},
        "target_rows": int(target_rows),
        "observations": [
            {
                **asdict(item),
                "device": asdict(item.device),
                "wall_clock_seconds": item.warmup_seconds + item.steady_seconds,
                "allocated_gpu_hours": (
                    (item.warmup_seconds + item.steady_seconds)
                    * item.device.gpu_count / 3600.0
                ),
            }
            for item in observations
        ],
    }
    if not complete:
        return {
            **evidence,
            "fit_status": "needs_more_calibration",
            "reason": "Thiếu ít nhất một size/replicate bắt buộc; không dùng operations thay thế",
        }

    x = np.log(np.asarray(CALIBRATION_ROWS, dtype=np.float64))
    time_p90 = np.asarray([
        np.percentile(
            [item.steady_seconds for item in successful if item.rows == rows], 90,
        )
        for rows in CALIBRATION_ROWS
    ])
    rss_p90 = np.asarray([
        np.percentile(
            [item.peak_rss_bytes for item in successful if item.rows == rows], 90,
        )
        for rows in CALIBRATION_ROWS
    ])
    time_fit = _fit_log_log(x, np.log(time_p90), target_rows)
    rss_fit = _fit_log_log(x, np.log(rss_p90), target_rows)
    return {
        **evidence,
        "fit_status": "fitted",
        "aggregation": "p90_per_size_no_outlier_removal",
        "time_model": time_fit,
        "rss_model": rss_fit,
        "predicted_seconds": float(math.exp(time_fit["target_log_point"])),
        "upper_prediction_seconds": float(math.exp(time_fit["target_log_upper"])),
        "predicted_peak_rss_bytes": float(math.exp(rss_fit["target_log_point"])),
        "upper_prediction_peak_rss_bytes": float(math.exp(rss_fit["target_log_upper"])),
    }


def gate_decision(
    fitted: dict[str, Any],
    *,
    supports_sharding: bool,
    selector_cap_seconds: float = SELECTOR_CAP_SECONDS,
) -> dict[str, Any]:
    """Ra quyết định hạ tầng; không sinh trạng thái run TIMEOUT dự báo."""
    if fitted.get("fit_status") != "fitted":
        return {
            "decision": "needs_more_calibration",
            "run_status": None,
            "reason": fitted.get("reason", "Calibration chưa đủ"),
        }
    upper = float(fitted["upper_prediction_seconds"])
    if upper <= selector_cap_seconds:
        decision = "run_direct"
    elif supports_sharding:
        decision = "run_shared_shard_wrapper"
    else:
        decision = "requires_protocol_resolution"
    return {
        "decision": decision,
        "run_status": None,
        "selector_cap_seconds": float(selector_cap_seconds),
        "upper_prediction_seconds": upper,
        "reason": (
            "Upper prediction bound nằm trong trần selector"
            if decision == "run_direct"
            else "Upper prediction bound vượt trần; không dùng operations để loại run"
        ),
    }


def _fit_log_log(x: np.ndarray, y: np.ndarray, target_rows: int) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    residuals = y - fitted
    residual_scale = float(np.sqrt(np.sum(residuals ** 2) / max(1, len(x) - 2)))
    target_x = math.log(target_rows)
    centered = float(np.sum((x - x.mean()) ** 2))
    leverage = 1.0 + 1.0 / len(x)
    if centered > 0:
        leverage += (target_x - float(x.mean())) ** 2 / centered
    point = float(intercept + slope * target_x)
    # 1.96 được dùng để tạo upper diagnostic; chưa coi đây là CI inferential.
    upper = point + 1.96 * residual_scale * math.sqrt(leverage)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_scale_log": residual_scale,
        "target_log_point": point,
        "target_log_upper": float(upper),
    }
