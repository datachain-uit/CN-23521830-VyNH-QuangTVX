"""File này quét các run đã lưu và tạo một bảng kết quả phẳng.

Mỗi hàng giữ run_id, cấu hình chính, trạng thái và metric tổng thể. Run thất bại
vẫn xuất hiện trong bảng để failure rate không bị làm đẹp bằng cách xóa dữ liệu.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def aggregate_results(artifact_root: str | Path) -> Path:
    root = Path(artifact_root)
    rows = []
    for manifest_path in root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = manifest_path.parent / "metrics_overall.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        failure_path = manifest_path.parent / "evaluation_failure.json"
        failure = (
            json.loads(failure_path.read_text(encoding="utf-8"))
            if failure_path.exists() else {}
        )
        cost_path = manifest_path.parent / "cost.json"
        cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else {}
        status = failure.get("status", manifest.get("status"))
        reason = failure.get("reason", manifest.get("reason"))
        artifact_complete = metrics_path.exists()
        if status == "success" and not artifact_complete:
            status = "interrupted"
            reason = reason or "Manifest đã ghi nhưng chưa có metrics_overall.json"
        rows.append({
            **manifest, **metrics, "status": status, "reason": reason,
            "artifact_complete": artifact_complete,
            **{f"cost_{key}": value for key, value in cost.items() if not isinstance(value, (dict, list))},
            "artifact_dir": str(manifest_path.parent),
        })
    frame = pd.DataFrame(rows)
    frame = _attach_break_even(frame)
    output = root / "aggregate" / "runs_long.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if not frame.empty and "status" in frame:
        summary = (
            frame.groupby(["dataset_id", "method_id", "status"], dropna=False)
            .size().rename("runs").reset_index()
        )
        summary.to_csv(output.parent / "status_summary.csv", index=False)
    return output


def _attach_break_even(frame: pd.DataFrame) -> pd.DataFrame:
    """Tính số lần train cần lặp để bù chi phí chọn coreset."""
    if frame.empty or "cost_fit_seconds" not in frame:
        return frame
    full = frame[frame.get("method_id") == "c00_full_train"]
    lookup = {
        (row.dataset_id, row.learner_id, row.model_seed): row.cost_fit_seconds
        for row in full.itertuples()
        if pd.notna(row.cost_fit_seconds)
    }
    values = []
    for row in frame.itertuples():
        full_fit = lookup.get((row.dataset_id, row.learner_id, row.model_seed))
        fit = getattr(row, "cost_fit_seconds", None)
        selection = getattr(row, "cost_selection_seconds", None)
        if full_fit is None or pd.isna(fit) or pd.isna(selection) or full_fit <= fit:
            values.append(None)
        else:
            values.append(float(selection / (full_fit - fit)))
    frame = frame.copy()
    frame["break_even_reuses"] = values
    return frame
