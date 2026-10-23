"""Test này chạy một luồng nhỏ từ config đến artifact kết quả.

Mục tiêu là phát hiện lỗi nối module trước khi chạy dataset thật. Nó dùng toy data,
LR và hai selector nhẹ nên hoàn thành nhanh và không cần Internet hay GPU.
"""

import json
from pathlib import Path

from skq_exp.config import ExperimentConfig
from skq_exp.experiments import run_smoke


def test_smoke_pipeline_writes_results(tmp_path: Path) -> None:
    project = Path(__file__).parents[1]
    raw = json.loads((project / "configs" / "s0_smoke.json").read_text(encoding="utf-8"))
    raw["paths"]["artifact_root"] = str(tmp_path / "artifacts")
    config_path = tmp_path / "s0_smoke.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config = ExperimentConfig.from_json(config_path)
    rows = run_smoke(config)
    assert len(rows) == 2
    assert all(row["status"] == "success" for row in rows)
    assert (config.paths.artifact_root / "smoke" / config.experiment_id / "smoke_results.json").exists()

