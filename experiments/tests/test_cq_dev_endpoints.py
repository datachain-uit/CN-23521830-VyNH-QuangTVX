"""Regression test cho canonical prediction path của CQ v1.

Runner ghi run vào ``artifacts/experiments/<experiment_id>``; endpoint reader
phải dùng cùng layout để không làm hỏng bước đóng gói M0 sau khi train xong.
"""

from pathlib import Path

from scripts.compute_cq_v1_dev_endpoints import _experiment_root


def test_endpoint_reader_uses_canonical_experiment_root() -> None:
    artifact_root = Path("artifacts")
    assert _experiment_root(artifact_root, "cq_run") == (
        artifact_root / "experiments" / "cq_run"
    )
