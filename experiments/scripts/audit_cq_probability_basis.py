"""Y30: verify raw versus A8 log loss in existing natural/balanced DEV archives.

Read only four FullTrain A/B seed42 runs from each ZIP, check prediction hashes,
recompute raw log loss per phase, and compare with the archived result.json.
Never reads test, train_support or model checkpoints; never fits a model.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh report directory")
    rows = []
    source = {}
    for policy, path in (("natural", args.natural), ("balanced", args.balanced)):
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("test_read") is not False:
                raise ValueError("Archive must explicitly be dev-only")
            source[policy] = {"archive": path.name,
                              "manifest_sha256": hashlib.sha256(archive.read("manifest.json")).hexdigest()}
            for pool in "AB":
                for task in ("binary", "three"):
                    prefix = f"{task}/Full{pool}_ms42/"
                    receipt = json.loads(archive.read(prefix + "COMPLETED.json"))
                    result = json.loads(archive.read(prefix + "result.json"))
                    data = archive.read(prefix + "predictions.parquet")
                    assert hashlib.sha256(data).hexdigest() == receipt["files"]["predictions.parquet"]
                    frame = pd.read_parquet(io.BytesIO(data))
                    labels = result["labels"]
                    if set(frame.phase) != {"P1", "P2", "P3", "P4"}:
                        raise ValueError("Four dev phases required")
                    losses = []
                    for phase in ("P1", "P2", "P3", "P4"):
                        part = frame[frame.phase == phase]
                        assert not part.row_id.duplicated().any()
                        p = part[["p_" + label for label in labels]].to_numpy(float)
                        p /= p.sum(1, keepdims=True)
                        value = log_loss(part.y, p, labels=np.arange(len(labels)))
                        assert np.isclose(value, result["per_phase"][phase]["LL"], atol=2e-7, rtol=2e-7)
                        losses.append(float(value))
                    ll = float(np.mean(losses))
                    assert np.isclose(ll, result["summary"]["LL"], atol=2e-7, rtol=2e-7)
                    rows.append({"pool": pool, "task": task, "policy": policy,
                                 "model_seed": 42, "patience": 20,
                                 "raw_LL_recomputed": ll,
                                 "A8_LL_archived_separate": result["summary_a8"]["LL"],
                                 "raw_per_phase": dict(zip(("P1","P2","P3","P4"), losses)),
                                 "prediction_sha256": receipt["files"]["predictions.parquet"],
                                 "probability_basis_in_previous_notebook": "raw p_* columns, NOT A8 p_* columns"})
    args.output.mkdir(parents=True)
    (args.output / "probability_basis_audit.json").write_text(json.dumps({
        "review": "2026-10-25_Y30", "test_data_read": False, "new_model_fits": 0,
        "archives_full_contents_not_audited": True, "source": source, "rows": rows},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = ["# Loại xác suất trong bảng natural và balanced", "",
          "Đã kiểm lại predictions của FullTrain-A/B seed42, hai/ba nhãn, trong hai ZIP natural/balanced. Các log loss ở bảng notebook ngày 24/10 tính trên xác suất thô, không phải sau A8. Chỉ đọc các member dev cần thiết và kiểm hash; không kiểm toàn bộ các lượt khác trong ZIP.", "",
          "| Model | Task | Trọng số | LL thô tính lại | LL sau A8 lưu riêng |",
          "|---|---|---|---:|---:|"]
    for row in rows:
        md.append(f"| {row['pool']} | {row['task']} | {row['policy']} | {row['raw_LL_recomputed']:.6f} | {row['A8_LL_archived_separate']:.6f} |")
    md += ["", "Cả tám lượt đều là V4 view, patience20 và model seed42. Không dùng để chọn target, chính sách trọng số hay cấu hình chính. Không kết luận balanced tốt hơn chỉ từ LL thô; ULA8 có thể cho đánh đổi khác. Phân tích hai nhãn chỉ là thăm dò theo QĐ32."]
    (args.output / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"PASS {len(rows)} existing model records, raw LL reproduced; fits=0; test=false")


if __name__ == "__main__":
    main()
