"""File này cung cấp các lệnh ngắn để chạy cùng một API trên local và Kaggle.

`doctor` kiểm tra môi trường, `prepare` tạo processed data, `smoke` kiểm tra luồng
nhỏ, `run` chạy stage trong config và `aggregate` gom artifact. Lệnh không chứa
thuật toán nên kết quả không phụ thuộc việc gọi từ notebook hay terminal.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .data import fetch_dataset, prepare_dataset, prepare_external_dataset
from .experiments import run_config, run_smoke
from .experiments.freeze import (
    create_freeze_manifest,
    create_freeze_manifest_v4,
    create_protocol_manifest_v4,
)
from .methods import allowed_learners_for, get_method_spec
from .reports import aggregate_results


def main(argv: list[str] | None = None) -> int:
    # Windows terminal cũ có thể dùng cp1252; đổi sang UTF-8 để phần trợ giúp tiếng Việt đọc được.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="skq", description="SKQ experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    plan = sub.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--show-cells", action="store_true")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument(
        "--selector-seed", type=int, action="append",
        help="Ghi đè selector seed; có thể lặp option để chạy nhiều seed.",
    )
    smoke.add_argument("--model-seed", type=int, help="Ghi đè seed của downstream model.")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--overwrite", action="store_true")
    prepare_external = sub.add_parser("prepare-external")
    prepare_external.add_argument("--config", required=True)
    prepare_external.add_argument("--input-dir", required=True)
    prepare_external.add_argument("--dataset", required=True)
    prepare_external.add_argument("--overwrite", action="store_true")
    fetch = sub.add_parser("fetch-data")
    fetch.add_argument("--config", required=True)
    fetch.add_argument("--dataset", required=True)
    fetch.add_argument("--overwrite", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset")
    run.add_argument("--method")
    run.add_argument(
        "--exclude-method", action="append", default=[],
        help=(
            "Bỏ method khỏi full run nhưng không đổi config/protocol hash; "
            "có thể lặp option."
        ),
    )
    run.add_argument("--learner")
    run.add_argument(
        "--resume", action="store_true",
        help="Bỏ qua các run đã có manifest và metrics khớp đúng run_id.",
    )
    run.add_argument(
        "--reuse-method-artifacts", action="store_true",
        help=(
            "Dùng lại artifact selector/generator canonical đã kiểm tra identity; "
            "hữu ích khi cô lập từng learner trong process riêng."
        ),
    )
    run.add_argument(
        "--selector-seed", type=int, action="append",
        help="Ghi đè selector seed; ví dụ --selector-seed 11. Lặp option để thêm seed.",
    )
    run.add_argument("--model-seed", type=int, help="Ghi đè seed của downstream model.")
    run.add_argument("--max-ram-gb", type=float, help="Giới hạn RSS cho selector.")
    run.add_argument("--timeout-seconds", type=float, help="Giới hạn thời gian selector.")
    run.add_argument("--max-threads", type=int, help="Số CPU thread tối đa cho learner.")
    run.add_argument(
        "--max-estimated-operations", type=float,
        help=(
            "Mốc tham chiếu diagnostic cho ước lượng phép tính; không tự loại run. "
            "Giới hạn thực thi dùng RAM/time guard đo thật."
        ),
    )
    run.add_argument("--save-model", action="store_true", help="Lưu fitted model; mặc định không lưu.")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--base-winner-method", required=True)
    freeze.add_argument("--base-winner-structure", required=True)
    freeze.add_argument("--published-reference", required=True)
    freeze.add_argument("--proposed-winner", required=True)
    freeze.add_argument("--selection-basis", required=True)
    freeze.add_argument(
        "--allowed-parent-source", action="append", default=[],
        help=(
            "Parent source đã được khai báo trước khi mở test; có thể lặp option. "
            "Mặc định chỉ cho base-winner-structure."
        ),
    )
    freeze.add_argument("--overwrite", action="store_true")
    freeze_v4 = sub.add_parser(
        "freeze-v4",
        help="Tạo M1 freeze bất biến từ config dev và file đặc tả đã duyệt.",
    )
    freeze_v4.add_argument("--config", required=True)
    freeze_v4.add_argument(
        "--spec", required=True,
        help="JSON khai báo candidate, winner, rho*, hash evidence và shard policy.",
    )
    freeze_m0 = sub.add_parser(
        "freeze-m0-v4",
        help="Khóa protocol M0 trước khi xem kết quả candidate trên dev.",
    )
    freeze_m0.add_argument("--config", required=True)
    freeze_m0.add_argument("--spec", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--artifact-root", required=True)
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(json.dumps(_doctor(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "aggregate":
        print(aggregate_results(args.artifact_root))
        return 0
    config = ExperimentConfig.from_json(args.config)
    if args.command == "plan":
        print(json.dumps(_plan(config, include_cells=args.show_cells), indent=2, ensure_ascii=False))
        return 0
    if args.command == "freeze":
        print(create_freeze_manifest(
            config,
            base_winner_method_id=args.base_winner_method,
            base_winner_structure_source=args.base_winner_structure,
            published_reference=args.published_reference,
            proposed_winner=args.proposed_winner,
            selection_basis=args.selection_basis,
            allowed_parent_sources=tuple(args.allowed_parent_source),
            overwrite=args.overwrite,
        ))
        return 0
    if args.command == "freeze-v4":
        spec_path = Path(args.spec)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        required = {
            "candidate_list", "selected_candidate", "published_reference",
            "strongest_baseline", "confirmatory_config_hash", "rho_star",
            "selection_evidence_hash", "stats_protocol_hash", "shard_policy",
            "allowed_extensions",
        }
        missing = required - set(spec)
        if missing:
            raise ValueError(f"Freeze-v4 spec thiếu trường: {sorted(missing)}")
        print(create_freeze_manifest_v4(
            config,
            candidate_list=tuple(spec["candidate_list"]),
            selected_candidate=str(spec["selected_candidate"]),
            published_reference=str(spec["published_reference"]),
            strongest_baseline=str(spec["strongest_baseline"]),
            confirmatory_config_hash=str(spec["confirmatory_config_hash"]),
            rho_star=float(spec["rho_star"]),
            selection_evidence_hash=str(spec["selection_evidence_hash"]),
            stats_protocol_hash=str(spec["stats_protocol_hash"]),
            shard_policy=dict(spec["shard_policy"]),
            allowed_extensions=tuple(spec["allowed_extensions"]),
            previous_manifest_id=spec.get("previous_manifest_id"),
            deviations=tuple(spec.get("deviations", ())),
            plan_version=str(spec.get("plan_version", "2.0.0")),
        ))
        return 0
    if args.command == "freeze-m0-v4":
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        required = {
            "candidate_list", "split_strategy", "rho_star", "n_min", "delta_c",
            "shard_policy", "resource_decision_rule", "data_card_hash",
            "headroom_evidence_hash", "resource_evidence_hash",
        }
        missing = required - set(spec)
        if missing:
            raise ValueError(f"Freeze-m0-v4 spec thiếu trường: {sorted(missing)}")
        print(create_protocol_manifest_v4(
            config,
            candidate_list=tuple(spec["candidate_list"]),
            split_strategy=str(spec["split_strategy"]),
            rho_star=float(spec["rho_star"]),
            n_min=int(spec["n_min"]),
            delta_c=float(spec["delta_c"]),
            shard_policy=dict(spec["shard_policy"]),
            resource_decision_rule=dict(spec["resource_decision_rule"]),
            data_card_hash=str(spec["data_card_hash"]),
            headroom_evidence_hash=str(spec["headroom_evidence_hash"]),
            resource_evidence_hash=str(spec["resource_evidence_hash"]),
            plan_version=str(spec.get("plan_version", "2.0.0")),
        ))
        return 0
    if getattr(args, "selector_seed", None):
        config = replace(config, selector_seeds=tuple(args.selector_seed))
    if getattr(args, "model_seed", None) is not None:
        config = replace(config, model_seed=int(args.model_seed))
    resource = dict(config.resource or {})
    for argument, key in [
        (getattr(args, "max_ram_gb", None), "max_ram_gb"),
        (getattr(args, "timeout_seconds", None), "timeout_seconds"),
        (getattr(args, "max_threads", None), "max_threads"),
        (getattr(args, "max_estimated_operations", None), "max_estimated_operations"),
    ]:
        if argument is not None:
            resource[key] = argument
    artifact_policy = dict(config.artifact_policy or {})
    if getattr(args, "save_model", False):
        artifact_policy["save_models"] = True
    config = replace(config, resource=resource, artifact_policy=artifact_policy)
    config.validate()
    if args.command == "smoke":
        print(json.dumps(run_smoke(config), indent=2, ensure_ascii=False))
        return 0
    if args.command == "prepare":
        print(prepare_dataset(
            args.dataset, config.paths.raw_root, config.paths.processed_root,
            overwrite=args.overwrite,
        ))
        return 0
    if args.command == "prepare-external":
        print(prepare_external_dataset(
            args.dataset, args.input_dir, config.paths.processed_root,
            overwrite=args.overwrite,
        ))
        return 0
    if args.command == "fetch-data":
        print(fetch_dataset(args.dataset, config.paths.raw_root, overwrite=args.overwrite))
        return 0
    print(json.dumps(run_config(
        config, dataset_id=args.dataset, method_id=args.method, learner_id=args.learner,
        exclude_method_ids=tuple(getattr(args, "exclude_method", ())),
        resume=bool(getattr(args, "resume", False)),
        reuse_method_artifacts=bool(getattr(args, "reuse_method_artifacts", False)),
    ), indent=2, ensure_ascii=False, default=str))
    return 0


def _doctor() -> dict:
    core = ["numpy", "scipy", "pandas", "sklearn", "xgboost", "catboost", "torch", "psutil", "ahocorasick", "statsmodels"]
    optional = [
        "faiss", "fast_enum", "iterstrat", "jax", "neural_tangents", "torchvision",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages_core": {name: importlib.util.find_spec(name) is not None for name in core},
        "packages_optional_native_synthetic": {
            name: importlib.util.find_spec(name) is not None for name in optional
        },
        "cwd": str(Path.cwd()),
    }


def _plan(config: ExperimentConfig, *, include_cells: bool = False) -> dict:
    cells = []
    for method_id in config.method_ids:
        spec = get_method_spec(method_id)
        allowed = allowed_learners_for(
            method_id, (config.method_options or {}).get(method_id),
        )
        for learner_id in config.learner_ids:
            contract = "planned"
            if allowed is not None and learner_id not in allowed:
                contract = "na_contract"
            cells.append({
                "method_id": method_id,
                "role": spec.role,
                "output_kind": spec.output_kind,
                "learner_id": learner_id,
                "contract": contract,
            })
    if config.stage_id in {"s2_confirm", "s4_temporal"}:
        evaluation_splits = (
            [f"test_phase{phase}" for phase in range(1, 5)]
            if "course_quality_med_v1" in config.dataset_ids else ["test"]
        )
    else:
        evaluation_splits = (
            [f"dev_phase{phase}" for phase in range(1, 5)]
            if any(dataset.startswith("course_quality_v1") for dataset in config.dataset_ids)
            else ["dev"]
        )
    result = {
        "experiment_id": config.experiment_id,
        "datasets": list(config.dataset_ids),
        "selector_seeds": list(config.selector_seeds),
        "methods_total_including_fulltrain": len(config.method_ids),
        "compression_or_generation_methods": sum(method != "c00_full_train" for method in config.method_ids),
        "learners": list(config.learner_ids),
        "cells_total_per_dataset_seed": len(cells),
        "cells_planned": sum(cell["contract"] == "planned" for cell in cells),
        "cells_na_contract": sum(cell["contract"] == "na_contract" for cell in cells),
        "evaluation_splits": evaluation_splits,
        "evaluation_rows_total_per_dataset_seed": len(cells) * len(evaluation_splits),
    }
    if include_cells:
        result["cells"] = cells
    return result


if __name__ == "__main__":
    raise SystemExit(main())
