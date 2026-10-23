"""File này điều phối toàn bộ ma trận selector/generator → learner → metric.

Các method chạy theo thứ tự trong config để native structure có thể cấp cho
proposed và selection thắng có thể cấp cho D05. Dev được dùng ở screening; test
chỉ mở ở confirmatory có freeze manifest. Mọi BLOCKED/OOM/TIMEOUT vẫn vào ledger.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .. import __version__
from ..artifacts import (
    ArtifactLayout,
    ArtifactStorageError,
    atomic_json,
    sha256_file,
    stable_hash,
)
from ..config import ExperimentConfig
from ..data import load_processed
from ..data.toy import make_toy_data
from ..methods import allowed_learners_for, build_generator, build_selector, get_method_spec
from ..methods.diagnostics import equal_group_weights
from ..methods.proposed.allocation import make_group_ids
from ..methods.proposed.query_losses import make_oof_query_losses
from ..methods.proposed.cq_ablations import (
    A2, A4, C03_MATCHED, CQ_ABLATIONS,
    derive_phase_class_selection, resolve_ablation_options,
)
from ..methods.result import SelectionResult
from ..methods.synthetic.result import GeneratedDatasetResult
from ..resources import (
    ResourceGuard,
    ResourceLimitError,
    enforce_preflight,
    estimate_selection_cost,
    operation_diagnostics,
)
from ..training import (
    evaluate_fitted_learner,
    fit_generated_learner,
    fit_selection_learner,
)
from .provenance import dataset_fingerprint, protocol_hash


LRQ_METHODS = {
    "p04_skq_lrq_sq",
    "p05_skq_lrq_mq",
    "p08_skq_gonzalez_lrq_sq",
    "p09_skq_gonzalez_lrq_mq",
    "skq_pc_lrq_sq",
}
GONZALEZ_PARENT_PREFIX = "gonzalez_pool_"


def run_smoke(config: ExperimentConfig) -> list[dict[str, Any]]:
    data = make_toy_data(config.split_seed)
    rows = []
    for method_id in config.method_ids:
        spec = get_method_spec(method_id)
        if spec.output_kind != "selection" or spec.status != "implemented":
            continue
        selector = build_selector(method_id, config.selector_seeds[0], n_components=128)
        kwargs = {"parent_ids": data["parent_train"]} if spec.needs_parent else {}
        selection = selector.select(
            data["X_train"], data["y_train"], config.budget_ratio,
            row_ids=data["row_ids_train"], **kwargs,
        )
        fitted = fit_selection_learner(
            selection, data["X_train"], data["y_train"],
            learner_id="lr", model_seed=config.model_seed,
        )
        metrics, _predictions = evaluate_fitted_learner(
            fitted, data["X_dev"], data["y_dev"],
        )
        rows.append({
            "method_id": method_id, "status": selection.status,
            "realized_rows": selection.realized_rows,
            "macro_f1": metrics["overall"]["f1_macro"], "mcc": metrics["overall"]["mcc"],
        })
    output = config.paths.artifact_root / "smoke" / config.experiment_id
    atomic_json(output / "smoke_results.json", rows)
    return rows


def run_config(
    config: ExperimentConfig,
    *,
    dataset_id: str | None = None,
    method_id: str | None = None,
    learner_id: str | None = None,
    exclude_method_ids: tuple[str, ...] = (),
    resume: bool = False,
    reuse_method_artifacts: bool = False,
) -> list[dict[str, Any]]:
    if config.stage_id == "s0_smoke":
        return run_smoke(config)
    datasets = [dataset_id] if dataset_id else list(config.dataset_ids)
    freeze_manifest = _load_freeze_manifest(
        config,
        required=config.stage_id == "s2_confirm" or config.requires_freeze_manifest,
    )
    excluded = set(exclude_method_ids)
    unknown_excluded = excluded - set(config.method_ids)
    if unknown_excluded:
        raise ValueError(
            f"Method loại trừ không có trong config: {sorted(unknown_excluded)}"
        )
    if method_id is not None and method_id in excluded:
        raise ValueError(f"Method {method_id} vừa được chọn vừa bị loại trừ")
    methods = [method_id] if method_id else [
        item for item in config.method_ids if item not in excluded
    ]
    if config.stage_id == "s2_confirm" and method_id is None:
        role_map = freeze_manifest["method_roles"]
        methods.extend(str(role_map[role]) for role in config.frozen_method_roles)
        methods = [item for item in dict.fromkeys(methods) if item not in excluded]
    if not methods:
        raise ValueError("Không còn method nào để chạy sau khi loại trừ")
    learners = [learner_id] if learner_id else list(config.learner_ids)
    ledger: list[dict[str, Any]] = []
    layout = ArtifactLayout(config.paths.artifact_root)
    for current_dataset in datasets:
        data = load_processed(current_dataset, config.paths.processed_root)
        data_fingerprint = dataset_fingerprint(config, current_dataset)
        if config.stage_id == "s2_confirm" or config.requires_freeze_manifest:
            expected = freeze_manifest["dataset_fingerprints"].get(current_dataset)
            if expected != data_fingerprint:
                raise RuntimeError(
                    f"Freeze fingerprint không khớp {current_dataset}: {expected} != {data_fingerprint}"
                )
        eval_names = _evaluation_splits(config, current_dataset, data)
        selection_cache: dict[tuple[str, int], SelectionResult] = {}
        structure_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        query_cache: dict[tuple[tuple[str, ...], int], tuple[np.ndarray, dict]] = {}
        for current_method in methods:
            is_full_reference = current_method in {"c00_full_train", "c00_full_train_a"}
            seeds = [0] if is_full_reference else list(config.selector_seeds)
            for selector_seed in seeds:
                artifact_identity = _method_artifact_identity(
                    config, current_dataset, current_method, selector_seed, data_fingerprint,
                )
                method_artifact_dir = layout.method_artifact_dir(
                    config.experiment_id, config.protocol_id, config.stage_id,
                    current_dataset, current_method, config.budget_ratio, selector_seed,
                )
                result = None
                if reuse_method_artifacts and not is_full_reference:
                    result = layout.load_method_artifact(
                        method_artifact_dir, expected_identity=artifact_identity,
                    )
                if result is None:
                    result = _produce_or_block(
                        current_method, selector_seed, config, data, current_dataset,
                        selection_cache, structure_cache, query_cache, data_fingerprint,
                        freeze_manifest,
                    )
                if result.status == "success":
                    artifact_y = np.asarray(
                        data["y_fulltrain_a_train"]
                        if current_method == "c00_full_train_a"
                        else data["y_train"]
                    )
                    _attach_generic_artifact_diagnostics(result, artifact_y)
                if isinstance(result, SelectionResult) and result.status == "success":
                    selection_cache[(current_method, selector_seed)] = result
                artifact_ref = None
                if result.status == "success" and not is_full_reference:
                    try:
                        _enforce_storage(config, layout, reserve_bytes=_result_nbytes(result))
                        artifact_ref = layout.save_method_artifact(
                            method_artifact_dir, result, artifact_identity,
                        )
                    except ArtifactStorageError as error:
                        result = _failure_like(
                            result, current_method, "storage_limit", str(error),
                        )
                for current_learner in learners:
                    common_run_dir = layout.run_dir(
                        config.experiment_id, config.protocol_id, config.stage_id,
                        current_dataset, current_method, config.budget_ratio,
                        selector_seed, current_learner, config.model_seed,
                    )
                    pending: list[tuple[str, dict[str, Any], Path]] = []
                    learner_rows: list[dict[str, Any]] = []
                    for eval_name in eval_names:
                        manifest = _manifest(
                            config, current_dataset, current_method, current_learner,
                            selector_seed, eval_name, data_fingerprint,
                        )
                        run_dir = (
                            common_run_dir / f"eval-{eval_name}"
                            if len(eval_names) > 1 else common_run_dir
                        )
                        if resume:
                            completed = _load_completed_run(run_dir, manifest)
                            if completed is not None:
                                learner_rows.append(completed)
                                continue
                        pending.append((eval_name, manifest, run_dir))

                    if not pending:
                        ledger.extend(learner_rows)
                        _persist_ledger_rows(config, learner_rows)
                        _save_temporal_summary(common_run_dir, learner_rows, eval_names)
                        continue

                    allowed = allowed_learners_for(
                        current_method,
                        (config.method_options or {}).get(current_method),
                    )
                    if allowed is not None and current_learner not in allowed:
                        blocked = _failure_like(
                            result, current_method, "na_contract",
                            f"Protocol chỉ cho {current_method} chạy với {allowed}",
                        )
                        for _eval_name, manifest, run_dir in pending:
                            _save_failure_manifest(run_dir, manifest, blocked)
                            learner_rows.append({
                                **manifest,
                                "status": "na_contract",
                                "reason": blocked.diagnostics["reason"],
                            })
                        ledger.extend(learner_rows)
                        _persist_ledger_rows(config, learner_rows)
                        _save_temporal_summary(common_run_dir, learner_rows, eval_names)
                        continue
                    if result.status != "success":
                        for _eval_name, manifest, run_dir in pending:
                            _save_failure_manifest(run_dir, manifest, result)
                            learner_rows.append({
                                **manifest,
                                "status": result.status,
                                "reason": result.diagnostics.get("reason"),
                            })
                        ledger.extend(learner_rows)
                        _persist_ledger_rows(config, learner_rows)
                        _save_temporal_summary(common_run_dir, learner_rows, eval_names)
                        continue

                    for _eval_name, manifest, run_dir in pending:
                        if is_full_reference:
                            layout.save_full_reference(run_dir, result, manifest)
                        else:
                            if artifact_ref is None:
                                raise RuntimeError(
                                    "Method thành công nhưng thiếu canonical artifact reference"
                                )
                            layout.save_run_reference(
                                run_dir, result, manifest, artifact_ref,
                            )

                    try:
                        _enforce_storage(
                            config, layout, reserve_bytes=2 * 1024 * 1024,
                        )
                        max_threads = int(
                            (config.resource or {}).get("max_threads", 4)
                        )
                        learner_params = (config.learner_params or {}).get(
                            current_learner,
                        )
                        X_dev = np.asarray(data["X_dev"])
                        y_dev = np.asarray(data["y_dev"])
                        if isinstance(result, GeneratedDatasetResult):
                            fitted = fit_generated_learner(
                                result,
                                np.asarray(data["y_train"]),
                                learner_id=current_learner,
                                model_seed=config.model_seed,
                                learner_params=learner_params,
                                X_inner_val=X_dev,
                                y_inner_val=y_dev,
                                max_threads=max_threads,
                                class_weight_policy=config.class_weight_policy,
                                early_stopping=config.early_stopping,
                            )
                        else:
                            fit_X = np.asarray(
                                data["X_fulltrain_a_train"]
                                if current_method == "c00_full_train_a"
                                else data["X_train"]
                            )
                            fit_y = np.asarray(
                                data["y_fulltrain_a_train"]
                                if current_method == "c00_full_train_a"
                                else data["y_train"]
                            )
                            fitted = fit_selection_learner(
                                result,
                                fit_X,
                                fit_y,
                                learner_id=current_learner,
                                model_seed=config.model_seed,
                                learner_params=learner_params,
                                X_inner_val=X_dev,
                                y_inner_val=y_dev,
                                max_threads=max_threads,
                                class_weight_policy=config.class_weight_policy,
                                early_stopping=config.early_stopping,
                            )
                    except MemoryError:
                        _save_evaluation_failures(
                            pending, learner_rows, "oom", None,
                        )
                        ledger.extend(learner_rows)
                        _persist_ledger_rows(config, learner_rows)
                        _save_temporal_summary(common_run_dir, learner_rows, eval_names)
                        continue
                    except Exception as error:
                        _save_evaluation_failures(
                            pending, learner_rows, "failed", repr(error),
                        )
                        ledger.extend(learner_rows)
                        _persist_ledger_rows(config, learner_rows)
                        _save_temporal_summary(common_run_dir, learner_rows, eval_names)
                        continue

                    for eval_index, (eval_name, manifest, run_dir) in enumerate(pending):
                        try:
                            metrics, predictions = evaluate_fitted_learner(
                                fitted,
                                np.asarray(data[f"X_{eval_name}"]),
                                np.asarray(data[f"y_{eval_name}"]),
                                evaluation_split_count=len(eval_names),
                            )
                            row_key = f"row_ids_{eval_name}"
                            course_key = f"course_groups_{eval_name}"
                            family_key = f"course_family_groups_{eval_name}"
                            if row_key in data:
                                predictions["row_id"] = np.asarray(data[row_key])
                            if course_key in data:
                                predictions["course_id"] = np.asarray(data[course_key])
                            if family_key in data:
                                predictions["course_family_id"] = np.asarray(data[family_key])
                            phase_policy = dict(config.artifact_policy or {})
                            if eval_index > 0:
                                phase_policy["save_models"] = False
                            _save_evaluation(
                                run_dir,
                                fitted.model,
                                metrics,
                                predictions,
                                policy=phase_policy,
                                layout=layout,
                                experiment_id=config.experiment_id,
                                reusable_artifact_bytes=int(
                                    (artifact_ref or {}).get("artifact_bytes", 0)
                                ),
                            )
                            row = {
                                **manifest,
                                "status": "success",
                                "macro_f1": metrics["overall"]["f1_macro"],
                                "mcc": metrics["overall"]["mcc"],
                            }
                        except ArtifactStorageError as error:
                            row = {
                                **manifest,
                                "status": "storage_limit",
                                "reason": str(error),
                            }
                            atomic_json(run_dir / "evaluation_failure.json", row)
                        except MemoryError:
                            row = {**manifest, "status": "oom"}
                            atomic_json(run_dir / "evaluation_failure.json", row)
                        except Exception as error:
                            row = {
                                **manifest,
                                "status": "failed",
                                "reason": repr(error),
                            }
                            atomic_json(run_dir / "evaluation_failure.json", row)
                        learner_rows.append(row)
                    ledger.extend(learner_rows)
                    _persist_ledger_rows(config, learner_rows)
                    _save_temporal_summary(common_run_dir, learner_rows, eval_names)
    _persist_ledger_rows(config, ledger)
    return ledger


def _persist_ledger_rows(
    config: ExperimentConfig, rows: list[dict[str, Any]],
) -> None:
    """Checkpoint ledger sau từng learner để SIGKILL không xóa tiến độ trước đó."""
    if not rows:
        return
    ledger_path = config.paths.artifact_root / config.experiment_id / "run_ledger.json"
    existing = []
    if ledger_path.exists():
        try:
            existing = json.loads(ledger_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    merged = {row.get("run_id", stable_hash(row)[:20]): row for row in existing}
    merged.update({row.get("run_id", stable_hash(row)[:20]): row for row in rows})
    atomic_json(ledger_path, list(merged.values()))


def _evaluation_splits(
    config: ExperimentConfig,
    dataset_id: str,
    data: dict[str, Any],
) -> tuple[str, ...]:
    """Chọn split theo stage; temporal CourseQuality bắt buộc đủ bốn phase."""
    if config.stage_id not in {"s2_confirm", "s4_temporal"}:
        if dataset_id.startswith("course_quality_v1") or dataset_id == "course_quality_v4":
            dev_phases = tuple(
                f"dev_phase{phase}"
                for phase in range(1, 5)
                if f"X_dev_phase{phase}" in data and f"y_dev_phase{phase}" in data
            )
            if len(dev_phases) != 4:
                raise RuntimeError(
                    "CQ development phải đánh giá đủ dev P1-P4; test vẫn khóa"
                )
            return dev_phases
        return ("dev",)
    phases = tuple(
        f"test_phase{phase}"
        for phase in range(1, 5)
        if f"X_test_phase{phase}" in data and f"y_test_phase{phase}" in data
    )
    if dataset_id == "course_quality_med_v1":
        if len(phases) != 4:
            raise RuntimeError(
                "CourseQuality phải có đủ test_phase1..test_phase4 để đánh giá"
            )
        return phases
    return phases or ("test",)


def _save_evaluation_failures(
    pending: list[tuple[str, dict[str, Any], Path]],
    rows: list[dict[str, Any]],
    status: str,
    reason: str | None,
) -> None:
    for _eval_name, manifest, run_dir in pending:
        row = {**manifest, "status": status}
        if reason is not None:
            row["reason"] = reason
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "evaluation_failure.json", row)
        rows.append(row)


def _save_temporal_summary(
    common_run_dir: Path,
    rows: list[dict[str, Any]],
    eval_names: tuple[str, ...],
) -> None:
    if len(eval_names) <= 1:
        return
    by_split = {str(row.get("evaluation_split")): row for row in rows}
    ordered = [by_split[name] for name in eval_names if name in by_split]
    successful = [row for row in ordered if row.get("status") == "success"]
    macro_values = [float(row["macro_f1"]) for row in successful]
    mcc_values = [float(row["mcc"]) for row in successful]
    atomic_json(common_run_dir / "temporal_summary.json", {
        "evaluation_splits": list(eval_names),
        "all_splits_present": len(ordered) == len(eval_names),
        "all_splits_success": len(successful) == len(eval_names),
        "status_by_split": {
            str(row.get("evaluation_split")): str(row.get("status"))
            for row in ordered
        },
        "macro_f1_by_split": {
            str(row["evaluation_split"]): row.get("macro_f1") for row in ordered
        },
        "mcc_by_split": {
            str(row["evaluation_split"]): row.get("mcc") for row in ordered
        },
        "macro_f1_mean": float(np.mean(macro_values)) if macro_values else None,
        "macro_f1_worst": min(macro_values) if macro_values else None,
        "mcc_mean": float(np.mean(mcc_values)) if mcc_values else None,
        "mcc_worst": min(mcc_values) if mcc_values else None,
        "fit_contract": "one_fitted_model_reused_for_all_temporal_splits",
    })


def _produce_or_block(
    method_id, seed, config, data, dataset_id, selection_cache, structure_cache,
    query_cache, data_fingerprint, freeze_manifest,
):
    y_train = np.asarray(data["y_train"])
    X_learner_train = np.asarray(data["X_train"])
    X_train = np.asarray(data.get("X_selector_train", X_learner_train))
    requested = max(1, int(round(len(y_train) * config.budget_ratio)))
    if method_id in {"c00_full_train", "c00_full_train_a"}:
        reference_y = (
            np.asarray(data.get("y_fulltrain_a_train", np.empty(0, dtype=y_train.dtype)))
            if method_id == "c00_full_train_a" else y_train
        )
        if method_id == "c00_full_train_a" and not len(reference_y):
            return SelectionResult.failure(
                method_id, "blocked", requested,
                "CQ V4 chưa có FullTrain-A lambda=1 đã qua cohort validation",
            )
        indices = np.arange(len(reference_y), dtype=np.int64)
        return SelectionResult(
            indices,
            np.ones(len(indices)),
            len(indices),
            len(indices),
            "full",
            method_id,
            diagnostics={
                "fulltrain_contract": (
                    "lambda_1_same_v4_train_cohort"
                    if method_id == "c00_full_train_a" else "pool_b_train"
                ),
                "passes_over_train": 0.0,
                "passes_over_train_kind": "exact_logical",
                "passes_over_train_definition": (
                    "FullTrain không chạy selector; lượt đọc khi fit learner được "
                    "báo trong chi phí huấn luyện, không tính vào selection pass."
                ),
            },
        )
    options = dict((config.method_options or {}).get(method_id, {}))
    lrq_provenance: dict[str, Any] = {}
    if options.get("parent_source") == "@base_winner_structure":
        if freeze_manifest is None or not freeze_manifest.get("base_winner_structure_source"):
            return SelectionResult.failure(
                method_id, "gate_locked", requested,
                "Method cần frozen base_winner_structure_source",
            )
        options["parent_source"] = freeze_manifest["base_winner_structure_source"]
    if options.get("source_method") == "@proposed_winner":
        frozen_roles = (freeze_manifest or {}).get("method_roles", {})
        if not frozen_roles.get("proposed_winner"):
            return SelectionResult.failure(
                method_id, "gate_locked", requested,
                "Method cần frozen proposed_winner",
            )
        options["source_method"] = frozen_roles["proposed_winner"]
    if method_id in CQ_ABLATIONS:
        if config.stage_id != "s1_screen" or not config.test_locked:
            return SelectionResult.failure(method_id, "gate_locked", requested, "Prepared CQ ablations are train/dev-only; test requires a separate reviewed freeze path")
        try:
            options = resolve_ablation_options(method_id, config.method_options or {})
        except ValueError as error:
            return SelectionResult.failure(method_id, "blocked", requested, str(error))
    query_method = options["source_method"] if method_id == A2 else method_id
    if query_method in LRQ_METHODS:
        gate_reason, lrq_provenance = _validate_lrq_gate(
            config, options, freeze_manifest, method_id=query_method
        )
        if gate_reason is not None:
            return SelectionResult.failure(
                method_id, "gate_locked", requested,
                gate_reason,
            )
    if method_id == "n_autocoreset_native":
        processed = config.paths.processed_root / dataset_id
        fingerprint = stable_hash({
            "X_train": sha256_file(processed / "X_train.npy"),
            "y_train": sha256_file(processed / "y_train.npy"),
        })
        options.setdefault("dataset_fingerprint", fingerprint)
        options.setdefault(
            "native_artifact_dir",
            str(config.paths.artifact_root / "native" / "autocoreset" / dataset_id / f"ss{seed}"),
        )
    spec = get_method_spec(method_id)
    estimate = estimate_selection_cost(
        method_id, len(y_train), X_train.shape[1], requested,
        rff_components=int(options.get("n_components", 256)),
        n_classes=len(np.unique(y_train)),
        structured_work_rows=(
            0.0
            if method_id.startswith(("p", "d02", "d04", "d05"))
            else None
        ),
        streaming_rff_rows=(
            min(len(y_train), int(options.get("shard_rows", len(y_train))))
            if method_id in {"p10_skq_mr_coretab_xgb", "p11_skq_mr_bdis"}
            else None
        ),
    )
    resource_policy = config.resource or {}
    resource_diagnostics = {
        "estimated_working_bytes": int(estimate.working_bytes),
        **operation_diagnostics(estimate, resource_policy),
        "selector_view": (
            "dedicated"
            if "X_selector_train" in data
            else "shared_legacy"
        ),
        "selector_feature_count": int(X_train.shape[1]),
        "learner_feature_count": int(X_learner_train.shape[1]),
        "feature_contract_hash": (
            (data.get("feature_contract") or {}).get("contract_hash")
        ),
    }
    try:
        enforce_preflight(estimate, resource_policy)
    except ResourceLimitError as error:
        failure = _failure_for_kind(
            spec.output_kind, method_id, error.status, requested, str(error)
        )
        failure.diagnostics.update({
            "resource_failure": True,
            "resource_gate": "preflight",
            "resource_status": error.status,
            "estimated_working_bytes": int(estimate.working_bytes),
            "estimated_operations": float(estimate.estimated_operations),
            "configured_max_ram_gb": float(resource_policy.get("max_ram_gb", 8.0)),
            "configured_max_estimated_operations": float(
                resource_policy.get("max_estimated_operations", 50_000_000_000)
            ),
        })
        return failure
    guard = ResourceGuard(resource_policy)
    if spec.output_kind == "synthetic":
        try:
            generator = build_generator(method_id, seed, external_root=config.paths.external_root, options=options)
            generated = generator.generate(
                X_train, y_train, config.budget_ratio, resource_guard=guard,
            )
            generated.diagnostics.update(resource_diagnostics)
            return generated
        except ResourceLimitError as error:
            failure = GeneratedDatasetResult.failure(
                method_id, error.status, requested, str(error),
            )
            failure.diagnostics.update(resource_diagnostics)
            return failure
        except Exception as error:
            return GeneratedDatasetResult.failure(method_id, "failed", requested, repr(error))
    if method_id in {A4, C03_MATCHED}:
        source = options["source_method"]
        base = selection_cache.get((source, seed))
        if base is None:
            base = _load_selection_artifact(config, dataset_id, source, seed)
        if base is None or base.method_id != source:
            return SelectionResult.failure(method_id, "blocked", requested, f"CQ ablation needs its declared source selection first: {source}")
        result = derive_phase_class_selection(
            method_id, base, y_train, data.get("phase_train"),
            config.budget_ratio, seed=seed, resource_guard=guard,
        )
        result.diagnostics.update(resource_diagnostics)
        return result
    if method_id == "d05_equal_group_weight":
        source = str(options.get("source_method", "p05_skq_lrq_mq"))
        base = selection_cache.get((source, seed))
        if base is None:
            base = _load_selection_artifact(config, dataset_id, source, seed)
        if base is None:
            return SelectionResult.failure(method_id, "blocked", requested, f"D05 cần source selection chạy trước: {source}")
        parent_result = _resolve_structure(
            options, seed, structure_cache, config, dataset_id,
            np.asarray(data["row_ids_train"]), data_fingerprint,
        )
        if isinstance(parent_result, str):
            return SelectionResult.failure(method_id, "blocked", requested, parent_result)
        group_ids = make_group_ids(parent_result.get("parent_ids"), y_train)
        result = equal_group_weights(base, group_ids)
        result.diagnostics.update(resource_diagnostics)
        return result
    try:
        selector = build_selector(
            method_id, seed, external_root=config.paths.external_root,
            options=options, n_components=int(options.get("n_components", 256)),
        )
    except (KeyError, RuntimeError) as error:
        return SelectionResult.failure(method_id, "blocked", requested, str(error))
    kwargs: dict[str, Any] = {
        "row_ids": np.asarray(data["row_ids_train"]),
        "resource_guard": guard,
        "resource_policy": config.resource,
    }
    if "phase_train" in data:
        kwargs["phases"] = np.asarray(data["phase_train"])
    structure = None
    if spec.needs_parent:
        parent_source = str(options.get("parent_source", ""))
        if parent_source.startswith(GONZALEZ_PARENT_PREFIX):
            parent_failure = _ensure_gonzalez_candidate_structure(
                config=config,
                dataset_id=dataset_id,
                seed=seed,
                X_train=X_train,
                y_train=y_train,
                row_ids=np.asarray(data["row_ids_train"]),
                data_fingerprint=data_fingerprint,
                options=options,
            )
            if parent_failure is not None:
                return SelectionResult.failure(
                    method_id, parent_failure[0], requested, parent_failure[1]
                )
        structure = _resolve_structure(
            options, seed, structure_cache, config, dataset_id,
            np.asarray(data["row_ids_train"]), data_fingerprint,
        )
        if isinstance(structure, str):
            return SelectionResult.failure(method_id, "blocked", requested, structure)
        kwargs["parent_ids"] = structure["parent_ids"]
        if method_id == "p03_skq_bdis_filtered" or bool(
            options.get("use_parent_candidate_mask", False)
        ):
            if "candidate_mask" not in structure:
                return SelectionResult.failure(
                    method_id, "blocked", requested,
                    "Parent structure thiếu candidate_mask",
                )
            kwargs["candidate_mask"] = structure["candidate_mask"]
    if query_method in LRQ_METHODS:
        query_ids = tuple(options.get("query_learners", ["lr"] if query_method.endswith("sq") else ["lr", "rf", "xgb"]))
        cache_key = (query_ids, seed)
        query_cache_hit = cache_key in query_cache
        if cache_key not in query_cache:
            try:
                query_started = time.perf_counter()
                query_values, query_metadata = make_oof_query_losses(
                    X_learner_train, y_train, query_ids, seed=seed,
                    folds=int(options.get("oof_folds", 5)),
                    max_threads=int((config.resource or {}).get("max_threads", 4)),
                )
                query_metadata["feature_view"] = "learner_view_with_qd3_static_codes"
                query_metadata["seconds"] = time.perf_counter() - query_started
                query_cache[cache_key] = (query_values, query_metadata)
            except Exception as error:
                return SelectionResult.failure(method_id, "failed", requested, f"OOF query loss lỗi: {error!r}")
        kwargs["query_features"], cached_metadata = query_cache[cache_key]
        kwargs["query_metadata"] = {
            **cached_metadata,
            **lrq_provenance,
            "parent_source": str(options.get("parent_source")),
        }
    try:
        result = selector.select(X_train, y_train, config.budget_ratio, **kwargs)
    except ResourceLimitError as error:
        failure = SelectionResult.failure(method_id, error.status, requested, str(error))
        failure.diagnostics.update(resource_diagnostics)
        return failure
    result.diagnostics.update(resource_diagnostics)
    if result.status == "success" and query_method == "skq_pc_lrq_sq":
        # SKQ-PC has no structure parent, so the parent-cost branch below does
        # not account for its OOF query. Declare full dependency and cache reuse;
        # an end-to-end workflow must charge this query once, not once per method.
        own_total = float(result.timings.get("total", 0.0))
        query_total = float((kwargs.get("query_metadata") or {}).get("seconds", 0.0))
        result.timings.update(skq_only=own_total, query=query_total, total=own_total + query_total)
        result.diagnostics.update(query_cache_hit=query_cache_hit, query_cost_shared=True)
    if result.status == "success" and spec.needs_parent:
        own_total = float(result.timings.get("total", 0.0))
        parent_total = float(np.asarray(structure.get("_producer_total_seconds", 0.0)).item())
        query_total = float((kwargs.get("query_metadata") or {}).get("seconds", 0.0))
        result.timings["skq_only"] = own_total
        result.timings["parent_structure"] = parent_total
        result.timings["query"] = query_total
        result.timings["total"] = own_total + parent_total + query_total
        result.diagnostics.setdefault("parent_source", str(options.get("parent_source")))
        if str(options.get("parent_source", "")).startswith(GONZALEZ_PARENT_PREFIX):
            result.diagnostics.setdefault(
                "gonzalez_candidate_multiplier",
                float(options.get("candidate_multiplier", 1.0)),
            )
    if getattr(selector, "structure_parent_ids_", None) is not None:
        structure = {
            "parent_ids": np.asarray(selector.structure_parent_ids_, dtype=np.int64),
            "row_ids": np.asarray(data["row_ids_train"], dtype=np.int64),
            "candidate_mask": np.asarray(
                getattr(selector, "candidate_mask_", None)
                if getattr(selector, "candidate_mask_", None) is not None
                else np.ones(len(y_train), dtype=bool),
                dtype=bool,
            ),
            "_producer_total_seconds": np.asarray(result.timings.get("total", 0.0), dtype=np.float64),
        }
        structure_cache[(method_id, seed)] = structure
        _save_structure(
            config, dataset_id, method_id, seed, structure, result, data_fingerprint,
        )
    return result


def _ensure_gonzalez_candidate_structure(
    *,
    config: ExperimentConfig,
    dataset_id: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    row_ids: np.ndarray,
    data_fingerprint: str,
    options: dict[str, Any],
) -> tuple[str, str] | None:
    """Tạo/lưu candidate Gonzalez dùng chung mà không thêm một utility row giả.

    Pool là artifact trung gian, không phải method được chấm điểm. Tên source và
    multiplier phải khớp tuyệt đối để protocol hash/provenance không nhập nhằng.
    Gonzalez upstream vẫn chạy nguyên hàm benchmark; SKQ chỉ dùng mask kết quả.
    """
    source = str(options.get("parent_source", ""))
    multiplier = float(options.get("candidate_multiplier", 1.0))
    if multiplier not in {1.0, 2.0}:
        return "blocked", "Gonzalez candidate_multiplier chỉ cho phép 1.0 hoặc 2.0 trong protocol v3"
    expected_source = f"gonzalez_pool_{int(multiplier)}x"
    if source != expected_source:
        return (
            "blocked",
            f"parent_source={source!r} không khớp candidate_multiplier={multiplier:g}; "
            f"cần {expected_source!r}",
        )
    try:
        requested_groups = int(options.get("gonzalez_structure_groups", 128))
    except (TypeError, ValueError):
        return "blocked", "gonzalez_structure_groups phải là số nguyên dương"
    if requested_groups < 1:
        return "blocked", "gonzalez_structure_groups phải là số nguyên dương"

    structure_path = (
        config.paths.artifact_root / "structures" / config.experiment_id /
        dataset_id / source / f"ss{seed}" / "structure.npz"
    )
    if structure_path.exists():
        metadata_path = structure_path.with_name("structure.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            diagnostics = metadata["diagnostics"]
            if (
                float(diagnostics["candidate_multiplier"]) != multiplier
                or int(diagnostics["structure_groups_requested"]) != requested_groups
            ):
                return (
                    "blocked",
                    "Gonzalez structure đã có nhưng không khớp multiplier/số anchor; "
                    "hãy dùng experiment_id mới hoặc xóa artifact trung gian cũ",
                )
        except (OSError, ValueError, TypeError, KeyError) as error:
            return "blocked", f"Gonzalez structure metadata không hợp lệ: {error!r}"
        return None

    candidate_ratio = min(1.0, float(config.budget_ratio) * multiplier)
    candidate_rows = max(1, int(round(len(y_train) * candidate_ratio)))
    try:
        estimate = estimate_selection_cost(
            "n_gcoreset_benchmark",
            len(y_train),
            X_train.shape[1],
            candidate_rows,
            n_classes=len(np.unique(y_train)),
        )
        enforce_preflight(estimate, config.resource)
        selector = build_selector(
            "n_gcoreset_benchmark",
            seed,
            external_root=config.paths.external_root,
        )
        parent_result = selector.select(X_train, y_train, candidate_ratio)
    except ResourceLimitError as error:
        return error.status, str(error)
    except Exception as error:
        return "failed", f"Không tạo được Gonzalez candidate pool: {error!r}"
    if parent_result.status != "success":
        return (
            parent_result.status,
            str(parent_result.diagnostics.get("reason", "Gonzalez candidate pool lỗi")),
        )

    center_indices = np.asarray(parent_result.indices, dtype=np.int64)
    candidate_mask = np.zeros(len(y_train), dtype=bool)
    candidate_mask[center_indices] = True
    anchor_indices = center_indices[:min(requested_groups, len(center_indices))]
    try:
        parent_ids = _nearest_gonzalez_center_ids(
            X_train, anchor_indices, ResourceGuard(config.resource)
        )
    except ResourceLimitError as error:
        return error.status, str(error)
    parent_result.diagnostics.update({
        "artifact_role": "candidate_pool_only_not_utility_row",
        "candidate_multiplier": multiplier,
        "final_budget_ratio": float(config.budget_ratio),
        "candidate_ratio": candidate_ratio,
        "structure_rule": (
            "nearest predeclared Gonzalez anchor Voronoi ID on every train row; "
            "candidate mask keeps the complete 1x/2x Gonzalez pool"
        ),
        "structure_groups_requested": requested_groups,
        "structure_groups": int(len(anchor_indices)),
    })
    structure = {
        "parent_ids": parent_ids,
        "row_ids": np.asarray(row_ids, dtype=np.int64),
        "candidate_mask": candidate_mask,
        "_producer_total_seconds": np.asarray(
            parent_result.timings.get("total", 0.0), dtype=np.float64
        ),
    }
    _save_structure(
        config, dataset_id, source, seed, structure, parent_result,
        data_fingerprint,
    )
    return None


def _nearest_gonzalez_center_ids(
    X_train: np.ndarray,
    center_indices: np.ndarray,
    guard: ResourceGuard,
    *,
    batch_rows: int = 512,
) -> np.ndarray:
    """Gán Voronoi ID theo batch để không tạo ma trận n×anchor trong RAM."""
    X = np.asarray(X_train, dtype=np.float32)
    centers = X[np.asarray(center_indices, dtype=np.int64)]
    center_norm = np.sum(centers * centers, axis=1)
    assignments = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), batch_rows):
        guard.check("gonzalez_voronoi_assignment")
        stop = min(len(X), start + batch_rows)
        block = X[start:stop]
        distances = (
            np.sum(block * block, axis=1)[:, None]
            + center_norm[None, :]
            - 2.0 * (block @ centers.T)
        )
        assignments[start:stop] = np.argmin(distances, axis=1)
    return assignments


def _validate_lrq_gate(config, options, freeze_manifest, method_id: str | None = None):
    """Cho phép dev screen có kiểm soát, nhưng không hạ gate confirmatory/test."""
    parent_source = str(options.get("parent_source", ""))
    if method_id == "skq_pc_lrq_sq":
        if config.stage_id == "s1_screen":
            if not config.test_locked:
                return "SKQ-PC + LRQ-SQ dev screen yêu cầu test_locked=true", {}
            return None, {
                "selection_phase": "pre_freeze_dev_screen",
                "parent_frozen": False,
                "confirmatory_eligible": False,
                "evaluation_split": "dev",
                "query_contract": "OOF-LR train-only",
            }
        if freeze_manifest is None:
            return "SKQ-PC + LRQ-SQ ngoài dev screen chỉ mở sau freeze", {}
        selected = (
            freeze_manifest.get("selected_candidate")
            or (freeze_manifest.get("method_roles") or {}).get("proposed_winner")
        )
        if selected != method_id:
            return "SKQ-PC + LRQ-SQ chưa phải proposed winner trong freeze", {}
        return None, {
            "selection_phase": "post_freeze_confirmatory",
            "parent_frozen": True,
            "confirmatory_eligible": True,
            "evaluation_split": "test" if config.stage_id == "s2_confirm" else "dev",
            "query_contract": "OOF-LR train-only",
        }
    if config.stage_id == "s1_screen":
        if not config.test_locked:
            return "LRQ dev screen yêu cầu test_locked=true", {}
        if not parent_source or parent_source.startswith("@"):
            return "LRQ dev screen cần parent_source cụ thể được khai báo trước", {}
        return None, {
            "selection_phase": "pre_freeze_dev_screen",
            "parent_frozen": False,
            "confirmatory_eligible": False,
            "evaluation_split": "dev",
        }
    if freeze_manifest is None:
        return "LRQ ngoài s1_screen chỉ mở sau khi freeze base winner trên dev", {}
    frozen_source = freeze_manifest.get("base_winner_structure_source")
    frozen_base = freeze_manifest.get("base_winner_method_id")
    if not frozen_source or not frozen_base:
        return "Freeze manifest thiếu base_winner_method_id/base_winner_structure_source", {}
    allowed_parent_sources = {
        str(source) for source in freeze_manifest.get(
            "allowed_parent_sources", [frozen_source]
        )
    }
    if parent_source not in allowed_parent_sources:
        return (
            f"LRQ parent_source={parent_source} không nằm trong các frozen source="
            f"{sorted(allowed_parent_sources)}",
            {},
        )
    return None, {
        "selection_phase": "post_freeze_confirmatory",
        "parent_frozen": True,
        "confirmatory_eligible": True,
        "evaluation_split": "test" if config.stage_id == "s2_confirm" else "dev",
        "frozen_base_winner_method_id": str(frozen_base),
        "frozen_parent_source": parent_source,
    }


def _resolve_structure(
    options, seed, structure_cache, config, dataset_id, expected_row_ids,
    data_fingerprint,
):
    source = options.get("parent_source")
    if not source:
        return "Thiếu method_options.parent_source; runner không tự đoán structure winner"
    structure = structure_cache.get((str(source), seed))
    if structure is None:
        path = (
            config.paths.artifact_root / "structures" / config.experiment_id /
            dataset_id / str(source) / f"ss{seed}" / "structure.npz"
        )
        if path.exists():
            metadata_path = path.with_name("structure.json")
            if not metadata_path.exists():
                return f"Structure thiếu metadata provenance: {metadata_path}"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                required_meta = {
                    "producer_method_id", "producer_repository_commit",
                    "dataset_fingerprint", "split_fingerprint", "extraction_rule", "seed",
                }
                if not required_meta <= set(metadata):
                    missing = sorted(required_meta - set(metadata))
                    return f"Structure metadata thiếu trường: {missing}"
                expected_split = sha256_file(
                    config.paths.processed_root / dataset_id / "split_manifest.json"
                )
                if metadata["producer_method_id"] != str(source) or int(metadata["seed"]) != int(seed):
                    return "Structure producer/seed không khớp request"
                if metadata["dataset_fingerprint"] != data_fingerprint:
                    return "Structure dataset fingerprint không khớp"
                if metadata["split_fingerprint"] != expected_split:
                    return "Structure split fingerprint không khớp"
                with np.load(path) as saved:
                    required_arrays = {"row_ids", "stratum_ids", "candidate_mask"}
                    if not required_arrays <= set(saved.files):
                        return f"Structure NPZ thiếu arrays: {sorted(required_arrays - set(saved.files))}"
                    row_ids = np.asarray(saved["row_ids"], dtype=np.int64)
                    if not np.array_equal(row_ids, np.asarray(expected_row_ids, dtype=np.int64)):
                        return "Structure row IDs không round-trip với train split"
                    structure = {
                        "row_ids": row_ids,
                        "parent_ids": np.asarray(saved["stratum_ids"], dtype=np.int64),
                        "candidate_mask": np.asarray(saved["candidate_mask"], dtype=bool),
                        "_producer_total_seconds": np.asarray(
                            saved["_producer_total_seconds"]
                            if "_producer_total_seconds" in saved.files else 0.0,
                            dtype=np.float64,
                        ),
                    }
            except (OSError, ValueError, TypeError, KeyError) as error:
                return f"Structure artifact không đọc được: {error!r}"
            structure_cache[(str(source), seed)] = structure
        else:
            return f"Structure source chưa chạy thành công trước method này: {source}, seed={seed}"
    return structure


def _save_structure(
    config, dataset_id, method_id, seed, structure, result, data_fingerprint,
):
    root = config.paths.artifact_root / "structures" / config.experiment_id / dataset_id / method_id / f"ss{seed}"
    root.mkdir(parents=True, exist_ok=True)
    row_ids = np.asarray(structure["row_ids"], dtype=np.int64)
    stratum_ids = np.asarray(structure["parent_ids"], dtype=np.int64)
    candidate_mask = np.asarray(structure["candidate_mask"], dtype=bool)
    if not (len(row_ids) == len(stratum_ids) == len(candidate_mask)):
        raise ValueError("Structure arrays phải cùng số dòng")
    np.savez_compressed(
        root / "structure.npz",
        row_ids=row_ids,
        stratum_ids=stratum_ids,
        candidate_mask=candidate_mask,
        _producer_total_seconds=np.asarray(
            structure.get("_producer_total_seconds", 0.0), dtype=np.float64
        ),
    )
    atomic_json(root / "structure.json", {
        "schema_version": 3,
        "producer_method_id": method_id,
        "producer_repository_commit": result.diagnostics.get(
            "upstream_commit", f"skq-exp-{__version__}"
        ),
        "dataset_id": dataset_id,
        "dataset_fingerprint": data_fingerprint,
        "split_fingerprint": sha256_file(
            config.paths.processed_root / dataset_id / "split_manifest.json"
        ),
        "extraction_rule": _structure_extraction_rule(method_id),
        "seed": seed,
        "n_rows": int(len(stratum_ids)),
        "n_groups": int(len(np.unique(stratum_ids))),
        "selection_status": result.status,
        "diagnostics": result.diagnostics,
    })


def _structure_extraction_rule(method_id: str) -> str:
    if method_id.startswith(GONZALEZ_PARENT_PREFIX):
        return (
            "Gonzalez farthest-first candidate mask at predeclared budget multiplier; "
            "parent stratum is nearest predeclared Gonzalez-anchor Voronoi ID assigned "
            "to every train row"
        )
    rules = {
        "n01_coretab_dt_subset": "CoreTab-DT apply leaf ID on every train row",
        "n02_coretab_xgb_subset": "CoreTab-XGB pred_leaf vector on every train row",
        "n_bdis_native": "BDIS candidate mask with class-labelled parent strata",
    }
    return rules.get(method_id, "registered producer structure")


def _load_freeze_manifest(
    config: ExperimentConfig, *, required: bool,
) -> dict[str, Any] | None:
    path = config.paths.artifact_root / "freeze" / "freeze_manifest.json"
    if not path.exists():
        if required:
            raise RuntimeError(f"Chưa có freeze manifest: {path}")
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Freeze manifest không đọc được: {error!r}") from error
    schema_version = manifest.get("schema_version")
    if config.evidence_role == "confirmatory":
        from .freeze import validate_freeze_manifest_v4

        validate_freeze_manifest_v4(manifest, config)
        return manifest
    if schema_version not in {3, 4}:
        raise RuntimeError("Freeze manifest historical chỉ hỗ trợ schema_version 3 hoặc 4")
    if schema_version == 4:
        from .freeze import validate_freeze_manifest_v4

        validate_freeze_manifest_v4(manifest, config)
        return manifest
    required_fields = {"config_hash", "code_commit", "dataset_fingerprints"}
    missing = required_fields - set(manifest)
    if missing:
        raise RuntimeError(f"Freeze manifest thiếu trường: {sorted(missing)}")
    if config.stage_id == "s2_confirm":
        role_map = manifest.get("method_roles")
        if not isinstance(role_map, dict):
            raise RuntimeError("Freeze manifest thiếu method_roles")
        from ..methods import METHOD_SPECS

        for role in config.frozen_method_roles:
            method_id = role_map.get(role)
            if method_id not in METHOD_SPECS or METHOD_SPECS[method_id].status == "not_runnable":
                raise RuntimeError(f"Freeze role {role} không resolve tới runnable method")
        if not isinstance(manifest.get("dataset_fingerprints"), dict):
            raise RuntimeError("Freeze dataset_fingerprints phải là object")
    return manifest


def _load_selection_artifact(config, dataset_id, method_id, seed):
    """Nạp selection có sẵn để D05 chạy riêng mà không rerun source selector."""
    canonical_root = (
        config.paths.artifact_root / "method_artifacts" / config.experiment_id /
        config.protocol_id / config.stage_id / dataset_id / method_id
    )
    canonical = sorted(canonical_root.rglob("selected_indices.npy")) if canonical_root.exists() else []
    canonical = [path for path in canonical if f"ss{seed}" in path.parts]
    if canonical:
        selected_path = canonical[0]
        artifact_dir = selected_path.parent
        try:
            stored = json.loads(
                (artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8")
            )["result"]
            return SelectionResult(
                indices=np.load(selected_path),
                weights=np.load(artifact_dir / "sample_weights.npy"),
                requested_rows=int(stored["requested_rows"]),
                realized_rows=int(stored["realized_rows"]),
                budget_mode=str(stored["budget_mode"]),
                method_id=method_id,
                diagnostics={"loaded_from": str(artifact_dir)},
                timings=stored.get("timings", {}),
            )
        except (OSError, KeyError, ValueError, TypeError):
            pass
    # Tương thích artifact cũ trước khi canonical store được đưa vào.
    root = (
        config.paths.artifact_root / "experiments" / config.experiment_id /
        config.protocol_id / config.stage_id / dataset_id / method_id
    )
    candidates = sorted(root.rglob("selected_indices.npy")) if root.exists() else []
    candidates = [path for path in candidates if f"ss{seed}" in path.parts]
    if not candidates:
        return None
    selected_path = candidates[0]
    run_dir = selected_path.parent
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        return SelectionResult(
            indices=np.load(selected_path),
            weights=np.load(run_dir / "sample_weights.npy"),
            requested_rows=int(manifest["requested_rows"]),
            realized_rows=int(manifest["realized_rows"]),
            budget_mode=str(manifest["budget_mode"]),
            method_id=method_id,
            diagnostics={"loaded_from": str(run_dir)},
            timings=manifest.get("timings", {}),
        )
    except Exception:
        return None


def _failure_for_kind(kind, method_id, status, requested, reason):
    if kind == "synthetic":
        return GeneratedDatasetResult.failure(method_id, status, requested, reason)
    return SelectionResult.failure(method_id, status, requested, reason)


def _failure_like(result, method_id, status, reason):
    return _failure_for_kind("synthetic" if isinstance(result, GeneratedDatasetResult) else "selection", method_id, status, result.requested_rows, reason)


def _save_failure_manifest(run_dir, manifest, result):
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "manifest.json", {**manifest, **result.to_dict()})


def _manifest(
    config, dataset_id, method_id, learner_id, selector_seed, eval_name,
    data_fingerprint,
):
    identity = {
        "experiment_id": config.experiment_id, "protocol_id": config.protocol_id,
        "stage_id": config.stage_id, "dataset_id": dataset_id, "method_id": method_id,
        "budget_ratio": config.budget_ratio, "selector_seed": selector_seed,
        "learner_id": learner_id, "model_seed": config.model_seed,
        "evaluation_split": eval_name, "data_fingerprint": data_fingerprint,
        "protocol_hash": protocol_hash(config),
    }
    return {**identity, "run_id": stable_hash(identity)[:20], "package_version": __version__, "python": platform.python_version(), "created_unix": time.time()}


def _save_evaluation(
    run_dir: Path,
    model,
    metrics,
    predictions,
    policy=None,
    *,
    layout: ArtifactLayout,
    experiment_id: str,
    reusable_artifact_bytes: int = 0,
) -> None:
    policy = policy or {}
    retention = {
        "model": "not_requested",
        "predictions": "not_requested",
        "probabilities": "not_requested",
        "y_true": "not_requested",
    }
    if bool(policy.get("save_models", False)):
        try:
            _enforce_storage_policy(
                policy, layout, experiment_id,
                reserve_bytes=int(policy.get("model_reserve_mb", 128) * 1024 ** 2),
            )
            joblib.dump(model, run_dir / "model.joblib", compress=3)
            retention["model"] = "saved"
        except ArtifactStorageError as error:
            retention["model"] = f"skipped_storage_guard: {error}"
    if bool(policy.get("save_predictions", False)):
        payload = {"y_pred": np.asarray(predictions["y_pred"])}
        if bool(policy.get("save_y_true", False)):
            payload["y_true"] = np.asarray(predictions["y_true"])
        if bool(policy.get("save_probabilities", False)) and predictions.get("y_prob") is not None:
            probability_dtype = str(policy.get("probability_dtype", "float32"))
            if probability_dtype not in {"float16", "float32", "float64"}:
                raise ValueError("artifact_policy.probability_dtype không hợp lệ")
            payload["y_prob"] = np.asarray(predictions["y_prob"], dtype=probability_dtype)
        for metadata_name in ("row_id", "course_id", "course_family_id"):
            if metadata_name in predictions:
                metadata = np.asarray(predictions[metadata_name])
                if len(metadata) != len(payload["y_pred"]):
                    raise ValueError(f"{metadata_name} không thẳng hàng với prediction")
                payload[metadata_name] = metadata
        estimated_bytes = int(sum(value.nbytes for value in payload.values()))
        per_run_limit = int(float(policy.get("max_prediction_mb_per_run", 64)) * 1024 ** 2)
        if estimated_bytes > per_run_limit:
            retention["predictions"] = (
                f"skipped_per_run_limit: {estimated_bytes} > {per_run_limit} bytes"
            )
        else:
            try:
                _enforce_storage_policy(
                    policy, layout, experiment_id, reserve_bytes=estimated_bytes,
                )
                np.savez_compressed(run_dir / "predictions.npz", **payload)
                retention["predictions"] = "saved_y_pred"
                retention["y_true"] = "saved" if "y_true" in payload else "reload_from_processed_split"
                retention["probabilities"] = "saved" if "y_prob" in payload else "metrics_only"
            except ArtifactStorageError as error:
                retention["predictions"] = f"skipped_storage_guard: {error}"
    with (run_dir / "metrics_per_label.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = metrics["per_label"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    np.savetxt(run_dir / "confusion_matrix.csv", np.asarray(metrics["confusion_matrix"]), delimiter=",", fmt="%d")
    cost = dict(metrics.get("cost", {}))
    evaluation_bytes = int(sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    ))
    cost["evaluation_artifact_bytes"] = evaluation_bytes
    cost["reusable_method_artifact_bytes"] = int(reusable_artifact_bytes)
    cost["artifact_bytes_total_for_replay"] = evaluation_bytes + int(reusable_artifact_bytes)
    # Tên cũ được giữ để bảng đã viết không vỡ, nhưng giá trị nay có định nghĩa rõ hơn.
    cost["artifact_bytes_lower_bound"] = cost["artifact_bytes_total_for_replay"]
    cost.setdefault("break_even_reuses", None)
    metrics["cost"] = cost
    atomic_json(run_dir / "cost.json", cost)
    atomic_json(run_dir / "retention.json", retention)
    atomic_json(run_dir / "metrics_overall.json", metrics["overall"])
    atomic_json(run_dir / "metrics_all.json", metrics)


def _method_artifact_identity(
    config: ExperimentConfig,
    dataset_id: str,
    method_id: str,
    selector_seed: int,
    data_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "experiment_id": config.experiment_id,
        "protocol_id": config.protocol_id,
        "stage_id": config.stage_id,
        "dataset_id": dataset_id,
        "method_id": method_id,
        "budget_ratio": config.budget_ratio,
        "selector_seed": selector_seed,
        "split_seed": config.split_seed,
        "method_options": (config.method_options or {}).get(method_id, {}),
        "data_fingerprint": data_fingerprint,
        "protocol_hash": protocol_hash(config),
        "package_version": __version__,
    }


def _attach_generic_artifact_diagnostics(result: Any, y_train: np.ndarray) -> None:
    n_train = int(len(y_train))
    realized = int(result.realized_rows)
    result.diagnostics.setdefault("requested_rows", int(result.requested_rows))
    result.diagnostics.setdefault("realized_rows", realized)
    result.diagnostics.setdefault("realized_fraction", realized / n_train if n_train else None)
    result.diagnostics.setdefault("compression_ratio", n_train / realized if realized else None)
    result.diagnostics.setdefault("passes_over_train", None)
    result.diagnostics.setdefault("passes_over_train_kind", "unknown")
    result.diagnostics.setdefault(
        "passes_over_train_definition",
        "Adapter chưa khai báo số lượt logic qua toàn bộ train; không được suy từ estimated operations.",
    )
    weights = np.asarray(result.weights, dtype=np.float64)
    if len(weights):
        denominator = float(len(weights) * np.sum(weights ** 2))
        result.diagnostics.setdefault(
            "normalized_ess",
            float(weights.sum() ** 2 / denominator) if denominator else None,
        )
        result.diagnostics.setdefault("zero_weight_share", float(np.mean(weights == 0)))
        result.diagnostics.setdefault("weight_sum", float(weights.sum()))
    if isinstance(result, SelectionResult):
        result.diagnostics.setdefault(
            "unique_id_ratio",
            float(len(np.unique(result.indices)) / realized) if realized else None,
        )
        selected_labels = np.unique(y_train[result.indices]) if realized else np.empty(0)
        result.diagnostics.setdefault(
            "class_coverage",
            float(len(selected_labels) / len(np.unique(y_train))) if len(y_train) else None,
        )


def _result_nbytes(result: Any) -> int:
    if isinstance(result, GeneratedDatasetResult):
        return int(result.X.nbytes + result.y.nbytes + result.weights.nbytes)
    return int(result.indices.nbytes + result.weights.nbytes)


def _enforce_storage(config: ExperimentConfig, layout: ArtifactLayout, reserve_bytes: int) -> None:
    _enforce_storage_policy(
        config.artifact_policy or {}, layout, config.experiment_id,
        reserve_bytes=reserve_bytes,
    )


def _enforce_storage_policy(
    policy: dict[str, Any],
    layout: ArtifactLayout,
    experiment_id: str,
    *,
    reserve_bytes: int,
) -> None:
    layout.enforce_storage_budget(
        experiment_id,
        max_artifact_gb=float(policy.get("max_experiment_artifact_gb", 1.0)),
        min_free_disk_gb=float(policy.get("min_free_disk_gb", 2.0)),
        reserve_bytes=reserve_bytes,
    )


def _load_completed_run(run_dir: Path, expected_manifest: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics_overall.json"
    if not manifest_path.exists() or not metrics_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != expected_manifest.get("run_id"):
            return None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            **expected_manifest,
            "status": "success",
            "macro_f1": metrics.get("f1_macro"),
            "mcc": metrics.get("mcc"),
            "resumed": True,
        }
    except (OSError, ValueError, TypeError):
        return None
