"""Đo QĐ16 train-only: 50k/200k/800k, ba lần mỗi cỡ.

Mỗi replicate lấy mẫu phân tầng phase×class chỉ từ train, rồi đo toàn bộ đường
chọn của selector. Với LRQ-SQ, thời gian gồm cả tạo OOF-LR train-only vì đây là
chi phí thực tế của selector. Script không đọc dev/test và không dùng operation
count để loại run. Khi pool train nhỏ hơn 800k, workload 800k được dựng bằng
lấy mẫu phân tầng có hoàn lại từ train và được khai báo rõ trong artifact; đây
chỉ là workload tài nguyên, không phải thêm dữ liệu hay evidence utility.

Panel Review 22/10 gồm 10 selector cố định, CoreTab-XGB OVA adapted và ba
candidate. CoreTab-XGB native vẫn giữ đúng contract binary-only: trên CQ
multiclass nó đo đường dựng cấu trúc nhưng trả ``na_contract``, không bị đổi tên
thành một baseline multiclass. Kết quả ghi observation thật, fit log--log và gate
3 giờ.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
from sklearn.model_selection import train_test_split

from skq_exp.artifacts import atomic_json
from skq_exp.experiments.resource_calibration import (
    CALIBRATION_ROWS,
    CalibrationObservation,
    DeviceMetadata,
    fit_scaling_model,
    gate_decision,
)
from skq_exp.methods import build_selector
from skq_exp.methods.proposed.query_losses import make_oof_query_losses


FIXED_METHODS = (
    "c01_uniform_random",
    "c02_stratified_random",
    "c03_phase_class_random",
    "n02_coretab_xgb_subset",
    "n_bdis_native",
    "n_gcoreset_benchmark",
    "b_moderate_ds_adapted",
    "b_ccs_oof_xgb_adapted",
    "b_dataoob_official",
    "b_kernel_thinning_classwise",
)
ADAPTED_METHODS = ("a_coretab_xgb_ova",)
CANDIDATE_METHODS = ("skq_pc", "skq_pc_lrq_sq", "skq_pc_sqrt")
METHODS = FIXED_METHODS + ADAPTED_METHODS + CANDIDATE_METHODS

# Hiện repo chỉ có wrapper shard thật cho P10/P11. Không selector nào trong
# panel QĐ16 được tự gắn nhãn shardable chỉ vì có thể chia mảng đầu vào.
SHARDABLE_METHODS: frozenset[str] = frozenset()


def _observation_payload(observations: list[CalibrationObservation]) -> list[dict[str, Any]]:
    return [
        {**item.__dict__, "device": item.device.__dict__}
        for item in observations
    ]


def _load_observations(
    path: Path,
    *,
    method_id: str,
    environment_id: str,
) -> list[CalibrationObservation]:
    """Nạp checkpoint chỉ khi method/environment khớp, không trộn máy đo."""
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Checkpoint không phải list: {path}")
    observations = []
    seen = set()
    for row in payload:
        device = DeviceMetadata(**row["device"])
        observation = CalibrationObservation(
            **{key: value for key, value in row.items() if key != "device"},
            device=device,
        )
        observation.validate()
        if observation.selector_id != method_id:
            raise ValueError(f"Checkpoint {path} chứa selector khác: {observation.selector_id}")
        if observation.device.environment_id != environment_id:
            raise ValueError(
                "Không resume calibration trên environment khác: "
                f"{observation.device.environment_id} != {environment_id}"
            )
        key = (observation.rows, observation.replicate)
        if key in seen:
            raise ValueError(f"Checkpoint trùng observation {key}: {path}")
        seen.add(key)
        observations.append(observation)
    return observations


def _gpu() -> tuple[str | None, int | None, int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not rows:
            return None, None, 0
        name, mib = [item.strip() for item in rows[0].split(",", maxsplit=1)]
        return name, int(float(mib) * 1024 ** 2), len(rows)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None, None, 0


def _process_tree_rss(process: psutil.Process) -> tuple[int, set[int]]:
    """Cộng RSS process cha và mọi process con còn sống tại thời điểm đo."""
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    rss = 0
    pids: set[int] = set()
    for item in processes:
        try:
            rss += int(item.memory_info().rss)
            pids.add(int(item.pid))
        except psutil.Error:
            continue
    return rss, pids


def _gpu_memory_for_pids(pids: set[int]) -> int | None:
    """Đọc tổng VRAM của đúng process tree; None nghĩa là không đo được."""
    if not pids:
        return 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    total_mib = 0.0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            raw_pid, raw_mib = [item.strip() for item in line.split(",", maxsplit=1)]
            if int(raw_pid) in pids:
                total_mib += float(raw_mib)
        except (ValueError, TypeError):
            continue
    return int(total_mib * 1024**2)


def _peak_resources(call: Callable[[], Any]) -> tuple[Any, float, int, int | None]:
    """Đo wall time, RSS process-tree và VRAM process-tree trong cùng run."""
    process = psutil.Process()
    peak, initial_pids = _process_tree_rss(process)
    peak_vram = _gpu_memory_for_pids(initial_pids)
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak, peak_vram
        while not stop.wait(0.2):
            rss, pids = _process_tree_rss(process)
            peak = max(peak, rss)
            measured_vram = _gpu_memory_for_pids(pids)
            if measured_vram is not None:
                peak_vram = max(peak_vram or 0, measured_vram)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    started = time.perf_counter()
    try:
        result = call()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        worker.join(timeout=1.0)
        rss, pids = _process_tree_rss(process)
        peak = max(peak, rss)
        measured_vram = _gpu_memory_for_pids(pids)
        if measured_vram is not None:
            peak_vram = max(peak_vram or 0, measured_vram)
    return result, elapsed, int(peak), peak_vram


def _headroom_payload(path: Path) -> dict[str, Any]:
    """Đọc headroom JSON trực tiếp hoặc từ archive Kaggle mà không bung test."""
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if not tarfile.is_tarfile(path):
        raise ValueError(f"Headroom evidence phải là JSON hoặc tar archive: {path}")
    with tarfile.open(path, "r:*") as archive:
        matches = [
            member for member in archive.getmembers()
            if member.isfile() and member.name.endswith("headroom_decision.json")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Archive phải có đúng một headroom_decision.json, tìm thấy {len(matches)}"
            )
        stream = archive.extractfile(matches[0])
        if stream is None:
            raise ValueError("Không đọc được headroom_decision.json trong archive")
        return json.loads(stream.read().decode("utf-8"))


def _resolve_budget_ratio(explicit: float | None, headroom: str | None) -> tuple[float, dict[str, Any]]:
    """Khóa rho từ evidence dev; explicit chỉ dùng khi không truyền headroom."""
    if headroom:
        path = Path(headroom).resolve()
        payload = _headroom_payload(path)
        ratio = float(payload["rho_star"])
        test_is_locked = (
            payload.get("test_locked") is True
            or payload.get("test_data_read") is False
        )
        if payload.get("evidence_split") != "dev" or not test_is_locked:
            raise ValueError("Headroom evidence phải là dev-only và chưa đọc test")
        source = {
            "mode": "headroom_evidence",
            "path": str(path),
            "rho_star": ratio,
            "repository_commit": payload.get("repository_commit"),
            "allowed_inputs": payload.get("allowed_inputs"),
        }
    elif explicit is not None:
        ratio = float(explicit)
        source = {"mode": "explicit_development_ratio", "rho_star": ratio}
    else:
        raise ValueError("Cần --headroom-decision hoặc --budget-ratio")
    if not 0 < ratio <= 1:
        raise ValueError(f"rho không hợp lệ: {ratio}")
    return ratio, source


def _stratified_indices(y: np.ndarray, phase: np.ndarray, rows: int, seed: int) -> np.ndarray:
    """Lấy workload phase×class; chỉ dùng replacement khi rows vượt pool train."""
    if rows == len(y):
        return np.arange(len(y), dtype=np.int64)
    strata = np.char.add(np.char.add(phase.astype(str), ":"), y.astype(str))
    all_index = np.arange(len(y), dtype=np.int64)
    if rows > len(y):
        rng = np.random.default_rng(seed)
        labels, counts = np.unique(strata, return_counts=True)
        raw = counts.astype(np.float64) * (float(rows) / float(len(y)))
        quotas = np.floor(raw).astype(np.int64)
        remainder = int(rows - int(quotas.sum()))
        if remainder:
            order = np.lexsort((labels.astype(str), -(raw - quotas)))
            quotas[order[:remainder]] += 1
        selected = []
        for label, quota in zip(labels, quotas, strict=True):
            members = all_index[strata == label]
            selected.append(rng.choice(members, size=int(quota), replace=True))
        result = np.concatenate(selected).astype(np.int64, copy=False)
        rng.shuffle(result)
        if len(result) != rows:
            raise RuntimeError(f"Sai kích thước workload replacement: {len(result)} != {rows}")
        return result
    selected, _ = train_test_split(
        all_index, train_size=rows, random_state=seed, shuffle=True, stratify=strata,
    )
    return np.sort(selected.astype(np.int64))


def _run_selector(
    method_id: str,
    X_selector: np.ndarray,
    X_learner: np.ndarray,
    y: np.ndarray,
    phase: np.ndarray,
    seed: int,
    budget: float,
    max_threads: int,
    external_root: Path,
    device: str,
):
    options = {
        "n_components": 256,
        "query_alpha": 0.5,
        "oof_folds": 5,
        "n_jobs": max_threads,
        "device": device,
        "binary_options": {"n_jobs": max_threads},
    }
    query = None
    query_meta = None
    if method_id == "skq_pc_lrq_sq":
        query, query_meta = make_oof_query_losses(
            X_learner, y, ("lr",), seed=seed, folds=5, max_threads=max_threads,
        )
    selector = build_selector(
        method_id, seed, external_root=external_root, options=options,
    )
    return selector.select(
        X_selector, y, budget,
        phases=phase,
        row_ids=np.arange(len(y), dtype=np.int64),
        query_features=query,
        query_metadata=query_meta,
    )


def _warmup_rows(method_id: str, rows: int) -> int:
    """Warmup sqrt giữ floor 20 cho 12 tầng ở rho=0,5%, budget tối thiểu 240."""
    return min(50_000 if method_id == "skq_pc_sqrt" else 10_000, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--dataset", default="course_quality_v1")
    parser.add_argument("--output", default="artifacts/cq_v1_m0_resource_calibration")
    parser.add_argument("--budget-ratio", type=float)
    parser.add_argument(
        "--headroom-decision",
        help="JSON hoặc tar.gz chứa headroom_decision.json; ưu tiên hơn ratio nhập tay.",
    )
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--external-root", default="external/repos")
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--resume", action="store_true",
        help="Tiếp tục từ observations_<method>.json trên đúng environment.",
    )
    args = parser.parse_args()
    if args.replicates != 3:
        raise ValueError("QĐ16 yêu cầu đúng ba replicate; không giảm trong evidence M0")
    budget_ratio, budget_source = _resolve_budget_ratio(
        args.budget_ratio, args.headroom_decision,
    )
    data_root = Path(args.processed_root).resolve() / args.dataset
    X_learner_all = np.load(data_root / "X_train.npy", mmap_mode="r")
    X_selector_all = np.load(data_root / "X_selector_train.npy", mmap_mode="r")
    y_all = np.load(data_root / "y_train.npy", mmap_mode="r")
    phase_all = np.load(data_root / "phase_train.npy", mmap_mode="r")
    if len(X_learner_all) != len(X_selector_all) or len(X_learner_all) < min(CALIBRATION_ROWS):
        raise ValueError(
            f"Pool learner/selector không hợp lệ: {len(X_learner_all):,}/{len(X_selector_all):,}; "
            f"cần ít nhất {min(CALIBRATION_ROWS):,}"
        )
    gpu_name, vram_bytes, gpu_count = _gpu()
    selector_device = "cuda" if gpu_count else "cpu"
    device = DeviceMetadata(
        cpu=platform.processor() or platform.machine(), gpu=gpu_name,
        ram_bytes=int(psutil.virtual_memory().total), vram_bytes=vram_bytes,
        max_threads=args.max_threads,
        environment_id=f"{platform.node()}-{platform.python_version()}-{gpu_name or 'cpu'}",
        provider="kaggle" if Path("/kaggle").exists() else "local",
        gpu_count=gpu_count,
        accelerator_policy="allocated_t4_selector_may_be_cpu_bound" if gpu_count else "cpu",
    )
    output = Path(args.output).resolve()
    external_root = Path(args.external_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    methods = tuple(args.method or METHODS)
    summary_path = output / "resource_decision_all.json"
    all_results: dict[str, Any] = {}
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            float(previous.get("budget_ratio", -1)) != budget_ratio
            or int(previous.get("target_rows", -1)) != len(X_learner_all)
        ):
            raise ValueError(
                "Không resume resource summary có rho/target_rows khác; "
                "dùng output directory mới"
            )
        all_results.update(previous.get("methods", {}))
    for method_id in methods:
        checkpoint = output / f"observations_{method_id}.json"
        observations = (
            _load_observations(
                checkpoint, method_id=method_id,
                environment_id=device.environment_id,
            )
            if args.resume else []
        )
        completed = {(item.rows, item.replicate) for item in observations}
        print(
            f"START method={method_id} resumed_observations={len(observations)}",
            flush=True,
        )
        for rows in CALIBRATION_ROWS:
            missing_replicates = [
                replicate for replicate in range(1, 4)
                if (rows, replicate) not in completed
            ]
            if not missing_replicates:
                print(f"SKIP method={method_id} rows={rows} reason=checkpoint_complete", flush=True)
                continue
            print(f"WARMUP_START method={method_id} rows={rows}", flush=True)
            warm_rows = _warmup_rows(method_id, rows)
            warm_index = _stratified_indices(np.asarray(y_all), np.asarray(phase_all), warm_rows, 9900 + rows)
            warm_result, warm_seconds, _, _ = _peak_resources(lambda: _run_selector(
                method_id,
                np.asarray(X_selector_all[warm_index]), np.asarray(X_learner_all[warm_index]),
                np.asarray(y_all[warm_index]),
                np.asarray(phase_all[warm_index]), 999, budget_ratio, args.max_threads,
                external_root, selector_device,
            ))
            print(
                f"WARMUP_DONE method={method_id} rows={rows} "
                f"status={warm_result.status} seconds={warm_seconds:.3f}",
                flush=True,
            )
            for replicate in missing_replicates:
                seed = 11 + replicate + rows
                print(
                    f"RUN_START method={method_id} rows={rows} replicate={replicate} seed={seed}",
                    flush=True,
                )
                index = _stratified_indices(np.asarray(y_all), np.asarray(phase_all), rows, seed)
                X_selector = np.asarray(X_selector_all[index])
                X_learner = np.asarray(X_learner_all[index])
                y = np.asarray(y_all[index])
                phase = np.asarray(phase_all[index])
                result, steady, peak_rss, peak_vram = _peak_resources(lambda: _run_selector(
                    method_id, X_selector, X_learner, y, phase, seed,
                    budget_ratio, args.max_threads, external_root, selector_device,
                ))
                observation = CalibrationObservation(
                    selector_id=method_id, rows=rows,
                    budget_rows=max(1, int(np.floor(rows * budget_ratio))),
                    replicate=replicate, warmup_seconds=warm_seconds,
                    steady_seconds=steady, peak_rss_bytes=peak_rss,
                    peak_vram_bytes=peak_vram, device=device,
                    status="success" if result.status == "success" else result.status,
                )
                observations.append(observation)
                observations.sort(key=lambda item: (item.rows, item.replicate))
                atomic_json(checkpoint, _observation_payload(observations))
                atomic_json(
                    output / f"details_{method_id}_{rows}_{replicate}.json",
                    {
                        "method_id": method_id,
                        "rows": rows,
                        "replicate": replicate,
                        "seed": seed,
                        "warmup_rows": warm_rows,
                        "source_train_rows": int(len(y_all)),
                        "sampling_mode": (
                            "stratified_with_replacement"
                            if rows > len(y_all)
                            else "stratified_without_replacement"
                        ),
                        "synthetic_workload_only": bool(rows > len(y_all)),
                        "status": result.status,
                        "requested_rows": result.requested_rows,
                        "realized_rows": result.realized_rows,
                        "budget_mode": result.budget_mode,
                        "diagnostics": result.diagnostics,
                        "timings": result.timings,
                        "steady_seconds": steady,
                        "peak_rss_bytes": peak_rss,
                        "peak_vram_bytes": peak_vram,
                    },
                )
                print(
                    f"RUN_DONE method={method_id} rows={rows} replicate={replicate} "
                    f"status={result.status} seconds={steady:.3f} peak_rss={peak_rss}",
                    flush=True,
                )
        fitted = fit_scaling_model(tuple(observations), target_rows=len(X_learner_all))
        decision = gate_decision(
            fitted, supports_sharding=method_id in SHARDABLE_METHODS,
        )
        all_results[method_id] = {"fit": fitted, "decision": decision}
        atomic_json(output / f"resource_fit_{method_id}.json", all_results[method_id])
        print(
            f"METHOD_DONE method={method_id} decision={decision['decision']} "
            f"upper_seconds={decision.get('upper_prediction_seconds')}",
            flush=True,
        )
    atomic_json(summary_path, {
        "schema_version": 1,
        "test_data_read": False,
        "calibration_rows": list(CALIBRATION_ROWS),
        "replicates": 3,
        "budget_ratio": budget_ratio,
        "budget_source": budget_source,
        "target_rows": int(len(X_learner_all)),
        "source_train_rows": int(len(X_learner_all)),
        "calibration_sampling_policy": {
            "strata": "phase_x_class",
            "without_replacement_up_to_source_rows": True,
            "with_replacement_above_source_rows": True,
            "purpose": "resource_workload_only_not_utility_evidence",
        },
        "selector_feature_count": int(X_selector_all.shape[1]),
        "learner_feature_count": int(X_learner_all.shape[1]),
        "fixed_methods": list(FIXED_METHODS),
        "adapted_methods": list(ADAPTED_METHODS),
        "candidate_methods": list(CANDIDATE_METHODS),
        "expected_method_count": len(METHODS),
        "measured_method_count": len(all_results),
        "panel_complete": set(all_results) == set(METHODS),
        "external_root": str(external_root),
        "selector_device": selector_device,
        "methods": all_results,
    })
    print(json.dumps({method: value["decision"] for method, value in all_results.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
