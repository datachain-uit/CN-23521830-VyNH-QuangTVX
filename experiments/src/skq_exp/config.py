"""File này đọc và kiểm tra cấu hình trước khi một run bắt đầu.

Nó ngăn các lỗi dễ bỏ sót như budget ngoài khoảng cho phép, stage không hợp lệ
hoặc confirmatory run chưa có freeze manifest. Đầu ra là một đối tượng cấu hình
ổn định để các module khác không phải tự đọc JSON theo nhiều cách khác nhau.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_STAGES = {
    "s0_smoke", "s1_screen", "s2_confirm", "s2_transfer",
    "s3_scale", "s4_temporal", "s5_native",
}


@dataclass(frozen=True)
class ProjectPaths:
    raw_root: Path
    processed_root: Path
    artifact_root: Path
    external_root: Path


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    protocol_id: str
    stage_id: str
    dataset_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    learner_ids: tuple[str, ...]
    budget_ratio: float
    split_seed: int
    selector_seeds: tuple[int, ...]
    model_seed: int
    paths: ProjectPaths
    evidence_role: str = "development"
    split_id: str = "registered_split_v1"
    frozen_method_roles: tuple[str, ...] = ()
    test_locked: bool = True
    requires_freeze_manifest: bool = False
    non_inferiority_margin: float = 0.005
    holm_contrasts: int = 5
    resource: dict[str, Any] | None = None
    artifact_policy: dict[str, Any] | None = None
    method_options: dict[str, dict[str, Any]] | None = None
    class_weight_policy: str = "balanced"
    learner_params: dict[str, dict[str, Any]] | None = None
    early_stopping: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if "extends" in payload:
            parent_path = (config_path.parent / payload.pop("extends")).resolve()
            if parent_path == config_path:
                raise ValueError("Config không được extends chính nó")
            parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
            if "extends" in parent_payload:
                raise ValueError("Chỉ hỗ trợ một tầng extends để config dễ kiểm toán")
            # method_options là bảng theo method nên config con được phép thêm
            # method mới mà không phải chép lại toàn bộ protocol cha. Các khóa
            # của cùng một method vẫn được thay nguyên khối để provenance rõ ràng.
            if "method_options" in payload:
                payload["method_options"] = {
                    **parent_payload.get("method_options", {}),
                    **payload["method_options"],
                }
            payload = {**parent_payload, **payload}
        base = config_path.parent.parent
        raw_paths = payload.pop("paths")
        raw_paths.setdefault("external_root", "external/repos")
        paths = ProjectPaths(**{
            key: _resolve_path(base, value) for key, value in raw_paths.items()
        })
        config = cls(
            dataset_ids=tuple(payload.pop("dataset_ids")),
            method_ids=tuple(payload.pop("method_ids")),
            learner_ids=tuple(payload.pop("learner_ids")),
            selector_seeds=tuple(int(v) for v in payload.pop("selector_seeds")),
            frozen_method_roles=tuple(payload.pop("frozen_method_roles", ())),
            paths=paths,
            **payload,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 3:
            raise ValueError("schema_version phải bằng 3")
        if self.stage_id not in VALID_STAGES:
            raise ValueError(f"stage_id không hợp lệ: {self.stage_id}")
        if not 0 < self.budget_ratio <= 1:
            raise ValueError("budget_ratio phải nằm trong (0, 1]")
        if not self.dataset_ids or not self.method_ids or not self.learner_ids:
            raise ValueError("dataset_ids, method_ids và learner_ids không được rỗng")
        from .data.registry import DATASET_SPECS
        from .methods.registry import METHOD_SPECS

        unknown_datasets = set(self.dataset_ids) - set(DATASET_SPECS)
        if self.stage_id == "s0_smoke":
            unknown_datasets -= {"toy_multiclass_v1"}
        if unknown_datasets:
            raise ValueError(f"Dataset chưa đăng ký: {sorted(unknown_datasets)}")
        unknown_methods = set(self.method_ids) - set(METHOD_SPECS)
        if unknown_methods:
            raise ValueError(f"Method chưa đăng ký: {sorted(unknown_methods)}")
        blocked_status = {
            method_id: METHOD_SPECS[method_id].status
            for method_id in self.method_ids
            if METHOD_SPECS[method_id].status
            not in {"implemented", "derived", "evaluator_only"}
        }
        if blocked_status:
            raise ValueError(f"Config chứa method không runnable: {blocked_status}")
        valid_learners = {"lr", "rf", "xgb", "cat", "mlp"}
        if set(self.learner_ids) - valid_learners:
            raise ValueError(f"Learner chưa đăng ký: {sorted(set(self.learner_ids) - valid_learners)}")
        if self.class_weight_policy not in {"natural", "balanced"}:
            raise ValueError("class_weight_policy chỉ nhận natural hoặc balanced")
        unknown_param_learners = set(self.learner_params or {}) - valid_learners
        if unknown_param_learners:
            raise ValueError(
                f"learner_params chứa learner chưa đăng ký: {sorted(unknown_param_learners)}"
            )
        early_stopping = self.early_stopping or {}
        if early_stopping:
            if not isinstance(early_stopping.get("enabled", False), bool):
                raise ValueError("early_stopping.enabled phải là bool")
            if int(early_stopping.get("patience", 20)) <= 0:
                raise ValueError("early_stopping.patience phải dương")
            if early_stopping.get("enabled", False) and early_stopping.get("source") != "dev":
                raise ValueError("early stopping chỉ được phép dùng source=dev")
        options = self.method_options or {}
        pending_candidates = [
            method_id for method_id in self.method_ids
            if METHOD_SPECS[method_id].role == "experimental_candidate"
        ]
        if pending_candidates:
            if self.evidence_role != "dev_selection" or not self.test_locked:
                raise ValueError(
                    "G00: experimental candidate chỉ được chạy dev_selection khi test_locked=true"
                )
            missing_ack = [
                method_id for method_id in pending_candidates
                if not bool(options.get(method_id, {}).get(
                    "pending_approval_acknowledged", False
                ))
            ]
            if missing_ack:
                raise ValueError(
                    "G00: experimental candidate cần xác nhận pending_approval_acknowledged: "
                    f"{missing_ack}"
                )
        invalid_contracts = {}
        for method_id, method_config in options.items():
            if "allowed_learners" not in method_config:
                continue
            allowed = tuple(method_config["allowed_learners"])
            if not allowed or set(allowed) - valid_learners:
                invalid_contracts[method_id] = list(allowed)
        if invalid_contracts:
            raise ValueError(f"method_options.allowed_learners không hợp lệ: {invalid_contracts}")
        missing_parent = [
            method_id for method_id in self.method_ids
            if METHOD_SPECS[method_id].needs_parent
            and not options.get(method_id, {}).get("parent_source")
        ]
        if missing_parent:
            raise ValueError(f"Method cần method_options.parent_source: {missing_parent}")
        if not self.selector_seeds or len(set(self.selector_seeds)) != len(self.selector_seeds):
            raise ValueError("selector_seeds phải có ít nhất một giá trị và không trùng")
        lrq_methods = {
            "p04_skq_lrq_sq", "p05_skq_lrq_mq",
            "p08_skq_gonzalez_lrq_sq", "p09_skq_gonzalez_lrq_mq",
        } & set(self.method_ids)
        if self.stage_id == "s1_screen" and lrq_methods:
            if not self.test_locked:
                raise ValueError("LRQ s1_screen chỉ được chạy khi test_locked=true")
            unresolved = [
                method_id for method_id in sorted(lrq_methods)
                if str(options.get(method_id, {}).get("parent_source", "")).startswith("@")
            ]
            if unresolved:
                raise ValueError(
                    "LRQ s1_screen cần parent_source cụ thể khai báo trước: "
                    f"{unresolved}"
                )
        if self.stage_id == "s4_temporal" and self.test_locked:
            raise ValueError("s4_temporal cần test_locked=false để đọc temporal test")
        if self.stage_id == "s2_confirm":
            if self.budget_ratio != 0.05 or tuple(self.learner_ids) != ("xgb",):
                raise ValueError("confirmatory contract yêu cầu budget 5% và chỉ XGBoost")
            if not self.requires_freeze_manifest:
                raise ValueError("confirmatory run phải yêu cầu freeze manifest")
            if self.test_locked:
                raise ValueError("s2_confirm chỉ được mở test sau freeze (test_locked=false)")
            if set(self.frozen_method_roles) != {"published_reference", "proposed_winner"}:
                raise ValueError(
                    "confirmatory config phải resolve published_reference và proposed_winner từ freeze"
                )
            invalid_panel = {
                method_id: {
                    "status": METHOD_SPECS[method_id].status,
                    "role": METHOD_SPECS[method_id].role,
                    "evidence_role": METHOD_SPECS[method_id].evidence_role,
                }
                for method_id in self.method_ids
                if (
                    METHOD_SPECS[method_id].output_kind == "synthetic"
                    or not METHOD_SPECS[method_id].confirmatory_allowed
                    or METHOD_SPECS[method_id].evidence_role not in {"confirmatory", "appendix"}
                )
            }
            if invalid_panel:
                raise ValueError(
                    "G00: confirmatory config chứa method retired/synthetic/chưa được duyệt: "
                    f"{invalid_panel}"
                )
        if self.evidence_role == "dev_selection":
            candidates = [
                method_id for method_id in self.method_ids
                if METHOD_SPECS[method_id].role == "proposed"
            ]
            if len(candidates) > 3:
                raise ValueError("G00: dev_selection chỉ được có tối đa ba candidate proposed")
            unapproved = [
                method_id for method_id in candidates
                if not METHOD_SPECS[method_id].confirmatory_allowed
            ]
            if unapproved:
                raise ValueError(
                    "G00: candidate dev_selection chưa được review/registry duyệt: "
                    f"{unapproved}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "protocol_id": self.protocol_id,
            "stage_id": self.stage_id,
            "dataset_ids": list(self.dataset_ids),
            "method_ids": list(self.method_ids),
            "learner_ids": list(self.learner_ids),
            "budget_ratio": self.budget_ratio,
            "split_seed": self.split_seed,
            "selector_seeds": list(self.selector_seeds),
            "model_seed": self.model_seed,
            "evidence_role": self.evidence_role,
            "split_id": self.split_id,
            "frozen_method_roles": list(self.frozen_method_roles),
            "test_locked": self.test_locked,
            "requires_freeze_manifest": self.requires_freeze_manifest,
            "non_inferiority_margin": self.non_inferiority_margin,
            "holm_contrasts": self.holm_contrasts,
            "resource": self.resource or {},
            "artifact_policy": self.artifact_policy or {},
            "method_options": self.method_options or {},
            "class_weight_policy": self.class_weight_policy,
            "learner_params": self.learner_params or {},
            "early_stopping": self.early_stopping or {},
            "paths": {
                "raw_root": str(self.paths.raw_root),
                "processed_root": str(self.paths.processed_root),
                "artifact_root": str(self.paths.artifact_root),
                "external_root": str(self.paths.external_root),
            },
        }


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()
