"""File này là danh mục chuẩn của toàn bộ panel thực nghiệm.

Registry phân biệt evaluator, selector dòng thật, generator synthetic, proposed và
ablation. `implemented` nghĩa là đã có đường chạy thật; dependency/resource gate
vẫn có thể trả BLOCKED/OOM trong run chứ không được đổi sang thuật toán khác.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import GonzalezBenchmarkSelector, LeverageBenchmarkSelector
from .adapted import (
    CCSOOFXGBSelector,
    CoreTabXGBOVASelector,
    DataOOBOfficialSelector,
    KernelThinningClasswiseSelector,
    ModerateDSTabularSelector,
)
from .controls import (
    PhaseClassStratifiedRandomSelector,
    StratifiedRandomSelector,
    UniformRandomSelector,
)
from .diagnostics import build_d02, build_d04
from .native import AutoCoresetNativeSelector, BDISNativeSelector, CoreTabNativeSelector, CRAIGNativeSelector
from .proposed.structured_kquad import StructuredKQuadSelector
from .proposed.skq_pc import SKQPCSelector
from .proposed.sharded_merge_reduce import ShardedMergeReduceSelector
from .synthetic import TAMEOfficialGenerator, TDBenchGenerator


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    role: str
    status: str
    needs_parent: bool = False
    output_kind: str = "selection"
    allowed_learners: tuple[str, ...] | None = None
    note: str = ""
    fidelity_tier: str = "unspecified_legacy"
    task_contract: str = "task_agnostic"
    budget_contract: str = "exact"
    evidence_role: str = "legacy"
    confirmatory_allowed: bool = True
    max_dev_configs: int = 3

    def __post_init__(self) -> None:
        valid_fidelity = {
            "unspecified_legacy", "official_native_path", "official_source_adapter",
            "official_source_with_declared_patches", "benchmark_source_adapter",
            "local_reimplementation", "proposed",
        }
        valid_tasks = {"task_agnostic", "binary_only", "multiclass", "regression"}
        valid_budgets = {"exact", "native_realized", "at_most", "synthetic_exact", "not_applicable"}
        valid_evidence = {
            "legacy", "exploratory", "dev_selection", "confirmatory",
            "appendix", "historical",
        }
        if self.fidelity_tier not in valid_fidelity:
            raise ValueError(f"fidelity_tier không hợp lệ: {self.fidelity_tier}")
        if self.task_contract not in valid_tasks:
            raise ValueError(f"task_contract không hợp lệ: {self.task_contract}")
        if self.budget_contract not in valid_budgets:
            raise ValueError(f"budget_contract không hợp lệ: {self.budget_contract}")
        if self.evidence_role not in valid_evidence:
            raise ValueError(f"evidence_role không hợp lệ: {self.evidence_role}")
        if self.max_dev_configs < 0 or self.max_dev_configs > 3:
            raise ValueError("max_dev_configs phải nằm trong [0, 3] theo review")


METHOD_SPECS = {
    spec.method_id: spec for spec in [
        MethodSpec(
            "c00_full_train", "FullTrain", "reference", "evaluator_only",
            note="Mốc trần dùng toàn bộ train; không phải phương pháp nén.",
        ),
        MethodSpec(
            "c00_full_train_a", "FullTrain-A lambda=1", "reference", "evaluator_only",
            note=(
                "QĐ22: snapshot lambda=1 của đúng cohort train V4; học một mô hình "
                "và đánh giá trên cùng dev P1-P4, không thuộc họ K."
            ),
            evidence_role="confirmatory",
        ),
        MethodSpec(
            "c01_uniform_random", "Uniform Random", "control", "implemented",
            fidelity_tier="local_reimplementation", task_contract="task_agnostic",
            budget_contract="exact", evidence_role="confirmatory",
            note="C01: exact-budget uniform sampling, không hoàn lại.",
        ),
        MethodSpec(
            "c02_stratified_random", "Stratified Random", "control", "implemented",
            allowed_learners=("xgb",),
            fidelity_tier="local_reimplementation", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
            note=(
                "Control cổ điển: subset thật, exact budget. Contract cũ chỉ chạy XGBoost; "
                "config full-v2 ghi đè rõ ràng để đánh giá đủ năm learner."
            ),
        ),
        MethodSpec(
            "c03_phase_class_random", "Phase×Class Stratified Random", "control", "implemented",
            fidelity_tier="local_reimplementation", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
            note="C03: exact budget theo phase×class, w=|Pg|/mg bảo toàn mass.",
        ),
        MethodSpec(
            "n01_coretab_dt_subset", "CoreTab-DT Official", "native", "implemented",
            note="Official-source adapter; subset thật theo kích thước native CoreTab-DT.",
            fidelity_tier="official_source_adapter", task_contract="binary_only",
            budget_contract="native_realized", evidence_role="historical",
        ),
        MethodSpec(
            "n02_coretab_xgb_subset", "CoreTab-XGB Official", "native", "implemented",
            note="Official-source adapter; subset thật theo kích thước native CoreTab-XGB.",
            fidelity_tier="official_source_adapter", task_contract="binary_only",
            budget_contract="native_realized", evidence_role="historical",
        ),
        MethodSpec(
            "a_coretab_xgb_ova", "CoreTab-XGB (OVA, adapted)", "adaptation", "implemented",
            fidelity_tier="official_source_adapter", task_contract="multiclass",
            budget_contract="at_most", evidence_role="confirmatory",
            note=(
                "QĐ14: mỗi lớp gọi CoreTab-XGB binary upstream, quota largest remainder; "
                "không pad, thiếu positive giữ realized size. Không phải native multiclass."
            ),
        ),
        MethodSpec(
            "b_moderate_ds_adapted", "Moderate-DS (tabular, adapted)",
            "baseline", "implemented",
            fidelity_tier="official_source_adapter", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
            note=(
                "Review 12 §7.6: khoảng cách tới median lớp trên selector_view chuẩn hóa; "
                "giữ cửa sổ score trung tâm exact-budget. Adapted vì source gốc dùng embedding ảnh."
            ),
        ),
        MethodSpec(
            "b_ccs_oof_xgb_adapted", "CCS (OOF-XGBoost EL2N, adapted)",
            "baseline", "implemented", allowed_learners=("xgb", "lr"),
            fidelity_tier="official_source_adapter", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
            note=(
                "Review 12 §7.6: stratified sampling từ source CCS; EL2N lấy từ xác suất "
                "OOF-XGBoost nên bắt buộc gắn nhãn adapted."
            ),
        ),
        MethodSpec(
            "b_dataoob_official", "Data-OOB official", "baseline", "implemented",
            fidelity_tier="official_source_with_declared_patches",
            task_contract="multiclass", budget_contract="exact",
            evidence_role="confirmatory",
            note=(
                "Review 12 §7.6: Data-OOB native cho dữ liệu bảng; gọi RandomForestDV upstream, "
                "giữ các dòng có OOB value cao nhất; vá import cho sklearn mới và khai báo "
                "top-k value→subset là adaptation của protocol."
            ),
        ),
        MethodSpec(
            "b_kernel_thinning_classwise", "Kernel Thinning (classwise, adapted)",
            "baseline", "implemented",
            fidelity_tier="official_source_adapter", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
            note=(
                "Review 12 §7.6: GoodPoints Kernel Thinning theo từng lớp; exact-budget "
                "pre-pool được khai báo rõ, không gọi là native paper reproduction."
            ),
        ),
        MethodSpec(
            "n_bdis_native", "BDIS-native", "native", "implemented",
            note="Official-source adapter; subset thật, cần faiss và giữ realized size upstream.",
            fidelity_tier="official_source_adapter", task_contract="task_agnostic",
            budget_contract="native_realized", evidence_role="confirmatory",
        ),
        MethodSpec(
            "n_autocoreset_native", "AutoCoreset-native", "native", "implemented",
            note="Official-source adapter qua driver riêng; subset thật có trọng số và artifact gate.",
        ),
        MethodSpec(
            "n_craig_native", "CRAIG feature-space (retired native label)",
            "native", "retired", note="Không tương đương full official logistic pipeline",
        ),
        MethodSpec(
            "a_craig_feature_space", "CRAIG feature-space adapter",
            "adaptation", "implemented",
            note="R1 feature-space adaptation; dùng upstream lazy-greedy, không claim native equivalence.",
        ),
        MethodSpec(
            "n_gcoreset_benchmark", "Gonzalez Coreset benchmark", "benchmark", "implemented",
            fidelity_tier="benchmark_source_adapter", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
            note="Benchmark-source adapter; farthest-first subset thật, không phải native paper source.",
        ),
        MethodSpec(
            "n_leverage_benchmark", "Leverage benchmark", "benchmark", "implemented",
            note="Benchmark-source adapter; PCA leverage subset thật, không phải native paper source.",
        ),
        MethodSpec(
            "s_kip_tdbench", "KIP-TDBench", "synthetic", "implemented",
            output_kind="synthetic",
            note="TDBench-source adapter; sinh dữ liệu bằng kernel inducing points, có dependency/resource gate.",
        ),
        MethodSpec(
            "s_mtt_tdbench", "MTT-TDBench", "synthetic", "implemented",
            output_kind="synthetic",
            note=(
                "Benchmark-source adapter có 7 patch khai báo; khôi phục gradient, "
                "snapshot và RNG cho trajectory matching; không phải TDBench nguyên trạng."
            ),
        ),
        MethodSpec(
            "s_gm_tdbench", "Gradient Matching-TDBench", "synthetic", "implemented",
            output_kind="synthetic",
            note=(
                "Benchmark-source adapter cho Dataset Condensation with Gradient Matching "
                "(ICLR 2021); vá requires_grad bị thiếu trong source TDBench và khai báo patch."
            ),
        ),
        MethodSpec(
            "s_datm_tdbench", "DATM-TDBench", "synthetic", "implemented",
            output_kind="synthetic",
            note=(
                "Benchmark-source adapter cho Difficulty-Aligned Trajectory Matching "
                "(ICLR 2024); dùng source TDBench với các patch gradient/snapshot/RNG khai báo."
            ),
        ),
        MethodSpec(
            "s_tame_official", "TAME official", "synthetic", "implemented",
            output_kind="synthetic",
            note="Official-source synthesis adapter; distribution matching, không claim full-paper reproduction.",
        ),
        MethodSpec(
            "p00_structured_kquad", "Structured KQuad engine", "engine", "not_runnable",
            note="Engine nội bộ của SKQ; không phải một dòng độc lập trong bảng utility.",
        ),
        MethodSpec(
            "p01_skq_coretab_dt", "SKQ-CoreTab-DT", "proposed", "implemented", True,
            note="Đề xuất: structure CoreTab-DT + RFF herding + simplex-QP.",
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p02_skq_coretab_xgb", "SKQ-CoreTab-XGB", "proposed", "implemented", True,
            note="Đề xuất: structure CoreTab-XGB + RFF herding + simplex-QP.",
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p03_skq_bdis_filtered", "SKQ-BDIS-Filtered", "proposed", "implemented", True,
            note=(
                "Đề xuất: candidate BDIS + phân bổ có cấu trúc + RFF herding + "
                "simplex-QP; realized=min(mục tiêu 5%, BDIS pool) và báo cáo size thực."
            ),
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p04_skq_lrq_sq", "SKQ-LRQ-SQ", "proposed", "implemented", True,
            note=(
                "Đề xuất query-aware một learner; được screen trên dev với parent "
                "khai báo trước, test chỉ mở sau freeze."
            ),
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p05_skq_lrq_mq", "SKQ-LRQ-MQ", "proposed", "implemented", True,
            note=(
                "Đề xuất query-aware nhiều learner; được screen trên dev với parent "
                "khai báo trước, test chỉ mở sau freeze."
            ),
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p06_gonzalez_qp", "Gonzalez-QP", "proposed", "implemented", True,
            note=(
                "Đề xuất screening: giữ candidate Gonzalez đúng 5%, dùng Voronoi theo "
                "Gonzalez anchor và simplex-QP; tách riêng để đo đóng góp của weighting."
            ),
            fidelity_tier="proposed", evidence_role="appendix", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p07_skq_gonzalez", "SKQ-Gonzalez", "proposed", "implemented", True,
            note=(
                "Đề xuất coverage-aware: Gonzalez tạo candidate pool 2x budget, "
                "SKQ RFF herding + simplex-QP nén về exact budget."
            ),
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p08_skq_gonzalez_lrq_sq", "SKQ-Gonzalez-LRQ-SQ", "proposed", "implemented", True,
            note=(
                "Đề xuất coverage/query-aware một learner: candidate Gonzalez 2x, "
                "OOF-LR query loss, RFF herding và simplex-QP; dev screen trước freeze."
            ),
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p09_skq_gonzalez_lrq_mq", "SKQ-Gonzalez-LRQ-MQ", "proposed", "implemented", True,
            note=(
                "Đề xuất coverage/query-aware nhiều learner: candidate Gonzalez 2x, "
                "OOF LR/RF/XGB query loss, RFF herding và simplex-QP; dev screen trước freeze."
            ),
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p10_skq_mr_coretab_xgb", "SKQ-MR-CoreTab-XGB", "proposed", "implemented",
            note=(
                "Đề xuất scale-out: CoreTab-XGB tạo leaf structure theo shard xác định, "
                "sau đó SKQ streaming cấp exact budget toàn cục; dùng toàn bộ train, không proxy."
            ),
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "p11_skq_mr_bdis", "SKQ-MR-BDIS", "proposed", "implemented",
            note=(
                "Đề xuất scale-out: BDIS tạo candidate theo shard, SKQ streaming reduce toàn cục; "
                "giữ realized size khi candidate pool nhỏ, không pad/trùng và không proxy."
            ),
            fidelity_tier="proposed", evidence_role="historical", confirmatory_allowed=False,
        ),
        MethodSpec(
            "skq_pc", "SKQ-PC", "proposed", "implemented",
            note=(
                "Candidate A được review xác nhận: phase×class, Gonzalez/Voronoi "
                "trong tầng, RFF herding và simplex-QP bảo toàn mass tầng."
            ),
            fidelity_tier="proposed", task_contract="task_agnostic",
            budget_contract="exact", evidence_role="dev_selection",
            confirmatory_allowed=True, max_dev_configs=1,
        ),
        MethodSpec(
            "skq_pc_lrq_sq", "SKQ-PC + LRQ-SQ", "proposed", "implemented",
            note=(
                "Candidate B được review xác nhận: SKQ-PC cộng OOF Logistic "
                "Regression query loss train-only, alpha khóa ở 0.5."
            ),
            fidelity_tier="proposed", task_contract="task_agnostic",
            budget_contract="exact", evidence_role="dev_selection",
            confirmatory_allowed=True, max_dev_configs=1,
        ),
        MethodSpec(
            "skq_pc_r", "SKQ-PC-R (A3)", "ablation", "implemented",
            note=(
                "Candidate C/ablation: giữ phase×class, Gonzalez cell và QP, "
                "nhưng thay kernel herding bằng random không hoàn lại trong cell."
            ),
            fidelity_tier="proposed", task_contract="task_agnostic",
            budget_contract="exact", evidence_role="appendix",
            confirmatory_allowed=True, max_dev_configs=1,
        ),
        MethodSpec(
            "skq_pc_sqrt", "SKQ-PC-sqrt", "proposed", "implemented",
            note=(
                "Candidate C theo QĐ20 Review 22/10/2026: giữ cell, herding và QP "
                "của SKQ-PC nhưng cấp quota phase×class theo căn bậc hai, floor 20."
            ),
            fidelity_tier="proposed", task_contract="task_agnostic",
            budget_contract="exact", evidence_role="dev_selection",
            confirmatory_allowed=True, max_dev_configs=1,
        ),
        MethodSpec(
            "x_coretab_mc_cell",
            "SKQ-PC/CoreTab-cell (pending approval)",
            "experimental_candidate",
            "implemented",
            True,
            note=(
                "Suy luận triển khai của nhóm: thay Gonzalez cell bằng leaf signature "
                "CoreTab-XGB, còn phase×class/herding/QP giữ nguyên; chạy dev-only, "
                "không phải CoreTab native và không được freeze trước phê duyệt."
            ),
            fidelity_tier="proposed", task_contract="task_agnostic",
            budget_contract="exact", evidence_role="exploratory",
            confirmatory_allowed=False, max_dev_configs=1,
        ),
        MethodSpec(
            "d02_parent_structured_random", "Parent-Structured Random", "ablation", "implemented", True,
            note="Ablation: giữ structure và QP nhưng thay kernel herding bằng random.",
            fidelity_tier="local_reimplementation", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
        ),
        MethodSpec(
            "d04_global_rff_quadrature", "Global RFF Quadrature", "ablation", "implemented",
            note="Ablation: bỏ parent structure, giữ class + RFF herding + QP.",
            fidelity_tier="local_reimplementation", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
        ),
        MethodSpec(
            "d05_equal_group_weight", "Equal-Within-Group Weight", "ablation", "derived", True,
            note="Ablation: giữ index winner, thay simplex-QP bằng trọng số đều trong nhóm.",
            fidelity_tier="local_reimplementation", task_contract="multiclass",
            budget_contract="exact", evidence_role="confirmatory",
        ),
    ]
}


def get_method_spec(method_id: str) -> MethodSpec:
    try:
        return METHOD_SPECS[method_id]
    except KeyError as error:
        raise KeyError(f"Method chưa đăng ký: {method_id}") from error


def allowed_learners_for(
    method_id: str, options: dict[str, Any] | None = None,
) -> tuple[str, ...] | None:
    """Trả contract learner, cho phép protocol mới ghi đè contract legacy.

    Override nằm trong config và vì vậy đi vào protocol hash/artifact. Cách này
    giữ nguyên hành vi của config v1 nhưng cho phép full-v2 mở control ngẫu nhiên
    trên đủ learner mà không sửa thuật toán chọn mẫu.
    """
    configured = dict(options or {}).get("allowed_learners")
    if configured is not None:
        return tuple(str(item) for item in configured)
    return get_method_spec(method_id).allowed_learners


def build_selector(method_id: str, seed: int, *, n_components: int = 256, external_root: str | Path | None = None, options: dict[str, Any] | None = None):
    spec = get_method_spec(method_id)
    if spec.status != "implemented":
        raise RuntimeError(f"{method_id} không phải selector trực tiếp: status={spec.status}")
    options = dict(options or {})
    repo_root = Path(external_root or "external/repos")
    if method_id == "c02_stratified_random":
        return StratifiedRandomSelector(seed)
    if method_id == "c01_uniform_random":
        return UniformRandomSelector(seed)
    if method_id == "c03_phase_class_random":
        return PhaseClassStratifiedRandomSelector(
            seed,
            allocation_policy=str(options.get("allocation_policy", "proportional")),
            allocation_floor=int(options.get("allocation_floor", 20)),
        )
    if method_id in {"n01_coretab_dt_subset", "n02_coretab_xgb_subset"}:
        return CoreTabNativeSelector(method_id, repo_root, seed, options)
    if method_id == "a_coretab_xgb_ova":
        return CoreTabXGBOVASelector(repo_root, seed, options)
    if method_id == "b_moderate_ds_adapted":
        return ModerateDSTabularSelector(repo_root, seed, options)
    if method_id == "b_ccs_oof_xgb_adapted":
        return CCSOOFXGBSelector(repo_root, seed, options)
    if method_id == "b_dataoob_official":
        return DataOOBOfficialSelector(repo_root, seed, options)
    if method_id == "b_kernel_thinning_classwise":
        return KernelThinningClasswiseSelector(repo_root, seed, options)
    if method_id == "n_bdis_native":
        return BDISNativeSelector(repo_root, seed, options)
    if method_id == "n_autocoreset_native":
        return AutoCoresetNativeSelector(repo_root, seed, options)
    if method_id == "a_craig_feature_space":
        return CRAIGNativeSelector(repo_root, seed, options)
    if method_id == "n_gcoreset_benchmark":
        return GonzalezBenchmarkSelector(repo_root, seed)
    if method_id == "n_leverage_benchmark":
        return LeverageBenchmarkSelector(repo_root, seed)
    if method_id == "d02_parent_structured_random":
        return build_d02(seed, n_components)
    if method_id == "d04_global_rff_quadrature":
        return build_d04(seed, n_components)
    if method_id in {
        "p01_skq_coretab_dt", "p02_skq_coretab_xgb", "p03_skq_bdis_filtered",
        "p04_skq_lrq_sq", "p05_skq_lrq_mq", "p06_gonzalez_qp",
        "p07_skq_gonzalez", "p08_skq_gonzalez_lrq_sq",
        "p09_skq_gonzalez_lrq_mq",
    }:
        default_alpha = 0.5 if method_id in {
            "p04_skq_lrq_sq", "p05_skq_lrq_mq",
            "p08_skq_gonzalez_lrq_sq", "p09_skq_gonzalez_lrq_mq",
        } else 1.0
        return StructuredKQuadSelector(
            method_id, seed=seed, n_components=n_components,
            bandwidth_multiplier=float(options.get("bandwidth_multiplier", 1.0)),
            query_alpha=float(options.get("query_alpha", default_alpha)),
            budget_policy=str(options.get("budget_policy", "exact_total")),
        )
    if method_id in {
        "skq_pc", "skq_pc_lrq_sq", "skq_pc_sqrt", "skq_pc_r", "x_coretab_mc_cell",
    }:
        variant = {
            "skq_pc": "standard",
            "skq_pc_lrq_sq": "lrq_sq",
            "skq_pc_sqrt": "standard",
            "skq_pc_r": "random",
            "x_coretab_mc_cell": "coretab_cell",
        }[method_id]
        return SKQPCSelector(
            method_id,
            variant=variant,
            seed=seed,
            n_components=int(options.get("n_components", n_components)),
            max_cells_per_stratum=int(options.get("max_cells_per_stratum", 128)),
            bandwidth_multiplier=float(options.get("bandwidth_multiplier", 1.0)),
            query_alpha=float(options.get("query_alpha", 0.5)),
            allocation_policy=str(options.get(
                "allocation_policy",
                "sqrt_floor" if method_id == "skq_pc_sqrt" else "proportional",
            )),
            allocation_floor=int(options.get("allocation_floor", 20)),
        )
    if method_id in {"p10_skq_mr_coretab_xgb", "p11_skq_mr_bdis"}:
        return ShardedMergeReduceSelector(
            method_id,
            repo_root,
            seed=seed,
            options=options,
            n_components=n_components,
        )
    raise RuntimeError(f"Factory chưa có implementation cho {method_id}")


def build_generator(method_id: str, seed: int, *, external_root: str | Path, options: dict[str, Any] | None = None):
    spec = get_method_spec(method_id)
    if spec.output_kind != "synthetic" or spec.status != "implemented":
        raise RuntimeError(f"{method_id} không phải generator synthetic runnable")
    if method_id == "s_tame_official":
        return TAMEOfficialGenerator(Path(external_root), seed, options)
    return TDBenchGenerator(method_id, Path(external_root), seed, options)
