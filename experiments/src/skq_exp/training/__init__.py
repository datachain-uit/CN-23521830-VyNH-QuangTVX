"""Package này train model và tính kết quả từ một selection đã hợp lệ.

Các learner dùng chung cấu hình đã freeze; evaluator không được tự tune riêng cho
từng coreset. Metric tổng thể, từng nhãn và chi phí được sinh từ cùng prediction.
"""

from .evaluator import (
    FittedLearner,
    evaluate_fitted_learner,
    fit_generated_learner,
    fit_selection_learner,
)
from .learners import LEARNER_SPECS, build_learner
from .metrics import compute_all_metrics
from .completeness import audit_run_completeness

__all__ = [
    "FittedLearner", "LEARNER_SPECS", "build_learner", "compute_all_metrics",
    "audit_run_completeness", "evaluate_fitted_learner", "fit_generated_learner",
    "fit_selection_learner",
]
