"""Package này điều phối stage mà không chứa công thức của selector hay model.

Runner đọc config, kiểm tra gate, gọi đúng module và lưu artifact. Việc tách riêng
giúp thay cách chạy local/Kaggle mà không thay thuật toán nghiên cứu.
"""

from .runner import run_config, run_smoke

__all__ = ["run_config", "run_smoke"]

