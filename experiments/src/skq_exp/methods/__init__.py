"""Package này chứa các phương pháp chọn hoặc nén dữ liệu.

Code bên ngoài chỉ lấy method qua registry để tên trong config, manifest và bảng
kết quả luôn giống nhau. Repo tác giả được gọi qua boundary riêng, không trộn code
native vào các implementation đề xuất.
"""

from .registry import (
    METHOD_SPECS,
    allowed_learners_for,
    build_generator,
    build_selector,
    get_method_spec,
)

__all__ = [
    "METHOD_SPECS",
    "allowed_learners_for",
    "build_generator",
    "build_selector",
    "get_method_spec",
]
