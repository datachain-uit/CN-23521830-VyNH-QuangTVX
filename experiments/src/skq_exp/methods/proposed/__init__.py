"""Package này chứa engine SKQ và các khối toán dùng chung.

Các biến thể chỉ thay nguồn structure hoặc representation; allocation, RFF,
herding và QP dùng chung để ablation không vô tình khác nhau ở phần code khác.
"""

from .structured_kquad import StructuredKQuadSelector
from .skq_pc import SKQPCSelector

__all__ = ["SKQPCSelector", "StructuredKQuadSelector"]

