"""Các adapter ở đây gọi repo tác giả đã khóa commit.

Mỗi adapter chỉ đổi dữ liệu vào/ra sang contract chung. Nếu repo hoặc dependency
không phù hợp, nó báo BLOCKED/FAILED và tuyệt đối không thay bằng baseline khác.
"""

from .autocoreset import AutoCoresetNativeSelector
from .bdis import BDISNativeSelector
from .coretab import CoreTabNativeSelector
from .craig import CRAIGNativeSelector

__all__ = [
    "AutoCoresetNativeSelector", "BDISNativeSelector",
    "CoreTabNativeSelector", "CRAIGNativeSelector",
]
