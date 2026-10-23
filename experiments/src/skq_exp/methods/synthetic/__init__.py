"""Các phương pháp trong package này sinh bảng train mới thay vì chọn dòng thật.

Chúng dùng một contract riêng vì dữ liệu sinh không có chỉ số trỏ về train gốc.
Runner vẫn đánh giá cùng learner và metric, nhưng lưu X/y sinh ra cùng provenance.
"""

from .result import GeneratedDatasetResult
from .tame import TAMEOfficialGenerator
from .tdbench import TDBenchGenerator

__all__ = ["GeneratedDatasetResult", "TAMEOfficialGenerator", "TDBenchGenerator"]
