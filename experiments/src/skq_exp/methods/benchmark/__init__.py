"""Các method ở đây tái hiện công thức của một benchmark chung, không gọi native paper.

Tên và provenance luôn ghi `benchmark` để bảng kết quả không nhập nhằng với repo
tác giả của một thuật toán gốc.
"""

from .tabular_distillation import GonzalezBenchmarkSelector, LeverageBenchmarkSelector

__all__ = ["GonzalezBenchmarkSelector", "LeverageBenchmarkSelector"]
