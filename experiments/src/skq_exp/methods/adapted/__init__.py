"""Các baseline thích nghi có contract khác native và phải mang nhãn adapted."""

from .coretab_ova import CoreTabXGBOVASelector
from .modern_baselines import (
    CCSOOFXGBSelector,
    DataOOBOfficialSelector,
    KernelThinningClasswiseSelector,
    ModerateDSTabularSelector,
)

__all__ = [
    "CCSOOFXGBSelector",
    "CoreTabXGBOVASelector",
    "DataOOBOfficialSelector",
    "KernelThinningClasswiseSelector",
    "ModerateDSTabularSelector",
]
