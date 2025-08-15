"""Identification utilities."""
from .SineSweepReader import SineSweepReader as PySineSweepReader
from .MapGeneration import CalibrationMap


try:
    from ._sinesweepreader import SineSweepReader  # type: ignore
except Exception:  # pragma: no cover - extension may not be built
    SineSweepReader = PySineSweepReader

__all__ = ["SineSweepReader", "PySineSweepReader"]
__all__ = ["CalibrationMap"]