"""Identification utilities."""
from .SineSweepReader import SineSweepReader as PySineSweepReader
from .MapGeneration import CalibrationMap

try:
    from ._sinesweepreader import SineSweepReader  # type: ignore
except Exception:  # pragma: no cover - extension may not be built
    SineSweepReader = PySineSweepReader

try:
    from .processCalibrationData import processCalibrationData  # type: ignore
except Exception:  # pragma: no cover - extension may not be built
    def processCalibrationData(*args, **kwargs):  # type: ignore
        raise ImportError("processCalibrationData extension not built")

__all__ = [
    "SineSweepReader",
    "PySineSweepReader",
    "CalibrationMap",
    "processCalibrationData",
]