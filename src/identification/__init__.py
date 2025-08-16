"""Identification utilities."""
from .MapGeneration import CalibrationMap
from .MapFitter import MapFitter
from .SineSweepReader import SineSweepReader

try:
    from .processCalibrationData import processCalibrationData  # type: ignore
except Exception:  # pragma: no cover - extension may not be built
    def processCalibrationData(*args, **kwargs):  # type: ignore
        raise ImportError("processCalibrationData extension not built")

__all__ = [
    "SineSweepReader",
    "CalibrationMap",
    "MapFitter",
    "processCalibrationData",
]