from .MapGeneration import CalibrationMap

"""Identification utilities.

The original Python implementation of :class:`SineSweepReader` is kept only
for backwards compatibility.  The C++ extension
``identification._sinesweepreader`` supersedes this legacy module and should
be preferred when available.
"""
try:  # pragma: no cover - optional legacy dependency
    from .SineSweepReader import SineSweepReader as PySineSweepReader
except Exception:  # pragma: no cover - legacy reader may be missing
    class PySineSweepReader:  # type: ignore
        """Stub for the legacy Python :class:`SineSweepReader`.

        The C++ extension provides the functional implementation.
        """

        pass

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