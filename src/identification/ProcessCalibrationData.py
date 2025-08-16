"""Python shim for the :mod:`identification.ProcessCalibrationData` extension.

This module imports the :func:`processCalibrationData` function from the
compiled extension module :mod:`identification.ProcessCalibrationData`. If the
extension is not available, an :class:`ImportError` is raised.
"""

from __future__ import annotations

try:  # pragma: no cover - extension may not be built
    from .ProcessCalibrationData import processCalibrationData  # type: ignore
except Exception as exc:  # pragma: no cover - extension may not be built
    raise ImportError("ProcessCalibrationData extension not built") from exc

__all__ = ["processCalibrationData"]