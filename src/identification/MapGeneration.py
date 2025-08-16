"""Python shim for the :mod:`identification.MapGeneration` extension.

This module attempts to import :class:`CalibrationMap` from the compiled
extension module :mod:`identification.MapGeneration`. If the extension is not
available, an :class:`ImportError` is raised.
"""

from __future__ import annotations

try:  # pragma: no cover - extension may not be built
    from .MapGeneration import CalibrationMap  # type: ignore
except Exception as exc:  # pragma: no cover - extension may not be built
    raise ImportError("MapGeneration extension not built") from exc

__all__ = ["CalibrationMap"]