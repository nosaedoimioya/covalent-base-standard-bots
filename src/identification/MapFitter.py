"""Python shim for the :mod:`identification.MapFitter` extension.

This module imports the :class:`MapFitter` class from the compiled extension
module :mod:`identification.MapFitter`. If the extension is not available, an
appropriate :class:`ImportError` is raised.
"""

from __future__ import annotations

try:  # pragma: no cover - extension may not be built
    from .MapFitter import MapFitter  # type: ignore
except Exception as exc:  # pragma: no cover - extension may not be built
    raise ImportError("MapFitter extension not built") from exc

__all__ = ["MapFitter"]
