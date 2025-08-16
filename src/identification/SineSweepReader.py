"""Legacy Python SineSweepReader stub.

This module formerly contained the original Python implementation of
``SineSweepReader``.  The functionality has since been superseded by the
C++ extension :mod:`identification._sinesweepreader`.

The stub is preserved to maintain backwards compatibility for projects
that imported :mod:`identification.SineSweepReader`.  It raises a
:class:`NotImplementedError` when instantiated, directing users to build
and use the C++ extension instead.
"""

from __future__ import annotations

try:  # pragma: no cover - extension may not be built
    from .SineSweepReader import SineSweepReader  # type: ignore
except Exception as exc:  # pragma: no cover - extension may not be built
    raise ImportError("SineSweepReader extension not built") from exc

__all__ = ["SineSweepReader"]