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

from typing import Any


class SineSweepReader:  # pragma: no cover - placeholder implementation
    """Stub for the legacy Python ``SineSweepReader``.

    The C++ extension :mod:`identification._sinesweepreader` supersedes this
    class.  Users should install and use the extension instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The Python implementation of SineSweepReader has been removed. "
            "Build the C++ extension `identification._sinesweepreader` for "
            "full functionality."
        )