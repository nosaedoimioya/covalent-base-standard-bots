from __future__ import annotations
"""Robot utilities."""
try:
    import robots.TestRobot
except Exception as exc:
    raise ImportError("TestRobot extension not built") from exc

try:
    import robots.Standard
except Exception as exc:
    raise ImportError("Standard extension not built") from exc