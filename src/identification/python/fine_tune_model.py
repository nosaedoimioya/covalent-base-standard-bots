"""Command line interface for the FineTuneModelGen C++ binding."""

from __future__ import annotations

import argparse
import sys
import os
from typing import Optional, Sequence

# Add the current directory to Python path to find the compiled modules
sys.path.insert(0, os.path.dirname(__file__))

try:  # pragma: no cover - extension may not be built
    from src.identification.lib.FineTuneModelGen import runFineTuneModelGen
except Exception as exc:  # pragma: no cover - extension may not be built
    raise ImportError("FineTuneModelGen extension not built") from exc

__all__ = ["runFineTuneModelGen", "main"]

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune saved shaper NN models")
    parser.add_argument("--model", required=True, help="Location file of existing models")
    parser.add_argument("maps", nargs="+", help="Calibration map pickle files for new data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save", default="", help="Output file for updated models")
    args = parser.parse_args(argv)

    return runFineTuneModelGen(args.model, args.maps, args.epochs, args.lr, args.save)

if __name__ == "__main__":  # pragma: no cover - manual execution
    main()