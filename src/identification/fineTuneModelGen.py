"""Python wrapper for the C++ fine-tuning executable."""

import pathlib
import subprocess
import sys
from typing import Optional, Sequence

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Invoke the compiled C++ fine-tuning binary.
    Parameters
    ----------
    argv: Optional[Sequence[str]]
        Optional list of command line arguments. If ``None`` ``sys.argv[1:]`` is used.
    """
    if argv is None:
        argv = sys.argv[1:]
    
    exe = pathlib.Path(__file__).with_name("fine_tune_model_gen")
    if not exe.is_file():
        raise FileNotFoundError(f"Compiled binary '{exe}' not found")
    cmd = [str(exe)] + list(argv)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":  # pragma: no cover - manual execution
    main()