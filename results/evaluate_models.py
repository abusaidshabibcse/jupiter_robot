#!/usr/bin/env python3
"""Score the YOLO26 s/m checkpoints. Same entry point as testing/test.py.

    python results/evaluate_models.py
    python testing/test.py
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "testing" / "test.py"),
        run_name="__main__",
    )
