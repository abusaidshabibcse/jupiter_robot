#!/usr/bin/env python3
"""Score the YOLO26 s/m checkpoints. Same entry point as yolo/test.py.

    python yolo/evaluate_models.py
    python yolo/test.py
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "test.py"),
        run_name="__main__",
    )
