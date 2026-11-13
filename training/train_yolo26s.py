#!/usr/bin/env python3
"""Fine-tune yolo26s with training/specs/yolo26s.yaml.

    python training/train_yolo26s.py

Uses dataset/combined/coco128_leaf/train_leaf_val.yaml by default so best.pt
follows leaf validation. See training/README.md.
"""

from __future__ import annotations

from train_common import run_trainer


if __name__ == "__main__":
    run_trainer("yolo26s", allow_oom_retry=False)
