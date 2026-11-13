#!/usr/bin/env python3
"""Fine-tune yolo26m with training/specs/yolo26m.yaml.

    python training/train_yolo26m.py

Uses dataset/combined/coco128_leaf/train_leaf_val.yaml by default so best.pt
follows leaf validation. On OOM, retries at batch 8. See training/README.md.
"""

from __future__ import annotations

from train_common import run_trainer


if __name__ == "__main__":
    run_trainer("yolo26m", allow_oom_retry=True)
