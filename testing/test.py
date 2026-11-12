#!/usr/bin/env python3
"""Test a trained checkpoint on the held-out leaf photos and on held-out COCO128.

    python testing/test.py

Leaf score is the number that says whether leaf detection worked.
The COCO128 score uses the 20% of COCO128 that was kept out of training.
Results are written under testing/runs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "training" / "runs" / "coco128-leaf" / "weights" / "best.pt",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(
            f"Weights not found: {args.weights}\n"
            "Train first with: python training/train.py"
        )

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    project = str(root / "testing" / "runs")
    sets = [
        ("leaf", root / "dataset" / "combined" / "coco128_leaf" / "leaf_val.yaml"),
        ("coco128", root / "dataset" / "combined" / "coco128_leaf" / "coco128_val.yaml"),
    ]
    for name, data in sets:
        if not data.is_file():
            raise SystemExit(f"Missing {data}. Run dataset/combined/build_coco128_leaf.py first.")
        print(f"\n=== test {name}: {data.name} ===")
        model.val(data=str(data), imgsz=args.imgsz, device=args.device, project=project, name=name, exist_ok=True)


if __name__ == "__main__":
    main()
