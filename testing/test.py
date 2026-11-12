#!/usr/bin/env python3
"""Test a trained checkpoint on the held-out leaf photos and on held-out COCO128.

    python testing/test.py --model yolo26n
    python testing/test.py --model yolo26s
    python testing/test.py --weights path/to/best.pt

Leaf score is the number that says whether leaf detection worked.
The COCO128 score uses the 20% of COCO128 that was kept out of training.
Results are written under testing/runs/<model>-leaf and <model>-coco128.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def resolve_weights(root: Path, model: str | None, weights: Path | None) -> Path:
    if weights is not None:
        return weights
    if model:
        candidate = root / "training" / "runs" / model / "weights" / "best.pt"
        if candidate.is_file():
            return candidate
        raise SystemExit(
            f"Weights not found for model '{model}': {candidate}\n"
            f"Train first with: python training/train.py --model {model}.pt"
        )
    # Prefer the CPU-friendly nano if present, else the small model.
    for name in ("yolo11n", "yolo26n", "yolov8n", "yolo26s", "coco128-leaf"):
        candidate = root / "training" / "runs" / name / "weights" / "best.pt"
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "No trained weights found under training/runs/.\n"
        "Train with: python training/train.py --model yolo11n.pt"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Run name under training/runs/ (yolo26n, yolo26s).")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Explicit checkpoint path. Overrides --model.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    weights = resolve_weights(root, args.model, args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    run_tag = args.model or weights.parent.parent.name

    from ultralytics import YOLO

    model = YOLO(str(weights))
    project = str(root / "testing" / "runs")
    sets = [
        (f"{run_tag}-leaf", root / "dataset" / "combined" / "coco128_leaf" / "leaf_val.yaml"),
        (f"{run_tag}-coco128", root / "dataset" / "combined" / "coco128_leaf" / "coco128_val.yaml"),
    ]
    print(f"Using weights: {weights}")
    for name, data in sets:
        if not data.is_file():
            raise SystemExit(f"Missing {data}. Run dataset/combined/build_coco128_leaf.py first.")
        print(f"\n=== test {name}: {data.name} ===")
        model.val(data=str(data), imgsz=args.imgsz, device=args.device, project=project, name=name, exist_ok=True)


if __name__ == "__main__":
    main()
