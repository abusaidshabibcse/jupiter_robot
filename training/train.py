#!/usr/bin/env python3
"""Fine-tune a pretrained YOLO checkpoint on the COCO128 + leaf dataset.

    python training/train.py --model yolov8n.pt
    python training/train.py --model yolo11n.pt
    python training/train.py --model yolo26n.pt
    python training/train.py --model yolo26s.pt

Weights land under training/runs/<name>/weights/best.pt.
The default run name is the model stem (yolov8n, yolo11n, yolo26n, yolo26s).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path,
                        default=root / "dataset" / "combined" / "coco128_leaf" / "data.yaml")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Pretrained Ultralytics checkpoint (yolov8n, yolo11n, yolo26n, yolo26s).")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU index, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers. Use 0 or 2 on a 2-core machine.")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Freeze the first N layers. 10 keeps the backbone. Use 0 to train everything.")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Initial learning rate. Use 0.0003 for the unfrozen second stage.")
    parser.add_argument("--name", default=None,
                        help="Run folder under training/runs/. Defaults to the model stem.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.model.endswith(".yaml"):
        raise SystemExit("Refusing to train from a .yaml file. Pass a pretrained .pt checkpoint.")
    if not args.data.exists():
        raise SystemExit(f"Dataset yaml not found: {args.data}")

    run_name = args.name or Path(args.model).stem

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        freeze=args.freeze if args.freeze > 0 else None,
        optimizer="AdamW",
        lr0=args.lr0,
        patience=20,
        warmup_epochs=1,
        # Studio leaves have little background variety. Mild geometric
        # augmentation teaches the model the leaf can be rotated or scaled.
        degrees=10,
        flipud=0.2,
        fliplr=0.5,
        mosaic=0.8,
        close_mosaic=10,
        cache=True,
        project=str(Path(__file__).resolve().parent / "runs"),
        name=run_name,
        exist_ok=False,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
