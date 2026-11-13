#!/usr/bin/env python3
"""Fine-tune YOLOv8, YOLO11, and YOLO26 in the s and m sizes.

    python training/train.py

Trains these checkpoints and nothing else, in this order:

    yolov8s.pt, yolov8m.pt, yolo11s.pt, yolo11m.pt, yolo26s.pt, yolo26m.pt

Each run is written to training/runs/<stem>/. A .yaml file or any other
model name is refused. Every run uses the same recipe (freeze 10, AdamW
at 0.001, 80 epochs, patience 20, image size 640, batch 16) so the scores
compare the architectures.

If an m model runs out of memory, that model is trained again at batch 8.
Repeat one checkpoint the same way:

    python training/train.py --model yolov8m.pt --batch 8
"""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path


ALLOWED = ("yolov8s", "yolov8m", "yolo11s", "yolo11m", "yolo26s", "yolo26m")


def allowed_list() -> str:
    return ", ".join(f"{name}.pt" for name in ALLOWED)


def parse_model_name(raw: str) -> str:
    if raw.endswith((".yaml", ".yml")):
        raise SystemExit(
            "Refusing to train from a .yaml file. Pass a pretrained .pt checkpoint: "
            + allowed_list()
        )
    stem = Path(raw).name
    if stem.endswith(".pt"):
        stem = stem[:-3]
    if stem not in ALLOWED:
        raise SystemExit(f"Refusing model '{raw}'. This script trains only: {allowed_list()}")
    return stem


def release_cuda() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def train_one(stem: str, args: argparse.Namespace, batch: int) -> None:
    run_dir = Path(__file__).resolve().parent / "runs" / stem
    if args.resume:
        if not run_dir.is_dir():
            raise SystemExit(f"Nothing to resume at {run_dir}")
    elif run_dir.exists():
        shutil.rmtree(run_dir)

    from ultralytics import YOLO

    model = YOLO(f"{stem}.pt")
    try:
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
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
            name=stem,
            exist_ok=False,
            resume=args.resume,
        )
    finally:
        del model
        release_cuda()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path,
                        default=root / "dataset" / "combined" / "coco128_leaf" / "data.yaml")
    parser.add_argument("--model", nargs="*", default=None,
                        help=f"Subset to train. Default: all six ({allowed_list()}).")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU index, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers. Use 0 or 2 on a 2-core machine.")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Freeze the first N layers. 10 keeps the backbone.")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Initial AdamW learning rate.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset yaml not found: {args.data}")

    models = ALLOWED if not args.model else tuple(parse_model_name(name) for name in args.model)
    if args.resume and len(models) != 1:
        raise SystemExit("--resume needs exactly one --model.")

    used: list[tuple[str, int]] = []
    for stem in models:
        batch = args.batch
        print(f"=== train {stem} batch {batch} ===", flush=True)
        try:
            train_one(stem, args, batch)
        except Exception as exc:
            if stem.endswith("m") and batch != 8 and is_oom(exc):
                print(f"{stem} ran out of memory at batch {batch}; retrying at batch 8", flush=True)
                release_cuda()
                batch = 8
                train_one(stem, args, batch)
            else:
                raise
        used.append((stem, batch))

    print("=== all requested training runs finished ===", flush=True)
    for stem, batch in used:
        print(f"  {stem}: batch {batch} -> training/runs/{stem}/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
