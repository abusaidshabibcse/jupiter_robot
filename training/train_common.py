#!/usr/bin/env python3
"""Shared helpers for per-model YOLO26 trainers."""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAINING = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset" / "combined" / "coco128_leaf" / "train_leaf_val.yaml"


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


def load_spec(path: Path) -> dict[str, Any]:
    from ultralytics.utils import YAML

    data = YAML.load(str(path))
    if not isinstance(data, dict):
        raise SystemExit(f"Spec must be a mapping: {path}")
    # `val: leaf` documents train_leaf_val.yaml; Ultralytics rejects it as a kwarg.
    data.pop("val", None)
    return data


def build_parser(stem: str, spec_path: Path) -> argparse.ArgumentParser:
    spec = load_spec(spec_path)
    parser = argparse.ArgumentParser(
        description=f"Fine-tune {stem} using {spec_path.name}."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=ROOT / f"{stem}.pt")
    parser.add_argument("--spec", type=Path, default=spec_path)
    parser.add_argument("--epochs", type=int, default=int(spec.get("epochs", 80)))
    parser.add_argument("--imgsz", type=int, default=int(spec.get("imgsz", 640)))
    parser.add_argument("--batch", type=int, default=int(spec.get("batch", 16)))
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser


def train_kwargs(stem: str, args: argparse.Namespace, batch: int) -> dict[str, Any]:
    spec = load_spec(args.spec)
    freeze = int(spec.get("freeze", 10))
    kwargs: dict[str, Any] = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "device": args.device,
        "workers": args.workers,
        "freeze": freeze if freeze > 0 else None,
        "optimizer": "AdamW",
        "lr0": float(spec.get("lr0", 0.001)),
        "lrf": float(spec.get("lrf", 0.01)),
        "patience": int(spec.get("patience", 20)),
        "warmup_epochs": float(spec.get("warmup_epochs", 1)),
        "weight_decay": float(spec.get("weight_decay", 0.0005)),
        "degrees": float(spec.get("degrees", 0)),
        "flipud": float(spec.get("flipud", 0)),
        "fliplr": float(spec.get("fliplr", 0.5)),
        "mosaic": float(spec.get("mosaic", 1.0)),
        "close_mosaic": int(spec.get("close_mosaic", 10)),
        "cache": True,
        "project": str(TRAINING / "runs"),
        "name": stem,
        "exist_ok": False,
        "resume": args.resume,
    }
    return kwargs


def train_one(stem: str, args: argparse.Namespace, batch: int) -> None:
    run_dir = TRAINING / "runs" / stem
    if args.resume:
        if not run_dir.is_dir():
            raise SystemExit(f"Nothing to resume at {run_dir}")
    elif run_dir.exists():
        shutil.rmtree(run_dir)

    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    if not weights.is_file():
        raise SystemExit(f"Base weights not found: {weights}")
    if not args.data.exists():
        raise SystemExit(
            f"Dataset yaml not found: {args.data}\n"
            "Run dataset/combined/build_coco128_leaf.py first."
        )

    from ultralytics import YOLO

    model = YOLO(str(weights))
    try:
        model.train(**train_kwargs(stem, args, batch))
    finally:
        del model
        release_cuda()


def run_trainer(stem: str, allow_oom_retry: bool = False) -> None:
    spec_path = TRAINING / "specs" / f"{stem}.yaml"
    if not spec_path.is_file():
        raise SystemExit(f"Missing spec: {spec_path}")
    parser = build_parser(stem, spec_path)
    args = parser.parse_args()
    # Re-apply epochs/imgsz/batch from CLI after parse; defaults already set.
    batch = args.batch
    print(f"=== train {stem} batch {batch} ===", flush=True)
    print(f"spec={args.spec} data={args.data}", flush=True)
    try:
        train_one(stem, args, batch)
    except Exception as exc:
        if allow_oom_retry and batch != 8 and is_oom(exc):
            print(f"{stem} ran out of memory at batch {batch}; retrying at batch 8", flush=True)
            release_cuda()
            batch = 8
            train_one(stem, args, batch)
        else:
            raise
    print(f"=== finished {stem} batch {batch} -> training/runs/{stem}/weights/best.pt ===", flush=True)
