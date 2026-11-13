#!/usr/bin/env python3
"""Refine leaf labels with a stronger teacher (SAM2 or YOLO).

    # Recommended: SAM2.1-L for accurate boxes on photos without a BW mask
    python dataset/labeling/relabel_with_teacher.py --mode hybrid --backend sam

    # Optional: fine-tuned YOLO26m
    python dataset/labeling/relabel_with_teacher.py --mode hybrid --backend yolo \
        --teacher training/runs/yolo26m/weights/best.pt

Modes:

  hybrid  Keep boxes that came from a BW TIFF mask. Re-label every other
          photo with the teacher (these were color / weaker boxes).
  all     Re-label every photo with the teacher (ignores BW masks).

SAM is preferred for accuracy: it segments the specimen without needing a
leaf class, so boxes follow the leaf edge instead of echoing old labels.
This script writes class 0 into dataset/labeled/leaf/. Labels are backed up
before overwrite.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "dataset" / "labeled" / "leaf"
RAW = ROOT / "dataset" / "raw" / "leaf"
BW_ROOT = RAW / "BW"
LEAF_CLASS = 0
TEACHER_LEAF_ID = 80
DEFAULT_SAM = "sam2.1_l.pt"
DEFAULT_YOLO = ROOT / "training" / "runs" / "yolo26m" / "weights" / "best.pt"


def mask_index() -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not BW_ROOT.is_dir():
        return found
    for path in BW_ROOT.rglob("*.TIFF"):
        stem = path.stem[:-2] if path.stem.endswith("_B") else path.stem
        found[stem] = path
    return found


def original_stem(labeled_stem: str) -> str:
    """labeled names are '{species}__{original}'. Return the original stem."""
    if "__" in labeled_stem:
        return labeled_stem.split("__", 1)[1]
    return labeled_stem


def to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    return f"{LEAF_CLASS} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}"


def draw_preview(image_path: Path, box: tuple[float, float, float, float], dest: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(0, 255, 0), width=4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.resize((max(1, image.width // 2), max(1, image.height // 2))).save(dest, quality=80)


def backup_labels() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = LABELED / f"labels_backup_{stamp}"
    shutil.copytree(LABELED / "labels", dest)
    return dest


def predict_box_yolo(model, image_path: Path, conf: float) -> tuple[float, float, float, float] | None:
    results = model.predict(str(image_path), conf=conf, verbose=False)
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    names = results[0].names or {}

    best_i = None
    best_conf = -1.0
    for i, cls_id in enumerate(clss):
        name = names.get(int(cls_id), str(cls_id))
        if int(cls_id) not in (TEACHER_LEAF_ID, LEAF_CLASS) and name != "leaf":
            continue
        if float(confs[i]) > best_conf:
            best_conf = float(confs[i])
            best_i = i
    if best_i is None:
        best_i = int(confs.argmax())
        best_conf = float(confs[best_i])
        if best_conf < conf:
            return None
    x1, y1, x2, y2 = map(float, xyxy[best_i])
    return x1, y1, x2, y2


def box_ok(box: tuple[float, float, float, float], width: int, height: int) -> bool:
    """Reject speck / full-frame boxes (same limits as check_labels.py)."""
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = (bw * bh) / max(1, width * height)
    return 0.002 <= area <= 0.90


def predict_box_sam(model, image_path: Path) -> tuple[float, float, float, float] | None:
    """Center-point prompt; keep the largest mask that is not full-frame."""
    with Image.open(image_path) as image:
        width, height = image.size
    results = model(
        str(image_path),
        points=[[[width // 2, height // 2]]],
        labels=[[1]],
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]
    for idx in order:
        box = tuple(map(float, boxes[int(idx)]))
        if box_ok(box, width, height):
            return box
    return None


def predict_box_color(image_path: Path) -> tuple[float, float, float, float] | None:
    import importlib.util

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "label_from_zip", Path(__file__).resolve().parent / "label_from_zip.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    box = mod.bbox_from_background(rgb)
    if box is None:
        return None
    height, width = rgb.shape[:2]
    out = tuple(map(float, box))
    return out if box_ok(out, width, height) else None


def yolo_line_to_xyxy(line: str, width: int, height: int) -> tuple[float, float, float, float] | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    _, cx, cy, bw, bh = map(float, parts)
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return x1, y1, x2, y2


def resolve_teacher(backend: str, teacher: Path | None) -> Path | str:
    if teacher is not None:
        path = teacher if teacher.is_absolute() else ROOT / teacher
        if path.is_file() or backend == "sam":
            # SAM weights may auto-download by stem name.
            return path if path.is_file() else str(teacher)
        if backend == "sam":
            return str(teacher)
        raise SystemExit(f"Teacher weights not found: {path}")
    if backend == "sam":
        return DEFAULT_SAM
    if not DEFAULT_YOLO.is_file():
        raise SystemExit(f"Default YOLO teacher not found: {DEFAULT_YOLO}")
    return DEFAULT_YOLO


def load_predictor(backend: str, teacher: Path | str, device: str, conf: float):
    """Return (predict_fn, teacher_name). predict_fn(image_path) -> box | None."""
    yolo_fallback = None
    if DEFAULT_YOLO.is_file():
        from ultralytics import YOLO

        yolo_fallback = YOLO(str(DEFAULT_YOLO))

    if backend == "sam":
        from ultralytics import SAM

        model = SAM(str(teacher))
        if device:
            try:
                model.to(device if device != "cpu" else "cpu")
            except Exception:
                pass

        def predict(image_path: Path):
            with Image.open(image_path) as image:
                width, height = image.size
            box = predict_box_sam(model, image_path)
            if box is not None and box_ok(box, width, height):
                return box, "sam"
            if yolo_fallback is not None:
                box = predict_box_yolo(yolo_fallback, image_path, conf)
                if box is not None and box_ok(box, width, height):
                    return box, "yolo-fallback"
            box = predict_box_color(image_path)
            if box is not None:
                return box, "color-fallback"
            return None, None

        return predict, str(teacher)

    from ultralytics import YOLO

    model = YOLO(str(teacher))

    def predict(image_path: Path):
        with Image.open(image_path) as image:
            width, height = image.size
        box = predict_box_yolo(model, image_path, conf)
        if box is not None and box_ok(box, width, height):
            return box, "yolo"
        box = predict_box_color(image_path)
        if box is not None:
            return box, "color-fallback"
        return None, None

    return predict, str(teacher)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("hybrid", "all"),
        default="hybrid",
        help="hybrid keeps BW-mask labels; all rewrites every label.",
    )
    parser.add_argument(
        "--backend",
        choices=("sam", "yolo"),
        default="sam",
        help="sam (default, SAM2.1-L) or yolo (fine-tuned detector).",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help=f"Weights path or stem. Default: {DEFAULT_SAM} for sam, yolo26m best.pt for yolo.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence only.")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--preview",
        type=int,
        default=-1,
        help="Number of preview images to rewrite. 0 writes none. -1 writes all changed.",
    )
    args = parser.parse_args()

    if not (LABELED / "images").is_dir() or not (LABELED / "labels").is_dir():
        raise SystemExit(f"Missing labeled dataset at {LABELED}. Run label_from_zip.py first.")

    teacher = resolve_teacher(args.backend, args.teacher)
    masks = mask_index()
    backup = backup_labels()
    print(f"Backed up labels to {backup}", flush=True)
    print(f"Loading {args.backend} teacher: {teacher}", flush=True)

    predict, teacher_name = load_predictor(args.backend, teacher, args.device, args.conf)

    stats = {"kept_mask": 0, "relabeled": 0, "failed": 0, "by_source": {}}
    failures: list[str] = []
    preview_budget = args.preview

    for split in ("train", "val"):
        image_dir = LABELED / "images" / split
        label_dir = LABELED / "labels" / split
        for image_path in sorted(image_dir.glob("*.JPG")):
            label_path = label_dir / f"{image_path.stem}.txt"
            orig = original_stem(image_path.stem)
            has_mask = orig in masks

            keep = args.mode == "hybrid" and has_mask
            if keep:
                stats["kept_mask"] += 1
                continue

            box, source = predict(image_path)
            if box is None:
                stats["failed"] += 1
                failures.append(str(image_path.relative_to(ROOT)))
                continue

            with Image.open(image_path) as image:
                width, height = image.size
            label_path.write_text(to_yolo(box, width, height) + "\n")
            stats["relabeled"] += 1
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

            if preview_budget != 0:
                draw_preview(
                    image_path,
                    box,
                    LABELED / "preview" / split / f"{image_path.stem}.jpg",
                )
                if preview_budget > 0:
                    preview_budget -= 1

            if (stats["relabeled"] + stats["failed"]) % 20 == 0:
                print(
                    f"  progress relabeled={stats['relabeled']} failed={stats['failed']}",
                    flush=True,
                )

    print(f"Mode            : {args.mode}", flush=True)
    print(f"Backend         : {args.backend}", flush=True)
    print(f"Teacher         : {teacher_name}", flush=True)
    print(f"Kept BW masks   : {stats['kept_mask']}", flush=True)
    print(f"Relabeled       : {stats['relabeled']}", flush=True)
    for source, count in sorted(stats["by_source"].items()):
        print(f"  via {source:16s}: {count}", flush=True)
    print(f"Failed          : {stats['failed']}", flush=True)
    if failures:
        print("Could not re-detect:", flush=True)
        for item in failures:
            print(f"  {item}", flush=True)
        raise SystemExit(1)
    print("Next: python dataset/labeling/check_labels.py", flush=True)
    print("Then: python dataset/combined/build_coco128_leaf.py", flush=True)


if __name__ == "__main__":
    main()
