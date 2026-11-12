#!/usr/bin/env python3
"""Label the leaf zip into a YOLO detection dataset.

    python dataset/labeling/label_from_zip.py

The zip is dataset/raw/leaf.zip. It is extracted to dataset/raw/leaf/
when that folder is missing. Boxes are written to dataset/labeled/leaf/.

Label source, in this order:

1. The matching black-and-white TIFF, when the zip contains one. The box
   is the white pixels of that mask. No neural network is involved.
2. Photos with no mask. The paper color is read from the border and the
   box surrounds every pixel that is not paper.
3. Optional: --method sam uses Ultralytics MobileSAM (mobile_sam.pt)
   for photos that have no mask.
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "raw" / "leaf"
ZIP_PATH = ROOT / "dataset" / "raw" / "leaf.zip"
RGB_ROOT = RAW / "RGB"
BW_ROOT = RAW / "BW"
OUT = ROOT / "dataset" / "labeled" / "leaf"

# YOLO class index inside the leaf-only dataset. The combined COCO+leaf
# dataset rewrites this to 80 so it does not collide with "person".
LEAF_CLASS = 0


def ensure_raw_extracted() -> None:
    """Unzip dataset/raw/leaf.zip into dataset/raw/leaf when RGB is missing."""
    if RGB_ROOT.is_dir() and any(RGB_ROOT.rglob("*.JPG")):
        return
    if not ZIP_PATH.is_file():
        raise SystemExit(f"Missing zip: {ZIP_PATH}")
    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(RAW)
    if not any(RGB_ROOT.rglob("*.JPG")):
        raise SystemExit(f"Unzipped {ZIP_PATH} but found no JPG files in {RGB_ROOT}")


def mask_index() -> dict[str, Path]:
    found = {}
    for path in BW_ROOT.rglob("*.TIFF"):
        stem = path.stem[:-2] if path.stem.endswith("_B") else path.stem
        found[stem] = path
    return found


def bbox_from_binary(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return x1, y1, x2, y2 of the white leaf. Invert if the leaf is black."""
    white = mask > 127
    ratio = float(white.mean())
    if ratio > 0.5:
        white = ~white
        ratio = 1.0 - ratio
    if ratio < 0.002:
        return None
    ys, xs = np.nonzero(white)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _otsu(values: np.ndarray) -> float:
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return low
    hist, edges = np.histogram(values, bins=128, range=(low, high))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return low
    prob = hist / total
    omega = np.cumsum(prob)
    centers = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(prob * centers)
    denom = omega * (1.0 - omega)
    sigma = np.zeros_like(denom)
    valid = denom > 1e-9
    sigma[valid] = (mu[-1] * omega[valid] - mu[valid]) ** 2 / denom[valid]
    index = int(np.argmax(sigma))
    return float((edges[index] + edges[index + 1]) / 2)


def bbox_from_background(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Box the whole specimen on colored paper, including separate leaflets.

    Pink and red paper is the usual background here, so green-versus-red is a
    stronger signal than raw brightness. Every leaflet of a compound leaf is
    included: the box is the extent of all foreground pixels, not only the
    single largest piece.
    """
    height, width = image.shape[:2]
    edge = 8
    samples = np.concatenate(
        [
            image[:edge].reshape(-1, 3),
            image[-edge:].reshape(-1, 3),
            image[:, :edge].reshape(-1, 3),
            image[:, -edge:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    paper_gr = float(samples[:, 1].mean() - samples[:, 0].mean())
    paper_bg = float(samples[:, 2].mean() - samples[:, 1].mean())
    green_red = image[:, :, 1].astype(np.float32) - image[:, :, 0].astype(np.float32)
    blue_green = image[:, :, 2].astype(np.float32) - image[:, :, 1].astype(np.float32)
    score = np.abs(green_red - paper_gr) + 0.35 * np.abs(blue_green - paper_bg)
    paper_score = np.abs((samples[:, 1] - samples[:, 0]) - paper_gr)
    paper_score += 0.35 * np.abs((samples[:, 2] - samples[:, 1]) - paper_bg)
    cutoff = max(float(np.percentile(paper_score, 99)) + 12.0, _otsu(score[edge:-edge, edge:-edge]))
    foreground = score >= cutoff
    foreground[:edge, :] = False
    foreground[-edge:, :] = False
    foreground[:, :edge] = False
    foreground[:, -edge:] = False

    # Drop isolated paper specks. A 3x3 sum below 3 is noise; a thin needle survives.
    padded = np.pad(foreground.astype(np.uint8), 1)
    density = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    foreground = density >= 3
    rows, cols = np.nonzero(foreground)
    if len(cols) < 40:
        return None
    x1, y1, x2, y2 = int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if box_area > 0.85 * height * width or box_area < 80:
        return None
    return x1, y1, x2, y2


def load_sam(weights: str):
    from ultralytics import SAM

    model = SAM(weights)

    def predict(image_path: Path) -> tuple[int, int, int, int] | None:
        with Image.open(image_path) as image:
            width, height = image.size
        # One positive point in the middle. These photos have one centered leaf.
        results = model(
            str(image_path),
            points=[[[width // 2, height // 2]]],
            labels=[[1]],
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        # SAM returns masks; the box is the mask extent. Take the largest.
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = boxes[int(areas.argmax())]
        return int(x1), int(y1), int(x2), int(y2)

    return predict


def to_yolo(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    return f"{LEAF_CLASS} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}"


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


def split_species(paths: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Keep at least one validation image per species when the species has 5+ photos."""
    rng = random.Random(seed)
    by_species: dict[str, list[Path]] = {}
    for path in paths:
        by_species.setdefault(path.parent.name, []).append(path)

    train, val = [], []
    for species in sorted(by_species):
        group = sorted(by_species[species])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio)) if len(group) >= 5 else max(1, len(group) // 5)
        n_val = min(n_val, len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def draw_preview(image_path: Path, box: tuple[int, int, int, int], dest: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(0, 255, 0), width=4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.resize((image.width // 2, image.height // 2)).save(dest, quality=80)


def write_yaml(out: Path) -> None:
    yaml = f"""# Leaf-only detection set. One class.
# Generated by scripts/label_leaves.py
path: {out.resolve()}
train: images/train
val: images/val

names:
  0: leaf
"""
    (out / "data.yaml").write_text(yaml)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("color", "sam"), default="color",
                        help="How to label RGB photos that have no BW mask.")
    parser.add_argument("--sam-weights", default="mobile_sam.pt",
                        help="SAM checkpoint used when --method sam. mobile_sam.pt is the fast one.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=-1,
                        help="Number of boxed preview images to write. 0 writes none. -1 writes all.")
    args = parser.parse_args()
    ensure_raw_extracted()

    images = sorted(RGB_ROOT.rglob("*.JPG"))
    if not images:
        raise SystemExit(f"No JPG files under {RGB_ROOT}")
    masks = mask_index()

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    sam_predict = load_sam(args.sam_weights) if args.method == "sam" else None
    train_paths, val_paths = split_species(images, args.val_ratio, args.seed)
    split_of = {path: "train" for path in train_paths}
    split_of.update({path: "val" for path in val_paths})

    stats = {"mask": 0, "color": 0, "sam": 0, "failed": 0}
    color_ious: list[float] = []
    color_recalls: list[float] = []
    preview_budget = args.preview
    failures: list[str] = []

    for image_path in images:
        split = split_of[image_path]
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        mask_path = masks.get(image_path.stem)
        source = ""
        box = None

        if mask_path is not None:
            mask = np.asarray(Image.open(mask_path).convert("L"))
            if mask.shape[:2] != (height, width):
                mask = np.asarray(Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST))
            box = bbox_from_binary(mask)
            source = "mask"
            # Quality check: would the color method have agreed with the mask?
            color_box = bbox_from_background(rgb)
            if box and color_box:
                color_ious.append(iou(box, color_box))
                ax1, ay1, ax2, ay2 = box
                bx1, by1, bx2, by2 = color_box
                inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
                inter_h = max(0, min(ay2, by2) - max(ay1, by1))
                color_recalls.append((inter_w * inter_h) / max(1, (ax2 - ax1) * (ay2 - ay1)))

        if box is None and sam_predict is not None:
            box = sam_predict(image_path)
            source = "sam"

        if box is None:
            box = bbox_from_background(rgb)
            source = "color" if box else source

        if box is None:
            stats["failed"] += 1
            failures.append(str(image_path))
            continue

        stats[source] += 1
        dest_image = OUT / "images" / split / f"{image_path.parent.name}__{image_path.name}"
        dest_label = OUT / "labels" / split / f"{dest_image.stem}.txt"
        shutil.copy2(image_path, dest_image)
        dest_label.write_text(to_yolo(box, width, height) + "\n")

        if preview_budget != 0:
            draw_preview(image_path, box, OUT / "preview" / split / f"{dest_image.stem}.jpg")
            if preview_budget > 0:
                preview_budget -= 1

    write_yaml(OUT)
    labeled = stats["mask"] + stats["color"] + stats["sam"]
    print(f"Labeled {labeled} / {len(images)} images")
    print(f"  from BW masks : {stats['mask']}")
    print(f"  from color    : {stats['color']}")
    print(f"  from SAM      : {stats['sam']}")
    print(f"  failed        : {stats['failed']}")
    print(f"Train images    : {len(list((OUT / 'images' / 'train').glob('*.JPG')))}")
    print(f"Val images      : {len(list((OUT / 'images' / 'val').glob('*.JPG')))}")
    if color_ious:
        arr = np.array(color_ious)
        recall = np.array(color_recalls)
        print(
            "Color boxes checked against the BW masks on the same photos: "
            f"they cover {recall.mean() * 100:.1f}% of the mask box on average "
            f"(IoU mean {arr.mean():.3f})."
        )
    if failures:
        print("Could not label:")
        for item in failures:
            print(" ", item)
    print(f"Dataset yaml    : {OUT / 'data.yaml'}")
    print(f"Review boxes in : {OUT / 'preview'}")


if __name__ == "__main__":
    main()
