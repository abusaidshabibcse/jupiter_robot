#!/usr/bin/env python3
"""Build the training dataset: COCO128 plus the labeled leaf photos.

    python dataset/combined/build_coco128_leaf.py

COCO128 is downloaded into dataset/combined/coco128 the first time.
Leaf boxes are rewritten from class 0 to class 80 so they do not collide
with the COCO class "person". COCO128's own label files stay unchanged.

A 20% slice of COCO128 is held out of training so testing can look at
COCO images the trainer did not see. Leaf validation images are the
split already made by the labeling step.
"""

from __future__ import annotations

import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from coco_names import COCO_NAMES, LEAF_CLASS_ID, LEAF_NAME

ROOT = Path(__file__).resolve().parents[2]
LEAF = ROOT / "dataset" / "labeled" / "leaf"
COCO_ROOT = ROOT / "dataset" / "combined" / "coco128"
COCO_ZIP = ROOT / "dataset" / "combined" / "coco128.zip"
OUT = ROOT / "dataset" / "combined" / "coco128_leaf"
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def find_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png"):
        files.extend(folder.glob(pattern))
    return sorted(files)


def ensure_coco128() -> None:
    image_dir = COCO_ROOT / "images" / "train2017"
    if image_dir.is_dir() and find_images(image_dir):
        return
    if not COCO_ZIP.is_file():
        print(f"Downloading COCO128\n  {COCO128_URL}")
        COCO_ZIP.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COCO128_URL, COCO_ZIP)
    with zipfile.ZipFile(COCO_ZIP) as archive:
        archive.extractall(COCO_ROOT.parent)
    if not find_images(image_dir):
        raise SystemExit(f"COCO128 download did not produce images in {image_dir}")


def rewrite_leaf_label(src: Path, dest: Path) -> None:
    lines = []
    for raw in src.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        parts[0] = str(LEAF_CLASS_ID)
        lines.append(" ".join(parts))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")


def copy_image(src: Path, dest: Path) -> None:
    """Copy the photo into the combined set.

    Symlinks are not used. Ultralytics resolves a symlink to the real
    image path and then looks for labels next to that real file. The
    leaf-only labels still say class 0, which is "person" in this
    81-class set. A real copy keeps the trainer on the rewritten
    class-80 labels under coco128_leaf/labels/.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.copy2(src, dest)


def yaml_names() -> str:
    lines = ["names:"]
    for index, name in enumerate(COCO_NAMES):
        lines.append(f"  {index}: {name}")
    lines.append(f"  {LEAF_CLASS_ID}: {LEAF_NAME}")
    return "\n".join(lines) + "\n"


def write_list(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-repeat", type=int, default=1,
                        help="How many times each leaf training image is listed.")
    parser.add_argument("--coco-val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (LEAF / "data.yaml").exists():
        raise SystemExit("Labeled leaf dataset not found. Run dataset/labeling/label_from_zip.py first.")
    ensure_coco128()

    coco_images = find_images(COCO_ROOT / "images" / "train2017")
    if len(coco_images) < 10:
        raise SystemExit(f"Too few COCO128 images in {COCO_ROOT}")
    rng = random.Random(args.seed)
    shuffled = coco_images[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.coco_val_ratio))
    coco_val = sorted(shuffled[:n_val])
    coco_train = sorted(shuffled[n_val:])

    if OUT.exists():
        shutil.rmtree(OUT)

    leaf_train = sorted(path for path in (LEAF / "images" / "train").iterdir() if path.is_file())
    leaf_val = sorted(path for path in (LEAF / "images" / "val").iterdir() if path.is_file())

    for split, paths in (("train", leaf_train), ("val", leaf_val)):
        for image_path in paths:
            label_src = LEAF / "labels" / split / f"{image_path.stem}.txt"
            if not label_src.exists():
                raise SystemExit(f"Missing label for {image_path.name}")
            copy_image(image_path, OUT / "images" / split / image_path.name)
            rewrite_leaf_label(label_src, OUT / "labels" / split / f"{image_path.stem}.txt")

    leaf_train_rows = [str((OUT / "images" / "train" / path.name).resolve()) for path in leaf_train]
    leaf_val_rows = [str((OUT / "images" / "val" / path.name).resolve()) for path in leaf_val]
    train_rows = [str(path.resolve()) for path in coco_train]
    for _ in range(args.leaf_repeat):
        train_rows.extend(leaf_train_rows)
    val_rows = [str(path.resolve()) for path in coco_val] + leaf_val_rows

    write_list(OUT / "train.txt", train_rows)
    write_list(OUT / "val.txt", val_rows)
    write_list(OUT / "leaf_val.txt", leaf_val_rows)
    write_list(OUT / "coco128_val.txt", [str(path.resolve()) for path in coco_val])

    names = yaml_names()
    (OUT / "data.yaml").write_text(
        f"# COCO128 train + leaf train. Val is held-out COCO128 plus leaf val.\n"
        f"# Leaf is class {LEAF_CLASS_ID}. Each leaf training image is listed {args.leaf_repeat} time(s).\n"
        f"path: {OUT.resolve()}\n"
        "train: train.txt\n"
        "val: val.txt\n\n"
        + names
    )
    (OUT / "leaf_val.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: leaf_val.txt\nval: leaf_val.txt\n\n" + names
    )
    (OUT / "coco128_val.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: coco128_val.txt\nval: coco128_val.txt\n\n" + names
    )
    # Same train as data.yaml; val is leaf-only so best.pt follows leaf detection.
    (OUT / "train_leaf_val.yaml").write_text(
        "# Same train list as data.yaml. Validation is the held-out leaf photos,\n"
        "# so best.pt follows leaf detection instead of the mixed 81-class score.\n"
        f"path: {OUT.resolve()}\n"
        "train: train.txt\n"
        "val: leaf_val.txt\n\n"
        + names
    )

    print(f"COCO128 train/held-out : {len(coco_train)} / {len(coco_val)}")
    print(f"Leaf train/val         : {len(leaf_train)} / {len(leaf_val)}")
    print(f"Train list rows        : {len(train_rows)}")
    print(f"Train yaml             : {OUT / 'data.yaml'}")
    print(f"Leaf-val train yaml    : {OUT / 'train_leaf_val.yaml'}")


if __name__ == "__main__":
    main()
