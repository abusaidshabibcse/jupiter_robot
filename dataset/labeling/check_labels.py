#!/usr/bin/env python3
"""Check that every labeled leaf photo has a valid YOLO box.

    python dataset/labeling/check_labels.py

Writes dataset/labeled/leaf/LABEL_CHECK.txt and prints PASS or FAIL.
A passing dataset has one label file per image, class 0, and a box that
sits inside the image and covers a real object rather than a speck or
the whole frame.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "labeled" / "leaf"
REPORT = DATASET / "LABEL_CHECK.txt"


def check_split(split: str) -> tuple[list[str], int]:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split
    problems: list[str] = []
    images = [path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        problems.append(f"{split}: no images")
        return problems, 0
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    image_stems = {path.stem for path in images}
    missing = sorted(image_stems - label_stems)
    extra = sorted(label_stems - image_stems)
    for stem in missing:
        problems.append(f"{split}: image has no label: {stem}")
    for stem in extra:
        problems.append(f"{split}: label has no image: {stem}")

    checked = 0
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            problems.append(f"{split}: {image_path.name} should have 1 box, found {len(lines)}")
            continue
        parts = lines[0].split()
        if len(parts) != 5:
            problems.append(f"{split}: {image_path.name} label must be 5 numbers")
            continue
        try:
            class_id = int(parts[0])
            x, y, w, h = (float(value) for value in parts[1:])
        except ValueError:
            problems.append(f"{split}: {image_path.name} label is not numeric")
            continue
        if class_id != 0:
            problems.append(f"{split}: {image_path.name} class is {class_id}, expected 0 (leaf)")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            problems.append(f"{split}: {image_path.name} box is outside the image ({x:.3f} {y:.3f} {w:.3f} {h:.3f})")
        if w * h < 0.002:
            problems.append(f"{split}: {image_path.name} box is tiny ({w * h:.4f} of the image)")
        if w * h > 0.90:
            problems.append(f"{split}: {image_path.name} box covers almost the whole image")
        checked += 1
    return problems, checked


def main() -> None:
    if not DATASET.is_dir():
        raise SystemExit(f"Labeled dataset not found: {DATASET}")
    problems = []
    counts = {}
    for split in ("train", "val"):
        split_problems, checked = check_split(split)
        problems.extend(split_problems)
        counts[split] = checked

    status = "PASS" if not problems else "FAIL"
    lines = [
        f"status: {status}",
        f"train images with a valid box: {counts['train']}",
        f"val images with a valid box: {counts['val']}",
        "expected class: 0 leaf",
        "label format: class x_center y_center width height, values from 0 to 1",
        f"problems: {len(problems)}",
    ]
    lines.extend(problems[:50])
    if len(problems) > 50:
        lines.append(f"... {len(problems) - 50} more")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    print(f"Report: {REPORT}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
