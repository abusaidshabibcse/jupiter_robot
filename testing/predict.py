#!/usr/bin/env python3
"""Draw detections and save the images under testing/runs/predict/.

    python testing/predict.py --source path/to/photo.jpg

By default this loads the trained COCO128+leaf checkpoint. Pass --model
to point at a different weights file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    testing_dir = Path(__file__).resolve().parent
    default_weights = testing_dir.parent / "training" / "runs" / "coco128-leaf" / "weights" / "best.pt"
    parser.add_argument("--model", default=str(default_weights),
                        help="Trained checkpoint. Defaults to training/runs/coco128-leaf/weights/best.pt.")
    parser.add_argument("--source", required=True, help="Image, folder, or video.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, default=testing_dir / "runs" / "predict")
    args = parser.parse_args()

    weights = Path(args.model)
    if not weights.is_file():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first with: python training/train.py"
        )

    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(
        source=args.source, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for result in results:
        image = Image.open(result.path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for x1, y1, x2, y2, label in _box(result, names=model.names):
            draw.rectangle((x1, y1, x2, y2), outline=(40, 220, 40), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(40, 220, 40), font=font)
        dest = args.out / Path(result.path).name
        image.save(dest)
        print(dest)

def _box(result, names, force_name: str = ""):
    if result.boxes is None:
        return []
    items = []
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), score, class_id in zip(xyxy, conf, cls):
        name = force_name or names[int(class_id)]
        items.append((float(x1), float(y1), float(x2), float(y2), f"{name} {score:.2f}"))
    return items


if __name__ == "__main__":
    main()
