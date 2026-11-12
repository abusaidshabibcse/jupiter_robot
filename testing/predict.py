#!/usr/bin/env python3
"""Draw detections and save the images under testing/runs/predict/.

    # 2-core / CPU laptop (nano)
    python testing/predict.py --model yolo26n --device cpu --imgsz 416 --source photo.jpg

    # Stronger GPU model
    python testing/predict.py --model yolo26s --source photo.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def resolve_weights(root: Path, model: str | None, weights: Path | None) -> Path:
    if weights is not None:
        return weights
    if model:
        path = Path(model)
        if path.suffix == ".pt" or path.is_file():
            return path
        candidate = root / "training" / "runs" / model / "weights" / "best.pt"
        if candidate.is_file():
            return candidate
        raise SystemExit(
            f"Weights not found for model '{model}': {candidate}\n"
            f"Train first with: python training/train.py --model {model}.pt"
        )
    for name in ("yolo26n", "yolo26s", "coco128-leaf"):
        candidate = root / "training" / "runs" / name / "weights" / "best.pt"
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "No trained weights found under training/runs/.\n"
        "Train with: python training/train.py --model yolo26n.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    testing_dir = Path(__file__).resolve().parent
    root = testing_dir.parent
    parser.add_argument("--model", default="yolo26n",
                        help="Run name (yolo26n, yolo26s) or a .pt path. Default: yolo26n.")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Explicit checkpoint path. Overrides --model.")
    parser.add_argument("--source", required=True, help="Image, folder, or video.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Inference size. 416 is a good default for 2-core CPU; use 640 on GPU.")
    parser.add_argument("--device", default="cpu",
                        help="cpu, or a GPU index such as 0.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder. Defaults to testing/runs/predict/<model>/.")
    args = parser.parse_args()

    weights = resolve_weights(root, args.model, args.weights)
    if not weights.is_file():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first with: python training/train.py --model yolo26n.pt"
        )

    run_tag = args.weights.parent.parent.name if args.weights else (
        Path(args.model).stem if str(args.model).endswith(".pt") else args.model
    )
    out_dir = args.out or (testing_dir / "runs" / "predict" / run_tag)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(
        source=args.source, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using weights: {weights}")
    print(f"device={args.device} imgsz={args.imgsz}")
    for result in results:
        image = Image.open(result.path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for x1, y1, x2, y2, label in _box(result, names=model.names):
            draw.rectangle((x1, y1, x2, y2), outline=(40, 220, 40), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(40, 220, 40), font=font)
        dest = out_dir / Path(result.path).name
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
