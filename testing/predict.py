#!/usr/bin/env python3
"""Draw detections and save the images under testing/runs/predict/.

    python testing/predict.py --source photo.jpg
    python testing/predict.py --model yolov8m --source photo.jpg

The default weights are training/runs/yolo26s/weights/best.pt.
--model accepts any of the six fine-tuned best.pt files:
yolov8s, yolov8m, yolo11s, yolo11m, yolo26s, yolo26m.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ALLOWED = ("yolov8s", "yolov8m", "yolo11s", "yolo11m", "yolo26s", "yolo26m")


def allowed_list() -> str:
    return ", ".join(ALLOWED)


def best_pt(root: Path, name: str) -> Path:
    return root / "training" / "runs" / name / "weights" / "best.pt"


def resolve_weights(root: Path, model: str) -> tuple[Path, str]:
    raw_path = Path(model)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend((Path.cwd() / raw_path, root / raw_path))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        for name in ALLOWED:
            weights = best_pt(root, name).resolve()
            if resolved == weights:
                return weights, name
    stem = raw_path.stem if raw_path.suffix == ".pt" else model
    if "/" not in model and "\\" not in model and stem in ALLOWED:
        return best_pt(root, stem), stem
    raise SystemExit(
        f"Refusing model '{model}'. Choose one of: {allowed_list()}\n"
        "Or pass that run's training/runs/<model>/weights/best.pt path."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    testing_dir = Path(__file__).resolve().parent
    root = testing_dir.parent
    parser.add_argument("--model", default="yolo26s",
                        help=f"One of {allowed_list()}, or that run's best.pt. Default: yolo26s.")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Explicit best.pt. Must be one of the six fine-tuned checkpoints.")
    parser.add_argument("--source", required=True, help="Image, folder, or video.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Inference size. 416 is a good default for CPU; use 640 on GPU.")
    parser.add_argument("--device", default="cpu",
                        help="cpu, or a GPU index such as 0.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder. Defaults to testing/runs/predict/<model>/.")
    args = parser.parse_args()

    weights, run_tag = resolve_weights(root, str(args.weights) if args.weights else args.model)
    if not weights.is_file():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first with: python training/train.py"
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
