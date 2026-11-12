#!/usr/bin/env python3
"""Score each trained leaf checkpoint and write report-ready files.

    python results/evaluate_models.py
    python results/evaluate_models.py --models yolo26n yolov8n yolo11n yolo26s

Writes:
  results/<model>/metrics.json
  results/<model>/metrics.md
  results/summary.csv
  results/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODELS = ("yolov8n", "yolo11n", "yolo26n", "yolo26s")


def leaf_metrics(metrics) -> dict[str, float]:
    """Pull the leaf class row when present, else the aggregate box metrics."""
    names = getattr(metrics, "names", {}) or {}
    leaf_id = next((int(i) for i, name in names.items() if name == "leaf"), None)
    if leaf_id is not None and hasattr(metrics, "class_result"):
        try:
            p, r, ap50, ap = metrics.class_result(leaf_id)
            return {
                "precision": float(p),
                "recall": float(r),
                "map50": float(ap50),
                "map50_95": float(ap),
            }
        except Exception:
            pass
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def evaluate_one(root: Path, model_name: str, device: str, imgsz: int) -> dict:
    from ultralytics import YOLO

    weights = root / "training" / "runs" / model_name / "weights" / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"Missing weights for {model_name}: {weights}")

    leaf_yaml = root / "dataset" / "combined" / "coco128_leaf" / "leaf_val.yaml"
    coco_yaml = root / "dataset" / "combined" / "coco128_leaf" / "coco128_val.yaml"
    if not leaf_yaml.is_file() or not coco_yaml.is_file():
        raise SystemExit("Combined dataset yaml missing. Run dataset/combined/build_coco128_leaf.py")

    model = YOLO(str(weights))
    project = str(root / "testing" / "runs")

    print(f"\n=== {model_name}: leaf ===")
    leaf = model.val(
        data=str(leaf_yaml),
        imgsz=imgsz,
        device=device,
        project=project,
        name=f"{model_name}-leaf",
        exist_ok=True,
    )
    print(f"\n=== {model_name}: coco128 ===")
    coco = model.val(
        data=str(coco_yaml),
        imgsz=imgsz,
        device=device,
        project=project,
        name=f"{model_name}-coco128",
        exist_ok=True,
    )

    leaf_row = leaf_metrics(leaf)
    coco_row = {
        "precision": float(coco.box.mp),
        "recall": float(coco.box.mr),
        "map50": float(coco.box.map50),
        "map50_95": float(coco.box.map),
    }
    return {
        "model": model_name,
        "weights": str(weights.relative_to(root)),
        "weights_mb": round(weights.stat().st_size / 1e6, 2),
        "imgsz": imgsz,
        "device": device,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "leaf": leaf_row,
        "coco128": coco_row,
        "testing_runs": {
            "leaf": f"testing/runs/{model_name}-leaf",
            "coco128": f"testing/runs/{model_name}-coco128",
        },
    }


def write_model_files(out_dir: Path, row: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
    leaf = row["leaf"]
    coco = row["coco128"]
    md = f"""# {row['model']}

| Item | Value |
| --- | --- |
| Weights | `{row['weights']}` |
| Size | {row['weights_mb']} MB |
| Image size | {row['imgsz']} |
| Device | {row['device']} |
| Evaluated | {row['evaluated_at']} |

## Leaf validation (85 photos)

| Metric | Value |
| --- | --- |
| Precision | {leaf['precision']:.4f} |
| Recall | {leaf['recall']:.4f} |
| mAP50 | {leaf['map50']:.4f} |
| mAP50-95 | {leaf['map50_95']:.4f} |

Ultralytics plots: `{row['testing_runs']['leaf']}`

## Held-out COCO128 (26 photos)

| Metric | Value |
| --- | --- |
| Precision | {coco['precision']:.4f} |
| Recall | {coco['recall']:.4f} |
| mAP50 | {coco['map50']:.4f} |
| mAP50-95 | {coco['map50_95']:.4f} |

Ultralytics plots: `{row['testing_runs']['coco128']}`
"""
    (out_dir / "metrics.md").write_text(md)


def write_summary(results_dir: Path, rows: list[dict]) -> None:
    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "weights_mb",
                "leaf_precision",
                "leaf_recall",
                "leaf_map50",
                "leaf_map50_95",
                "coco128_map50",
                "coco128_map50_95",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["weights_mb"],
                    f"{row['leaf']['precision']:.4f}",
                    f"{row['leaf']['recall']:.4f}",
                    f"{row['leaf']['map50']:.4f}",
                    f"{row['leaf']['map50_95']:.4f}",
                    f"{row['coco128']['map50']:.4f}",
                    f"{row['coco128']['map50_95']:.4f}",
                ]
            )

    lines = [
        "# Model comparison",
        "",
        "Leaf detection on the held-out 85 leaf photos, plus held-out COCO128.",
        "Each model has its own folder under `results/` for the report.",
        "",
        "| Model | Size (MB) | Leaf P | Leaf R | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | Report folder |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['weights_mb']} | "
            f"{row['leaf']['precision']:.3f} | {row['leaf']['recall']:.3f} | "
            f"{row['leaf']['map50']:.3f} | {row['leaf']['map50_95']:.3f} | "
            f"{row['coco128']['map50']:.3f} | [`results/{row['model']}/`]({row['model']}/) |"
        )
    lines.extend(
        [
            "",
            "## How to reproduce",
            "",
            "```bash",
            "python results/evaluate_models.py",
            "```",
            "",
        ]
    )
    (results_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in args.models:
        row = evaluate_one(root, model_name, args.device, args.imgsz)
        write_model_files(results_dir / model_name, row)
        rows.append(row)
        print(
            f"{model_name}: leaf mAP50={row['leaf']['map50']:.3f} "
            f"mAP50-95={row['leaf']['map50_95']:.3f}"
        )

    write_summary(results_dir, rows)
    print(f"\nWrote {results_dir / 'summary.md'}")
    print(f"Wrote {results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
