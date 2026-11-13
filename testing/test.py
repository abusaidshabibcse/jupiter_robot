#!/usr/bin/env python3
"""Score the fine-tuned YOLO26 checkpoints on leaf val and held-out COCO128.

    python testing/test.py
    python testing/test.py --model yolo26s yolo26m

Leaf score is the number that says whether leaf detection worked.
The COCO128 score uses the 20% of COCO128 that was kept out of training.
Results are written under testing/runs/<model>/leaf and testing/runs/<model>/coco128
so the runs do not overwrite each other.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ALLOWED = ("yolo26s", "yolo26m")


def allowed_list() -> str:
    return ", ".join(ALLOWED)


def best_pt(root: Path, name: str) -> Path:
    return root / "training" / "runs" / name / "weights" / "best.pt"


def parse_model_name(root: Path, raw: str) -> str:
    if raw.endswith((".yaml", ".yml")):
        raise SystemExit(f"Refusing '{raw}'. Score only: {allowed_list()}")
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((Path.cwd() / path, root / path))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        for name in ALLOWED:
            if resolved == best_pt(root, name).resolve():
                return name
    stem = path.stem if path.suffix == ".pt" else raw
    if "/" not in raw and "\\" not in raw and stem in ALLOWED:
        return stem
    raise SystemExit(
        f"Refusing model '{raw}'. Choose one of: {allowed_list()}\n"
        "Or pass that run's training/runs/<model>/weights/best.pt path."
    )


def leaf_metrics(metrics) -> dict[str, float]:
    """Pull the leaf class row when present, else the aggregate box metrics."""
    names = getattr(metrics, "names", {}) or {}
    leaf_id = next((int(i) for i, name in names.items() if name == "leaf"), None)
    if leaf_id is not None and hasattr(metrics, "class_result"):
        try:
            precision, recall, ap50, ap = metrics.class_result(leaf_id)
            return {
                "precision": float(precision),
                "recall": float(recall),
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


def trained_batch(root: Path, model_name: str) -> int | None:
    args_path = root / "training" / "runs" / model_name / "args.yaml"
    if not args_path.is_file():
        return None
    for line in args_path.read_text().splitlines():
        if line.startswith("batch:"):
            return int(float(line.split(":", 1)[1].strip()))
    return None


def score_one(root: Path, model_name: str, device: str, imgsz: int) -> dict:
    from ultralytics import YOLO

    weights = best_pt(root, model_name)
    if not weights.is_file():
        raise SystemExit(
            f"Weights not found for model '{model_name}': {weights}\n"
            "Train first with: python training/train_yolo26s.py or train_yolo26m.py"
        )

    leaf_yaml = root / "dataset" / "combined" / "coco128_leaf" / "leaf_val.yaml"
    coco_yaml = root / "dataset" / "combined" / "coco128_leaf" / "coco128_val.yaml"
    if not leaf_yaml.is_file() or not coco_yaml.is_file():
        raise SystemExit("Combined dataset yaml missing. Run dataset/combined/build_coco128_leaf.py")

    model = YOLO(str(weights))
    project = str(root / "testing" / "runs" / model_name)
    print(f"\n=== {model_name}: leaf ===", flush=True)
    leaf = model.val(
        data=str(leaf_yaml),
        imgsz=imgsz,
        device=device,
        project=project,
        name="leaf",
        exist_ok=True,
    )
    print(f"\n=== {model_name}: coco128 ===", flush=True)
    coco = model.val(
        data=str(coco_yaml),
        imgsz=imgsz,
        device=device,
        project=project,
        name="coco128",
        exist_ok=True,
    )
    return {
        "model": model_name,
        "weights": str(weights.relative_to(root)),
        "weights_mb": round(weights.stat().st_size / 1e6, 2),
        "batch": trained_batch(root, model_name),
        "imgsz": imgsz,
        "device": device,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "leaf": leaf_metrics(leaf),
        "coco128": {
            "precision": float(coco.box.mp),
            "recall": float(coco.box.mr),
            "map50": float(coco.box.map50),
            "map50_95": float(coco.box.map),
        },
        "testing_runs": {
            "leaf": f"testing/runs/{model_name}/leaf",
            "coco128": f"testing/runs/{model_name}/coco128",
        },
    }


def write_model_files(out_dir: Path, row: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
    leaf = row["leaf"]
    coco = row["coco128"]
    batch = row.get("batch")
    batch_line = f"| Train batch | {batch} |\n" if batch not in (None, 16) else ""
    md = f"""# {row['model']}

| Item | Value |
| --- | --- |
| Weights | `{row['weights']}` |
| Size | {row['weights_mb']} MB |
| Image size | {row['imgsz']} |
| Device | {row['device']} |
{batch_line}| Evaluated | {row['evaluated_at']} |

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
                "batch",
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
                    "" if row.get("batch") is None else row["batch"],
                    row["weights_mb"],
                    f"{row['leaf']['precision']:.4f}",
                    f"{row['leaf']['recall']:.4f}",
                    f"{row['leaf']['map50']:.4f}",
                    f"{row['leaf']['map50_95']:.4f}",
                    f"{row['coco128']['map50']:.4f}",
                    f"{row['coco128']['map50_95']:.4f}",
                ]
            )

    odd_batch = [row for row in rows if row.get("batch") not in (None, 16)]
    lines = [
        "# Model comparison",
        "",
        "Leaf detection on the held-out 85 leaf photos, plus held-out COCO128.",
        "Each model has its own folder under `results/` for the report.",
        "The leaf row is the performance number for this project.",
        "",
        "| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | "
            f"{row['leaf']['precision']:.3f} | {row['leaf']['recall']:.3f} | "
            f"{row['leaf']['map50']:.3f} | {row['leaf']['map50_95']:.3f} | "
            f"{row['coco128']['map50']:.3f} | {row['coco128']['map50_95']:.3f} |"
        )
    if odd_batch:
        notes = ", ".join(f"{row['model']} used batch {row['batch']}" for row in odd_batch)
        lines.extend(["", f"Train batch was 16 except: {notes}.", ""])
    else:
        lines.append("")
    lines.extend(
        [
            "## How to reproduce",
            "",
            "```bash",
            "python testing/test.py",
            "```",
            "",
        ]
    )
    (results_dir / "summary.md").write_text("\n".join(lines))


def load_rows(results_dir: Path) -> list[dict] | None:
    rows = []
    for name in ALLOWED:
        path = results_dir / name / "metrics.json"
        if not path.is_file():
            return None
        rows.append(json.loads(path.read_text()))
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="*", default=None,
                        help=f"Subset to score. Default: both ({allowed_list()}).")
    parser.add_argument("--weights", type=Path, default=None,
                        help="One of the training/runs/<model>/weights/best.pt files.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.weights is not None:
        models = (parse_model_name(root, str(args.weights)),)
    elif args.model:
        models = tuple(parse_model_name(root, name) for name in args.model)
    else:
        models = ALLOWED

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        row = score_one(root, model_name, args.device, args.imgsz)
        write_model_files(results_dir / model_name, row)
        print(
            f"{model_name}: leaf P={row['leaf']['precision']:.3f} "
            f"R={row['leaf']['recall']:.3f} "
            f"mAP50={row['leaf']['map50']:.3f} "
            f"mAP50-95={row['leaf']['map50_95']:.3f} "
            f"coco128 mAP50={row['coco128']['map50']:.3f} "
            f"mAP50-95={row['coco128']['map50_95']:.3f}",
            flush=True,
        )

    rows = load_rows(results_dir)
    if rows is not None:
        write_summary(results_dir, rows)
        print(f"\nWrote {results_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
