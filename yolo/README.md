# YOLO26

Train, score, and draw boxes for the leaf detector. Dataset scripts stay in `dataset/`.

| Path | What it is |
| --- | --- |
| `weights/yolo26s.pt`, `weights/yolo26m.pt` | COCO starting checkpoints. Gitignored. Ultralytics can download them if they are missing |
| `specs/yolo26s.yaml`, `specs/yolo26m.yaml` | Epochs, learning rate, freeze, mosaic |
| `train_yolo26s.py`, `train_yolo26m.py` | One recipe per size |
| `runs/<model>/weights/best.pt` | Fine-tuned checkpoint. Gitignored |
| `test.py` | Leaf val and held-out COCO128 |
| `predict.py` | Draw boxes on a new photo |
| `eval/` | Plots from `test.py` and `predict.py`. Gitignored |
| `results/` | Report numbers (`summary.md`, `<model>/metrics.md`) |

```bash
source .venv/bin/activate
python yolo/train_yolo26s.py
python yolo/train_yolo26m.py
python yolo/test.py
python yolo/predict.py --source photo.jpg
python yolo/predict.py --model yolo26m --source photo.jpg
```

Default data yaml is `dataset/combined/coco128_leaf/train_leaf_val.yaml`, so `best.pt` is chosen on the held-out leaf photos. `predict.py` defaults to `yolo/runs/yolo26s/weights/best.pt`.

## Recipes

| Model | Spec | What is different |
| --- | --- | --- |
| `yolo26s` | `yolo/specs/yolo26s.yaml` | 100 epochs, patience 40, AdamW `0.001`, mosaic 0.6 |
| `yolo26m` | `yolo/specs/yolo26m.yaml` | 100 epochs, patience 45, AdamW `0.0006`, freeze 8, mosaic 0.5. On OOM the script retries at batch 8 |

Shared choices: supervised fine-tune, leaf is class `80`, image size 640, batch 16.

## Scores after whole-specimen hybrid relabel

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26s | 0.988 | 0.988 | 0.984 | 0.946 | 0.376 | 0.207 |
| yolo26m | 0.976 | 0.988 | 0.981 | 0.924 | 0.350 | 0.196 |

Earlier broken SAM leaflet boxes scored leaf mAP50-95 **0.937** and looked wrong on compound leaves. Full procedure: [../docs/PROJECT.md](../docs/PROJECT.md).
