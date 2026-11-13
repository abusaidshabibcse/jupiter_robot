# Training

Fine-tune YOLO26 **s** and **m** on the combined COCO128 + leaf set. Each size has its own script and recipe.

```bash
source .venv/bin/activate
python training/train_yolo26s.py
python training/train_yolo26m.py
```

| Script | Base weights | Spec |
| --- | --- | --- |
| `train_yolo26s.py` | `yolo26s.pt` | `training/specs/yolo26s.yaml` |
| `train_yolo26m.py` | `yolo26m.pt` | `training/specs/yolo26m.yaml` |

Default data yaml is `dataset/combined/coco128_leaf/train_leaf_val.yaml`, so `best.pt` is chosen on the held-out leaf photos.

## Recipes

| Model | Spec | What is different |
| --- | --- | --- |
| `yolo26s` | `training/specs/yolo26s.yaml` | 100 epochs, patience 40, AdamW `0.001`, mosaic 0.6. The small head needs the longer run |
| `yolo26m` | `training/specs/yolo26m.yaml` | 80 epochs, patience 30, AdamW `0.0004`, weight decay `0.001`, mosaic 0.4, rotate 5°. The medium model takes a smaller step |

Shared choices:

- Supervised fine-tune of a COCO-pretrained checkpoint
- Leaf is class `80` so COCO classes `0`–`79` keep their pretrained head rows
- First 10 layers frozen
- Image size 640, batch 16
- Mild geometric augmentation (per-spec degrees / flips / mosaic)

If `yolo26m` runs out of memory on a 12 GB GPU, that script retries at batch 8:

```bash
python training/train_yolo26m.py --batch 8
```

## Output

Each run writes `training/runs/<model>/weights/best.pt`.

## Baseline (before hybrid relabel)

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26s | 1.000 | 0.998 | 0.995 | 0.937 | 0.364 | 0.220 |
| yolo26m | 0.998 | 1.000 | 0.995 | 0.937 | 0.329 | 0.204 |

Compare new runs against leaf mAP50-95 **0.937**. Full rationale: [../docs/PROJECT.md](../docs/PROJECT.md).
