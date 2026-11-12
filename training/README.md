# Training

Fine-tune YOLOv8, YOLO11, and YOLO26 in the s and m sizes on `dataset/combined/coco128_leaf/data.yaml`.

```bash
source .venv/bin/activate
python training/train.py
```

That command trains these six checkpoints and nothing else:

| Run folder | Base weights |
| --- | --- |
| `yolov8s/` | `yolov8s.pt` |
| `yolov8m/` | `yolov8m.pt` |
| `yolo11s/` | `yolo11s.pt` |
| `yolo11m/` | `yolo11m.pt` |
| `yolo26s/` | `yolo26s.pt` |
| `yolo26m/` | `yolo26m.pt` |

A `.yaml` file or any other model name is refused. YOLO11 has no `v` in the filename.

## Recipe

Every run uses the same settings, so the scores compare the architectures:

- Supervised fine-tune of a COCO-pretrained checkpoint
- Leaf is class `80` so COCO classes `0`–`79` keep their pretrained head rows
- First 10 layers frozen
- AdamW at learning rate `0.001`
- 80 epochs, patience 20, image size 640, batch 16
- Mild geometric augmentation (rotate, flips, mosaic)

If an m model runs out of memory on the 12 GB GPU, that model is trained again at batch 8:

```bash
python training/train.py --model yolov8m.pt --batch 8
```

## Output

Each run writes `training/runs/<model>/weights/best.pt`.

## Measured on this machine

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8s | 0.988 | 0.972 | 0.983 | 0.741 | 0.334 | 0.183 |
| yolov8m | 0.996 | 0.988 | 0.993 | 0.814 | 0.319 | 0.183 |
| yolo11s | 0.977 | 0.991 | 0.994 | 0.872 | 0.265 | 0.174 |
| yolo11m | 0.967 | 0.918 | 0.973 | 0.817 | 0.278 | 0.150 |
| yolo26s | 0.893 | 0.879 | 0.930 | 0.528 | 0.484 | 0.260 |
| yolo26m | 0.963 | 0.988 | 0.992 | 0.753 | 0.359 | 0.199 |

All six finished at batch 16.

Full rationale: [../docs/PROJECT.md](../docs/PROJECT.md).
