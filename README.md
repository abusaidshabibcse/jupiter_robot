# Leaf detection

Dataset work, training, testing, and report results each have their own folder.

| Folder | What is in it |
| --- | --- |
| `dataset/` | The zip, the labels, the label check, and the COCO128 + leaf training set |
| `training/` | Fine-tune YOLOv8, YOLO11, and YOLO26 s and m. Weights go in `training/runs/<model>/` |
| `testing/` | Score held-out photos and draw boxes |
| `results/` | Per-model metrics for the report (`summary.md`, `<model>/metrics.md`) |
| `docs/PROJECT.md` | Full procedure, technique, and scores |

Start with [docs/PROJECT.md](docs/PROJECT.md).

## Models

One recipe trains `yolov8s`, `yolov8m`, `yolo11s`, `yolo11m`, `yolo26s`, and `yolo26m`. The leaf row is the performance number for this project. `predict.py` defaults to `training/runs/yolo26s/weights/best.pt`.

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8s | 0.988 | 0.972 | 0.983 | 0.741 | 0.334 | 0.183 |
| yolov8m | 0.996 | 0.988 | 0.993 | 0.814 | 0.319 | 0.183 |
| yolo11s | 0.977 | 0.991 | 0.994 | 0.872 | 0.265 | 0.174 |
| yolo11m | 0.967 | 0.918 | 0.973 | 0.817 | 0.278 | 0.150 |
| yolo26s | 0.893 | 0.879 | 0.930 | 0.528 | 0.484 | 0.260 |
| yolo26m | 0.963 | 0.988 | 0.992 | 0.753 | 0.359 | 0.199 |

```bash
python training/train.py
python testing/test.py
python testing/predict.py --source photo.jpg
```
