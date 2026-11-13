# Model comparison

Leaf detection on the held-out 85 leaf photos, plus held-out COCO128.
Each model has its own folder under `results/` for the report.
The leaf row is the performance number for this project.

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8s | 0.988 | 0.972 | 0.983 | 0.741 | 0.334 | 0.183 |
| yolov8m | 0.996 | 0.988 | 0.993 | 0.814 | 0.319 | 0.183 |
| yolo11s | 0.977 | 0.991 | 0.994 | 0.872 | 0.265 | 0.174 |
| yolo11m | 0.967 | 0.918 | 0.973 | 0.817 | 0.278 | 0.150 |
| yolo26s | 0.893 | 0.879 | 0.930 | 0.528 | 0.484 | 0.260 |
| yolo26m | 0.963 | 0.988 | 0.992 | 0.753 | 0.359 | 0.199 |

## How to reproduce

```bash
python testing/test.py
```
