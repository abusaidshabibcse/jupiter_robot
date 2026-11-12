# Testing

```bash
source .venv/bin/activate

# Score all six checkpoints on leaf val and held-out COCO128
python testing/test.py

# Draw boxes. Default weights: training/runs/yolo26s/weights/best.pt
python testing/predict.py --source photo.jpg

# Any of the other five best.pt files
python testing/predict.py --model yolov8s --source photo.jpg
python testing/predict.py --model yolov8m --source photo.jpg
python testing/predict.py --model yolo11s --source photo.jpg
python testing/predict.py --model yolo11m --source photo.jpg
python testing/predict.py --model yolo26m --source photo.jpg
```

`test.py` writes plots to `testing/runs/<model>/leaf` and `testing/runs/<model>/coco128`.
`predict.py` saves boxed images under `testing/runs/predict/<model>/`.
The leaf row is the performance number for this project.

## Measured on this machine

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8s | 0.988 | 0.972 | 0.983 | 0.741 | 0.334 | 0.183 |
| yolov8m | 0.996 | 0.988 | 0.993 | 0.814 | 0.319 | 0.183 |
| yolo11s | 0.977 | 0.991 | 0.994 | 0.872 | 0.265 | 0.174 |
| yolo11m | 0.967 | 0.918 | 0.973 | 0.817 | 0.278 | 0.150 |
| yolo26s | 0.893 | 0.879 | 0.930 | 0.528 | 0.484 | 0.260 |
| yolo26m | 0.963 | 0.988 | 0.992 | 0.753 | 0.359 | 0.199 |

Per-model write-ups: [../results/summary.md](../results/summary.md).
