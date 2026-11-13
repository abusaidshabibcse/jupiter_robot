# Testing

```bash
source .venv/bin/activate

# Score both YOLO26 checkpoints on leaf val and held-out COCO128
python testing/test.py

# Draw boxes. Default weights: training/runs/yolo26s/weights/best.pt
python testing/predict.py --source photo.jpg

python testing/predict.py --model yolo26s --source photo.jpg
python testing/predict.py --model yolo26m --source photo.jpg
```

`test.py` writes plots to `testing/runs/<model>/leaf` and `testing/runs/<model>/coco128`.
`predict.py` saves boxed images under `testing/runs/predict/<model>/`.
The leaf row is the performance number for this project.

## Baseline (before hybrid relabel)

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26s | 1.000 | 0.998 | 0.995 | 0.937 | 0.364 | 0.220 |
| yolo26m | 0.998 | 1.000 | 0.995 | 0.937 | 0.329 | 0.204 |

Per-model write-ups: [../results/summary.md](../results/summary.md).
