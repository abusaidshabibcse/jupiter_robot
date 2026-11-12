# Testing

```bash
source .venv/bin/activate

# 2-core / CPU laptop (nano, default)
python testing/predict.py --model yolo26n --device cpu --imgsz 416 --source photo.jpg

# Stronger GPU model
python testing/predict.py --model yolo26s --device 0 --imgsz 640 --source photo.jpg

# Score held-out sets (needs local dataset images)
python testing/test.py --model yolo26n
python testing/test.py --model yolo26s
```

`predict.py` saves boxed images under `testing/runs/predict/<model>/`. The leaf class is class **80** and prints as `leaf`.

## Measured leaf scores

| Model | Weights | mAP50 | mAP50-95 | Use |
| --- | --- | --- | --- | --- |
| yolo26n | `training/runs/yolo26n/weights/best.pt` | 0.817 | 0.514 | 2-core / CPU |
| yolo26s | `training/runs/yolo26s/weights/best.pt` | 0.930 | 0.528 | GPU / quality |

Details: [../docs/PROJECT.md](../docs/PROJECT.md).
