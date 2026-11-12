# Testing

```bash
source .venv/bin/activate

# Recommended on a 2-core / CPU laptop
python testing/predict.py --model yolo11n --device cpu --imgsz 416 --source photo.jpg

# Other trained models
python testing/predict.py --model yolov8n --device cpu --imgsz 416 --source photo.jpg
python testing/predict.py --model yolo26n --device cpu --imgsz 416 --source photo.jpg
python testing/predict.py --model yolo26s --device 0 --imgsz 640 --source photo.jpg

# Score held-out sets (needs local dataset images)
python testing/test.py --model yolo11n
python results/evaluate_models.py
```

`predict.py` saves boxed images under `testing/runs/predict/<model>/`.
`evaluate_models.py` writes report files under `results/<model>/` plus `results/summary.md`.

## Leaf validation scores

| Model | mAP50 | mAP50-95 | Size |
| --- | ---: | ---: | ---: |
| yolo11n | 0.980 | 0.813 | 5.5 MB |
| yolo26s | 0.930 | 0.528 | 20.4 MB |
| yolo26n | 0.817 | 0.514 | 5.5 MB |
| yolov8n | 0.525 | 0.285 | 6.5 MB |

Full tables: [../results/summary.md](../results/summary.md).
