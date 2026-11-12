# Model comparison

Leaf detection on the held-out 85 leaf photos, plus held-out COCO128.
Each model has its own folder under `results/` for the report.

| Model | Size (MB) | Leaf P | Leaf R | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | Report folder |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| yolov8n | 6.53 | 0.462 | 0.906 | 0.525 | 0.285 | 0.354 | [`results/yolov8n/`](yolov8n/) |
| yolo11n | 5.53 | 0.987 | 0.965 | 0.980 | 0.813 | 0.298 | [`results/yolo11n/`](yolo11n/) |
| yolo26n | 5.52 | 0.706 | 0.906 | 0.817 | 0.514 | 0.457 | [`results/yolo26n/`](yolo26n/) |
| yolo26s | 20.42 | 0.893 | 0.879 | 0.930 | 0.528 | 0.484 | [`results/yolo26s/`](yolo26s/) |

## How to reproduce

```bash
python results/evaluate_models.py
```
