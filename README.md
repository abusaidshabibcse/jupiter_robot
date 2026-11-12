# Leaf detection

Dataset work, training, testing, and report results each have their own folder.

| Folder | What is in it |
| --- | --- |
| `dataset/` | The zip, the labels, the label check, and the COCO128 + leaf training set |
| `training/` | Fine-tune YOLO. Weights go in `training/runs/<model>/` |
| `testing/` | Score held-out photos and draw boxes |
| `results/` | Per-model metrics for the report (`summary.md`, `<model>/metrics.md`) |
| `docs/PROJECT.md` | Full procedure, technique, and scores |

Start with [docs/PROJECT.md](docs/PROJECT.md). Comparison table: [results/summary.md](results/summary.md).

## Models

| Model | Path | Leaf mAP50 | Best for |
| --- | --- | ---: | --- |
| **yolo11n** | `training/runs/yolo11n/weights/best.pt` | **0.980** | Best nano leaf score / strong on CPU |
| yolo26n | `training/runs/yolo26n/weights/best.pt` | 0.817 | YOLO26 nano |
| yolov8n | `training/runs/yolov8n/weights/best.pt` | 0.525 | Classic YOLOv8 nano |
| yolo26s | `training/runs/yolo26s/weights/best.pt` | 0.930 | Larger YOLO26 on GPU |

```bash
# Recommended on a 2-core laptop
python testing/predict.py --model yolo11n --device cpu --imgsz 416 --source photo.jpg
```
