# Testing

```bash
python testing/test.py
python testing/predict.py --source path/to/photo.jpg
```

`test.py` scores the leaf validation photos and the held-out COCO128 photos. `predict.py` saves boxed images under `testing/runs/predict/`. Both read `training/runs/coco128-leaf/weights/best.pt`.
