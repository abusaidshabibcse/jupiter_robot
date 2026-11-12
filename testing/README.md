# Testing

```bash
source .venv/bin/activate
python testing/test.py
python testing/predict.py --source path/to/photo.jpg
```

`test.py` scores `training/runs/coco128-leaf/weights/best.pt` on the leaf validation photos and on held-out COCO128. `predict.py` saves boxed images under `testing/runs/predict/`. The leaf class is class **80** and prints as `leaf`.

## Measured scores

| Set | Images | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| Leaf validation | 85 | 0.893 | 0.879 | 0.930 | 0.528 |
| Held-out COCO128 | 26 | 0.523 | 0.447 | 0.484 | 0.260 |

The leaf row is the leaf-detection result. Details are in [../docs/PROJECT.md](../docs/PROJECT.md).
