# Model comparison

Leaf detection on the held-out 85 leaf photos, plus held-out COCO128.
Each model has its own folder under `results/` for the report.
The leaf row is the performance number for this project.

Baseline before hybrid relabel (previous shared recipe):

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26s | 1.000 | 0.998 | 0.995 | 0.937 | 0.364 | 0.220 |
| yolo26m | 0.998 | 1.000 | 0.995 | 0.937 | 0.329 | 0.204 |

After hybrid relabel + per-model training, re-run `python testing/test.py` to refresh this table.

## How to reproduce

```bash
python testing/test.py
```
