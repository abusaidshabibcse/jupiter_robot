# Model comparison

Leaf detection on the held-out 85 leaf photos, plus held-out COCO128.
Each model has its own folder under `yolo/results/` for the report.
The leaf row is the performance number for this project.

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26s | 0.988 | 0.988 | 0.984 | 0.946 | 0.376 | 0.207 |
| yolo26m | 0.976 | 0.988 | 0.981 | 0.924 | 0.350 | 0.196 |

## How to reproduce

```bash
python yolo/test.py
```
