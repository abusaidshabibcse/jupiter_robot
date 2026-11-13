# Results for the report

Each trained model has its own folder. Use these files in the report instead of digging through Ultralytics run logs.

| Path | What it is |
| --- | --- |
| `summary.md` | Comparison table across yolo26s and yolo26m |
| `summary.csv` | Same numbers for spreadsheets |
| `<model>/metrics.md` | One-model write-up |
| `<model>/metrics.json` | Machine-readable metrics |
| `../testing/runs/<model>/leaf/` | Ultralytics plots for leaf val |
| `../testing/runs/<model>/coco128/` | Ultralytics plots for held-out COCO128 |

```bash
source .venv/bin/activate
python testing/test.py
```

Models compared: `yolo26s`, `yolo26m`. Baseline leaf mAP50-95 before hybrid relabel: **0.937**.
