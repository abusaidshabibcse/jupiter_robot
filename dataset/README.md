# Dataset

| Path | Role |
| --- | --- |
| `raw/leaf.zip` | Original archive |
| `raw/leaf/` | Unzipped photos, masks, and `leaf.csv` |
| `labeling/label_from_zip.py` | Build YOLO boxes from the zip |
| `labeling/check_labels.py` | Confirm every photo has one valid box |
| `labeled/leaf/` | Images, labels, previews, and `LABEL_CHECK.txt` |
| `combined/coco128/` | Downloaded COCO128 |
| `combined/coco128_leaf/` | COCO128 and leaf together. This is what training reads |

Commands and the labeling method are in [../docs/PROJECT.md](../docs/PROJECT.md).
