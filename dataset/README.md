# Dataset

| Path | Role |
| --- | --- |
| `raw/leaf.zip` | Original archive |
| `raw/leaf/` | Unzipped photos, masks, and `leaf.csv` |
| `labeling/label_from_zip.py` | Build YOLO boxes from the zip |
| `labeling/relabel_with_teacher.py` | Hybrid refine: keep BW masks; re-label others with SAM2.1-L (+ YOLO/color fallback) |
| `labeling/check_labels.py` | Confirm every photo has one valid box |
| `labeled/leaf/` | Images, labels, previews, and `LABEL_CHECK.txt` |
| `combined/coco128/` | Downloaded COCO128 |
| `combined/coco128_leaf/` | COCO128 and leaf together. Leaf photos are copied here with class-80 labels. This is what training reads |

Commands and the labeling method are in [../docs/PROJECT.md](../docs/PROJECT.md). Leaf must be class `80` in the combined set. Symlinking the leaf photos back to `labeled/leaf/` makes Ultralytics read the class-`0` labels and train leaves as `person`.
