# Leaf detection

Dataset work, training, and testing each have their own folder.

| Folder | What is in it |
| --- | --- |
| `dataset/` | The zip, the labels, the label check, and the COCO128 + leaf training set |
| `training/` | Fine-tune YOLO26s. Weights go in `training/runs/` |
| `testing/` | Score the held-out photos and draw boxes on new ones |
| `docs/PROJECT.md` | The full procedure, including which method labeled the zip |

Start with [docs/PROJECT.md](docs/PROJECT.md).
