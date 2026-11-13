# Leaf detection

Dataset work, training, and testing each have their own folder.

| Folder | What is in it |
| --- | --- |
| `dataset/` | The zip, the labels, the label check, and the COCO128 + leaf training set |
| `training/` | Fine-tune YOLO26s. Weights go in `training/runs/` |
| `testing/` | Score the held-out photos and draw boxes on new ones |
| `docs/PROJECT.md` | Labeling through training and testing, including the technique and scores |
| `docs/LEARNING.md` | What YOLO is, why the sizes differ, how to read the scores, and how to experiment |

Start with [docs/PROJECT.md](docs/PROJECT.md). For the ideas behind those steps, read [docs/LEARNING.md](docs/LEARNING.md).

On the leaf validation set, the trained checkpoint scores **mAP50 0.930** and **mAP50-95 0.528**.
