# Leaf detection

Dataset work, training, and testing each have their own folder.

| Folder | What is in it |
| --- | --- |
| `dataset/` | The zip, the labels, the label check, and the COCO128 + leaf training set |
| `training/` | Fine-tune YOLO26. Weights go in `training/runs/<model>/` |
| `testing/` | Score the held-out photos and draw boxes on new ones |
| `docs/PROJECT.md` | Labeling through training and testing, including the technique and scores |

Start with [docs/PROJECT.md](docs/PROJECT.md).

## Which weights to use

| Model | Path | Leaf mAP50 | Best for |
| --- | --- | --- | --- |
| **yolo26n** | `training/runs/yolo26n/weights/best.pt` | 0.817 | **2-core / CPU laptop** |
| **yolo26s** | `training/runs/yolo26s/weights/best.pt` | 0.930 | GPU / better boxes |

```bash
python testing/predict.py --model yolo26n --device cpu --imgsz 416 --source photo.jpg
```
