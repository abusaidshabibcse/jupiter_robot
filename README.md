# Leaf detection

Dataset work, training, testing, and report results each have their own folder.

| Folder | What is in it |
| --- | --- |
| `dataset/` | The zip, the labels, the label check, and the COCO128 + leaf training set |
| `training/` | Fine-tune YOLO26s. Weights go in `training/runs/` |
| `testing/` | Score the held-out photos and draw boxes on new ones |
| `docs/PROJECT.md` | Labeling through training and testing, including the technique and scores |
| `docs/LEARNING.md` | What YOLO is, why the sizes differ, how to read the scores, and how to experiment |

Start with [docs/PROJECT.md](docs/PROJECT.md). For the ideas behind those steps, read [docs/LEARNING.md](docs/LEARNING.md).

## Models

`yolo26s` and `yolo26m` each have their own training script and hyperparameter spec. The leaf row is the performance number for this project. `predict.py` defaults to `training/runs/yolo26s/weights/best.pt`.

Baseline leaf mAP50-95 (before hybrid relabel): **0.937** for both.

```bash
# Free GPU if local LLM is up:  cd ~/Files/project/local_llm && ./stop.sh

# Refine color-only boxes with SAM2.1-L (BW masks kept)
python dataset/labeling/relabel_with_teacher.py --mode hybrid --backend sam
python dataset/labeling/check_labels.py
python dataset/combined/build_coco128_leaf.py

python training/train_yolo26s.py
python training/train_yolo26m.py
python testing/test.py
python testing/predict.py --source photo.jpg
```
