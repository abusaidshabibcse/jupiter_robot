# Leaf detection

Dataset work stays in `dataset/`. Training, testing, and report numbers live in `yolo/`.

| Folder | What is in it |
| --- | --- |
| `dataset/` | The zip, the labels, the label check, and the COCO128 + leaf training set |
| `yolo/` | Fine-tune YOLO26s and YOLO26m, score them, and draw boxes. Weights go in `yolo/runs/` |
| `docs/PROJECT.md` | Labeling through training and testing, including the technique and scores |
| `docs/LEARNING.md` | What YOLO is, why the sizes differ, how to read the scores, and how to experiment |

Start with [docs/PROJECT.md](docs/PROJECT.md). For the ideas behind those steps, read [docs/LEARNING.md](docs/LEARNING.md).

## Models

`yolo26s` and `yolo26m` each have their own training script and hyperparameter spec. The leaf row is the performance number for this project. `predict.py` defaults to `yolo/runs/yolo26s/weights/best.pt`.

Leaf mAP50-95 after whole-specimen hybrid relabel: **0.946** (`yolo26s`), **0.924** (`yolo26m`). Earlier broken SAM leaflet boxes scored **0.937** but looked wrong on compound leaves.

```bash
# Free GPU if local LLM is up:  cd ~/Files/project/local_llm && ./stop.sh

# Refine color-only boxes with SAM2.1-L (BW masks kept)
python dataset/labeling/relabel_with_teacher.py --mode hybrid --backend sam
python dataset/labeling/check_labels.py
python dataset/combined/build_coco128_leaf.py

python yolo/train_yolo26s.py
python yolo/train_yolo26m.py
python yolo/test.py
python yolo/predict.py --source photo.jpg
```

Checkpoint files (`*.pt`, including `sam2.1_l.pt` and `yolo/weights/`) stay on this machine. They are listed in `.gitignore` and are not part of the GitHub repo.
