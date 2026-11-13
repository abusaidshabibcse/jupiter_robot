# Leaf detection project

This project turns `leaf.zip` into a YOLO dataset, checks the boxes, optionally refines weak labels with a teacher model, then fine-tunes **YOLO26 s** and **YOLO26 m** with **separate recipes** on **COCO128 plus the leaf photos**.

| Run folder | Base weights | Trainer |
| --- | --- | --- |
| `training/runs/yolo26s/` | `yolo26s.pt` | `training/train_yolo26s.py` |
| `training/runs/yolo26m/` | `yolo26m.pt` | `training/train_yolo26m.py` |

## Folders

```
jupiter/
  docs/PROJECT.md                 this file
  dataset/                        everything about the data
    raw/leaf.zip                  the zip you start from
    raw/leaf/                     unzipped RGB photos, BW masks, leaf.csv
    labeling/label_from_zip.py    makes the boxes
    labeling/relabel_with_teacher.py  hybrid refine of color-only boxes
    labeling/check_labels.py      checks that every box is valid
    labeled/leaf/                 YOLO leaf dataset and the preview images
    combined/coco128/             the 128 COCO images, downloaded once
    combined/coco128_leaf/        the dataset training actually reads
  training/
    train_yolo26s.py / train_yolo26m.py
    specs/yolo26s.yaml / yolo26m.yaml
    runs/<model>/                 one folder per trained checkpoint
  testing/
    test.py                       scores leaf val and held-out COCO128
    predict.py                    draws boxes on new photos
  results/
    summary.md                    comparison table for the report
    <model>/metrics.md            one-model write-up
```

## 0. Environment

Use Python 3.12. System Python 3.14 does not have a reliable PyTorch wheel for this stack.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -U -r requirements.txt
```

Training needs a free GPU. If a local LLM is holding the card, stop it first, then confirm with `nvidia-smi`.

## 1. Label the zip

```bash
source .venv/bin/activate
python dataset/labeling/label_from_zip.py
```

| Photos | How the box was made |
| --- | --- |
| Photos with a file in `dataset/raw/leaf/BW/` | Extent of the white mask pixels |
| Photos with no mask | Paper color from the border; box surrounds non-paper pixels |

## 2. Check labels

```bash
python dataset/labeling/check_labels.py
```

A good run prints `status: PASS`. Also review `dataset/labeled/leaf/preview/`.

## 2b. Hybrid relabel (recommended)

Keep BW-mask boxes. Re-label photos without a BW mask using **SAM2.1-L** (accurate segmentation boxes). Full-frame SAM misses fall back to fine-tuned yolo26m, then the color method:

```bash
# Free the GPU first if the local LLM stack is running:
#   cd ~/Files/project/local_llm && ./stop.sh
#   systemctl --user stop llama-server.service llm-gateway.service cloudflared-tunnel.service

python dataset/labeling/relabel_with_teacher.py --mode hybrid --backend sam
python dataset/labeling/check_labels.py
```

Labels are backed up under `dataset/labeled/leaf/labels_backup_<timestamp>/` before overwrite.

## 3. Build the COCO128 + leaf training set

```bash
python dataset/combined/build_coco128_leaf.py
```

Leaf photos are **copied** (not symlinked) and class `0` is rewritten to class **`80`**. Training default for the YOLO26 scripts is `train_leaf_val.yaml` so `best.pt` follows leaf validation.

## 4. Train

Separate recipes:

| Model | Spec highlights |
| --- | --- |
| `yolo26s` | 100 epochs, patience 40, lr0 `0.001`, mosaic 0.6 |
| `yolo26m` | 80 epochs, patience 30, lr0 `0.0004`, weight decay `0.001`, mosaic 0.4 |

```bash
python training/train_yolo26s.py
python training/train_yolo26m.py
```

If `yolo26m` OOMs, it retries at batch 8 (or pass `--batch 8`).

### Baseline before hybrid relabel

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26s | 1.000 | 0.998 | 0.995 | 0.937 | 0.364 | 0.220 |
| yolo26m | 0.998 | 1.000 | 0.995 | 0.937 | 0.329 | 0.204 |

Compare new leaf mAP50-95 against **0.937**.

## 5. Test and predict

```bash
python testing/test.py
python testing/predict.py --source path/to/photo.jpg
python testing/predict.py --model yolo26m --source path/to/photo.jpg
```

## Order of commands (Plan A)

```bash
source .venv/bin/activate
python dataset/labeling/label_from_zip.py   # if starting fresh
python dataset/labeling/check_labels.py
python dataset/labeling/relabel_with_teacher.py --mode hybrid --backend sam
python dataset/labeling/check_labels.py
python dataset/combined/build_coco128_leaf.py
python training/train_yolo26s.py
python training/train_yolo26m.py
python testing/test.py
```

Run the check after relabel. Rebuild the combined dataset after any labeling change so class 80 stays in sync.
