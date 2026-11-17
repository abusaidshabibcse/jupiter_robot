# Leaf detection project

This project turns `leaf.zip` into a YOLO dataset, checks the boxes, optionally refines weak labels with a teacher model, then fine-tunes **YOLO26 s** and **YOLO26 m** with **separate recipes** on **COCO128 plus the leaf photos**.

| Run folder | Base weights | Trainer |
| --- | --- | --- |
| `yolo/runs/yolo26s/` | `yolo/weights/yolo26s.pt` | `yolo/train_yolo26s.py` |
| `yolo/runs/yolo26m/` | `yolo/weights/yolo26m.pt` | `yolo/train_yolo26m.py` |

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
  yolo/
    train_yolo26s.py / train_yolo26m.py
    specs/yolo26s.yaml / yolo26m.yaml
    weights/                      COCO starting checkpoints, not pushed
    runs/<model>/                 one folder per trained checkpoint, not pushed
    test.py                       scores leaf val and held-out COCO128
    predict.py                    draws boxes on new photos
    eval/                         plots, not pushed
    results/
      summary.md                  comparison table for the report
      <model>/metrics.md          one-model write-up
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
| `yolo26m` | 100 epochs, patience 45, lr0 `0.0006`, freeze 8, mosaic 0.5 |

```bash
python yolo/train_yolo26s.py
python yolo/train_yolo26m.py
```

If `yolo26m` OOMs, it retries at batch 8 (or pass `--batch 8`).

### Scores after whole-specimen hybrid relabel

Hybrid SAM keeps BW-mask boxes. For photos without a mask it prefers SAM only when the box covers roughly the same area as the paper-color estimate; leaflet-sized SAM masks fall back to color (`color-over-sam`).

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26s | 0.988 | 0.988 | 0.984 | 0.946 | 0.376 | 0.207 |
| yolo26m | 0.976 | 0.988 | 0.981 | 0.924 | 0.350 | 0.196 |

Earlier run with broken SAM leaflet labels reported leaf mAP50-95 **0.937** for both, but compound-leaf previews were wrong.

## 5. Test and predict

```bash
python yolo/test.py
python yolo/predict.py --source path/to/photo.jpg
python yolo/predict.py --model yolo26m --source path/to/photo.jpg
```

## Order of commands (Plan A)

```bash
source .venv/bin/activate
python dataset/labeling/label_from_zip.py   # if starting fresh
python dataset/labeling/check_labels.py
python dataset/labeling/relabel_with_teacher.py --mode hybrid --backend sam
python dataset/labeling/check_labels.py
python dataset/combined/build_coco128_leaf.py
python yolo/train_yolo26s.py
python yolo/train_yolo26m.py
python yolo/test.py
```

Run the check after relabel. Rebuild the combined dataset after any labeling change so class 80 stays in sync.
