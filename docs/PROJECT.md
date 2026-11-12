# Leaf detection project

This project turns `leaf.zip` into a YOLO dataset, checks the boxes, then trains YOLO26 on **COCO128 plus the leaf photos** so the model can detect a leaf.

The training model is **YOLO26s** (`yolo26s.pt`). Ultralytics already trained that file on the full COCO dataset. This project fine-tunes it. The labeling step does not use that model.

## Folders

```
jupiter/
  docs/PROJECT.md                 this file
  dataset/                        everything about the data
    raw/leaf.zip                  the zip you start from
    raw/leaf/                     unzipped RGB photos, BW masks, leaf.csv
    labeling/label_from_zip.py    makes the boxes
    labeling/check_labels.py      checks that every box is valid
    labeled/leaf/                 YOLO leaf dataset and the preview images
    combined/coco128/             the 128 COCO images, downloaded once
    combined/coco128_leaf/        the dataset training actually reads
  training/
    train.py                      fine-tune YOLO26s
    runs/                         logs and weights appear here after training
  testing/
    test.py                       scores leaf val and held-out COCO128
    predict.py                    draws boxes on new photos
    runs/                         test outputs appear here
```

Dataset work stays inside `dataset/`. Training stays inside `training/`. Testing stays inside `testing/`.

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

`label_from_zip.py` unzips `dataset/raw/leaf.zip` into `dataset/raw/leaf/` if the photos are not already there. The current machine already has that folder, so the script skips the unzip.

It writes:

| Output | What it is |
| --- | --- |
| `dataset/labeled/leaf/images/train` | 358 photos |
| `dataset/labeled/leaf/images/val` | 85 photos. Every species is in both splits |
| `dataset/labeled/leaf/labels/` | one `.txt` file per photo |
| `dataset/labeled/leaf/preview/` | the same photos with the box drawn in green |
| `dataset/labeled/leaf/data.yaml` | class `0: leaf` |

A label line looks like this:

```
0 0.597222 0.427083 0.391667 0.431250
```

Those four numbers are the box center x, center y, width, and height, each divided by the image size so they fall between 0 and 1.

### Which model labeled the photos

No detection model labeled this zip. Two sources inside the zip were used, because a detector is less accurate on these studio shots than the files you already have.

| Photos | How the box was made | Model |
| --- | --- | --- |
| 340 photos that have a file in `dataset/raw/leaf/BW/` | The BW file is a black-and-white mask. The box is the extent of the white pixels | None |
| 103 photos with no mask (Acer negundo, Taxus, Papaver, Polypodium, Pinus, Fraxinus, Aesculus, Chelidonium, Schinus, Fragaria) | The paper color is measured from the image border. The box surrounds every pixel that is not that paper, so a compound leaf keeps all of its leaflets | None |

MobileSAM is available if you want a neural model on the photos that have no mask. It is slower, and on this colored-paper set it is not tighter than the border method. The checkpoint name is `mobile_sam.pt`.

```bash
python dataset/labeling/label_from_zip.py --method sam --sam-weights mobile_sam.pt
```

YOLO26 is the **training** model, used in the next sections. It is not the labeler.

## 2. Check that labeling worked

Two checks. Do both.

**Numbers.** This reads every label and fails if an image has no box, the box is outside the photo, the box is a speck, or the box covers the whole frame.

```bash
python dataset/labeling/check_labels.py
```

A good run prints `status: PASS` and writes the same text to `dataset/labeled/leaf/LABEL_CHECK.txt`. `status: FAIL` means do not train yet. The report names the bad files.

**Eyes.** Open `dataset/labeled/leaf/preview/`. A good box touches the outer pixels of the specimen and leaves the paper outside the rectangle. Look especially at Pinus (thin needles) and Papaver (thin lobes). If a box is wrong, edit the matching file in `dataset/labeled/leaf/labels/` and run the check again. Training reads `labels/`, not `preview/`.

## 3. Build the COCO128 + leaf training set

```bash
python dataset/combined/build_coco128_leaf.py
```

The first run downloads COCO128 (128 images, about 7 MB) into `dataset/combined/coco128/`. The script then builds `dataset/combined/coco128_leaf/`:

- Leaf photos are **copied** into `coco128_leaf/images/`. Symlinks are not used. Ultralytics resolves a symlink to the real leaf path and then reads the leaf-only labels next to that path, which still say class `0`. In the 81-class set, class `0` is `person`. A real copy keeps training on the rewritten class-`80` labels.
- Leaf boxes are rewritten from class `0` to class **`80`**. Classes `0`–`79` stay the COCO names, so a COCO label such as `person` is not overwritten by `leaf`.
- 80% of COCO128 is listed in `train.txt` together with the leaf training photos (102 COCO + 358 leaf = 460 train rows).
- 20% of COCO128 is held out (26 photos), and the leaf validation photos are held out (85 photos). Those are what testing uses.
- `data.yaml` is the file `training/train.py` reads. It has 81 names. The last one is `leaf`.

COCO128 is a small slice of COCO. It is the set this project trains on, as requested. It is large enough to run the combined pipeline on one machine. It is not large enough to make the 80 COCO classes reliable after fine-tuning. The class this run is built to learn is **leaf**.

## 4. Train

### Technique

This is **supervised fine-tuning** of a pretrained detector, not training from scratch.

| Choice | Why it helps |
| --- | --- |
| Start from `yolo26s.pt` | Ultralytics already trained it on full COCO. The backbone already knows edges, shapes, and texture. The leaf set is 443 photos, so a larger model would memorize the studio paper |
| Leaf as class `80` | Classes `0`–`79` keep the COCO names. Ultralytics remaps the pretrained classification head by name, so the `person` row is not thrown away when `leaf` is added |
| `--freeze 10` | The early backbone stays as COCO learned it. The neck and head learn the new class |
| AdamW at `0.001` | A small step. The default COCO learning rate would move the pretrained weights too far on this set |
| Mild augmentation | Rotate up to 10°, vertical flip 0.2, horizontal flip 0.5, mosaic 0.8, mosaic off for the last 10 epochs. The photos are one leaf on colored paper, so heavy augmentation would invent scenes this set does not contain |
| Early stop, patience 20 | Stops when the mixed validation mAP50-95 has not improved for 20 epochs. `best.pt` is that best epoch, not the last epoch |

The number that says leaf detection worked is the **`leaf` row** from `testing/test.py`, not the mixed `all` row during training. The mixed validation set mixes held-out COCO128 with leaf photos, so COCO class noise can stop training early even when leaf is already strong.

### Command

```bash
python training/train.py
```

That is the same as:

```bash
python training/train.py \
  --data dataset/combined/coco128_leaf/data.yaml \
  --model yolo26s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --freeze 10 \
  --name coco128-leaf
```

Use GPU `0` by default. If the 3060 runs out of memory: `--batch 8`. If you have no GPU: `--device cpu --batch 8`.

Weights and the curve are written to:

```
training/runs/coco128-leaf/weights/best.pt
training/runs/coco128-leaf/results.csv
```

### Measured run (this machine)

| Item | Value |
| --- | --- |
| Hardware | NVIDIA GeForce RTX 3060, 12 GB |
| Stack | Python 3.12, PyTorch 2.15 + CUDA, Ultralytics 8.4 |
| Train rows | 460 (102 COCO128 + 358 leaf) |
| Epochs run | 21 (early stop; best checkpoint at epoch 1 on mixed val mAP50-95) |
| Checkpoint | `training/runs/coco128-leaf/weights/best.pt` |

Early stopping watched the mixed COCO+leaf validation score. Epoch 1 had the best mixed mAP50-95, so that checkpoint was kept. Leaf-only testing below is the score that matters for this project.

## 5. Test

```bash
python testing/test.py
```

This scores `best.pt` twice and writes both runs under `testing/runs/`:

| Run | Data | Measured result |
| --- | --- | --- |
| `testing/runs/leaf/` | 85 leaf validation photos | **leaf** precision 0.893, recall 0.879, **mAP50 0.930**, **mAP50-95 0.528** |
| `testing/runs/coco128/` | 26 held-out COCO128 photos | all-class mAP50 0.484, mAP50-95 0.260 |

The leaf row is the leaf-detection result. The COCO128 score only shows whether the original classes still fire on images the trainer did not use. COCO128 is too small to make those 80 classes reliable after this fine-tune.

To draw boxes on a new photo:

```bash
python testing/predict.py --source path/to/photo.jpg
```

Boxed images go to `testing/runs/predict/`. The leaf class is class **80** in this checkpoint, and the printed name is `leaf`. A check on a held-out Quercus photo predicted `leaf` with confidence about 0.9.

## Order of commands

```bash
source .venv/bin/activate
python dataset/labeling/label_from_zip.py
python dataset/labeling/check_labels.py
python dataset/combined/build_coco128_leaf.py
python training/train.py
python testing/test.py
python testing/predict.py --source path/to/photo.jpg
```

Run the check before the build. If the check fails, fix the labels and run the check again. Then rebuild the combined dataset so class 80 stays in sync with the leaf labels. Rebuild also recopies the leaf photos, which is required after any labeling change.
