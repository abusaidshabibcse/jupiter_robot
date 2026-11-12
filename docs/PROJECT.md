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

## 1. Label the zip

From the project root, with the virtual environment active:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
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

- Leaf boxes are rewritten from class `0` to class **`80`**. Classes `0`–`79` stay the COCO names, so a COCO label such as `person` is not overwritten by `leaf`.
- 80% of COCO128 is listed in `train.txt` together with the leaf training photos.
- 20% of COCO128 is held out, and the leaf validation photos are held out. Those are what testing uses.
- `data.yaml` is the file `training/train.py` reads. It has 81 names. The last one is `leaf`.

COCO128 is a small slice of COCO. It is the set this project trains on, as requested. It is large enough to run the combined pipeline on one machine. It is not large enough to make the 80 COCO classes reliable after fine-tuning. The class this run is built to learn is **leaf**.

## 4. Train

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

| Setting | Why |
| --- | --- |
| `yolo26s.pt` | Smallest model that still has strong COCO features. 443 leaf photos will overfit a larger model |
| `--freeze 10` | The backbone stays as COCO learned it. The head learns the leaf class |
| `AdamW`, learning rate `0.001` | A small fine-tune. The default COCO learning rate is too high here |
| 80 epochs, patience 20 | Stops when validation has not improved for 20 epochs |
| device | GPU `0` by default. If you have no GPU: `--device cpu --batch 8` |

Weights and the curve are written to:

```
training/runs/coco128-leaf/weights/best.pt
training/runs/coco128-leaf/results.csv
```

`best.pt` is the epoch with the highest validation mAP50-95. That is the file testing uses.

## 5. Test

```bash
python testing/test.py
```

This scores `best.pt` twice and writes both runs under `testing/runs/`:

| Run | Data | What a good result means |
| --- | --- | --- |
| `testing/runs/leaf/` | the 85 leaf validation photos | The row named `leaf` is high. This is the leaf-detection result |
| `testing/runs/coco128/` | the held-out COCO128 photos | The original COCO classes on images the trainer did not use |

To draw boxes on a new photo:

```bash
python testing/predict.py --source path/to/photo.jpg
```

Boxed images go to `testing/runs/predict/`. The leaf class is class 80 in this checkpoint, and the printed name is `leaf`.

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

Run the check before the build. If the check fails, fix the labels and run the check again. Then rebuild the combined dataset so class 80 stays in sync with the leaf labels.
