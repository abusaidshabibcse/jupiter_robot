# Leaf detection project

This project turns `leaf.zip` into a YOLO dataset, checks the boxes, then fine-tunes YOLOv8, YOLO11, and YOLO26 in the **s** and **m** sizes on **COCO128 plus the leaf photos**. The six runs share one recipe, so the scores compare the architectures.

Each checkpoint keeps its own run folder. Ultralytics weight names (YOLO11 has no `v` in the filename):

| Run folder | Base weights |
| --- | --- |
| `training/runs/yolov8s/` | `yolov8s.pt` |
| `training/runs/yolov8m/` | `yolov8m.pt` |
| `training/runs/yolo11s/` | `yolo11s.pt` |
| `training/runs/yolo11m/` | `yolo11m.pt` |
| `training/runs/yolo26s/` | `yolo26s.pt` |
| `training/runs/yolo26m/` | `yolo26m.pt` |

No nano, large, or extra-large sizes. No other YOLO generations. The labeling step does not use these models.

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
    train.py                      fine-tune the six s and m checkpoints
    runs/<model>/                 one folder per trained checkpoint
  testing/
    test.py                       scores leaf val and held-out COCO128
    predict.py                    draws boxes on new photos
    runs/<model>/leaf             Ultralytics leaf plots per model
    runs/<model>/coco128          Ultralytics COCO128 plots per model
  results/
    summary.md                    comparison table for the report
    <model>/metrics.md            one-model write-up
```

Dataset work stays inside `dataset/`. Training stays inside `training/`. Testing stays inside `testing/`. Report files stay inside `results/`.

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

The six YOLO checkpoints are the **training** models, used in the next sections. They are not the labeler.

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

This is **supervised fine-tuning** of a pretrained detector, not training from scratch. Every checkpoint uses the same settings, so a higher score means that architecture fit this set better.

| Choice | Why it helps |
| --- | --- |
| YOLOv8, YOLO11, and YOLO26, sizes s and m | Ultralytics already trained each checkpoint on full COCO. The backbone already knows edges, shapes, and texture. Separate run folders keep the comparison fair |
| Leaf as class `80` | Classes `0`–`79` keep the COCO names. Ultralytics remaps the pretrained classification head by name, so the `person` row is not thrown away when `leaf` is added |
| `--freeze 10` | The early backbone stays as COCO learned it. The neck and head learn the new class |
| AdamW at `0.001` | A small step. The default COCO learning rate would move the pretrained weights too far on this set |
| 80 epochs, patience 20, image size 640, batch 16 | Same budget for every run. On the 12 GB GPU, an m model that runs out of memory is trained again at batch 8, and that batch is noted in the results table |
| Mild augmentation | Rotate up to 10°, vertical flip 0.2, horizontal flip 0.5, mosaic 0.8, mosaic off for the last 10 epochs. The photos are one leaf on colored paper, so heavy augmentation would invent scenes this set does not contain |
| Early stop, patience 20 | Stops when the mixed validation mAP50-95 has not improved for 20 epochs. `best.pt` is that best epoch, not the last epoch |

The number that says leaf detection worked is the **`leaf` row** from `testing/test.py`, not the mixed `all` row during training. The mixed validation set mixes held-out COCO128 with leaf photos, so COCO class noise can stop training early even when leaf is already strong.

### Commands

```bash
python training/train.py
```

That one command trains the six checkpoints and refuses a `.yaml` or any other model name. Weights land in:

- `training/runs/yolov8s/`, `training/runs/yolov8m/`
- `training/runs/yolo11s/`, `training/runs/yolo11m/`
- `training/runs/yolo26s/`, `training/runs/yolo26m/`

To repeat a single checkpoint, pass its name:

```bash
python training/train.py --model yolo26m.pt
```

Use GPU `0` by default. If an m model runs out of memory on the 12 GB GPU, `train.py` retries that model at batch 8. The same retry by hand is:

```bash
python training/train.py --model yolov8m.pt --batch 8
```

### Measured runs (this machine)

Hardware: NVIDIA GeForce RTX 3060 (12 GB). Stack: Python 3.12, PyTorch 2.15 + CUDA, Ultralytics 8.4. Train rows: 460. Recipe: freeze 10, AdamW `0.001`, 80 epochs, patience 20, image size 640, batch 16 unless a row notes otherwise. The leaf columns are the performance numbers for this project.

| Model | Leaf precision | Leaf recall | Leaf mAP50 | Leaf mAP50-95 | COCO128 mAP50 | COCO128 mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8s | 0.988 | 0.972 | 0.983 | 0.741 | 0.334 | 0.183 |
| yolov8m | 0.996 | 0.988 | 0.993 | 0.814 | 0.319 | 0.183 |
| yolo11s | 0.977 | 0.991 | 0.994 | 0.872 | 0.265 | 0.174 |
| yolo11m | 0.967 | 0.918 | 0.973 | 0.817 | 0.278 | 0.150 |
| yolo26s | 0.893 | 0.879 | 0.930 | 0.528 | 0.484 | 0.260 |
| yolo26m | 0.963 | 0.988 | 0.992 | 0.753 | 0.359 | 0.199 |

All six finished at batch 16.

Early stopping watched the mixed COCO+leaf validation score. Leaf-only testing above is the score that matters for this project. Full per-model write-ups: [`results/summary.md`](../results/summary.md).

## 5. Test and predict

Score all six checkpoints:

```bash
python testing/test.py
```

Each `weights/best.pt` is scored on the 85 leaf validation photos and on the 26 held-out COCO128 photos. Plots go to `testing/runs/<model>/leaf` and `testing/runs/<model>/coco128`, so the six runs do not overwrite each other. The table in section 4 is that measurement.

`predict.py` draws boxes for one checkpoint. The default weights are `training/runs/yolo26s/weights/best.pt`. `--model` accepts any of the six `best.pt` files.

```bash
python testing/predict.py --source path/to/photo.jpg
python testing/predict.py --model yolov8m --source path/to/photo.jpg
```

Boxed images go to `testing/runs/predict/<model>/`. The leaf class is class **80** and prints as `leaf`.

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
