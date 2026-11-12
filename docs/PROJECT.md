# Leaf detection project

This project turns `leaf.zip` into a YOLO dataset, checks the boxes, then trains several Ultralytics YOLO families on **COCO128 plus the leaf photos** so the model can detect a leaf.

Each family keeps its own run folder. Report numbers are written separately under `results/`.

| Run folder | Base weights | Role |
| --- | --- | --- |
| `training/runs/yolo11n/` | `yolo11n.pt` | Best nano leaf score; good on a 2-core CPU |
| `training/runs/yolo26n/` | `yolo26n.pt` | YOLO26 nano |
| `training/runs/yolov8n/` | `yolov8n.pt` | Classic YOLOv8 nano |
| `training/runs/yolo26s/` | `yolo26s.pt` | Stronger YOLO26 small (GPU) |
| `training/runs/coco128-leaf/` | legacy YOLO26s path | Older clones only |

The labeling step does not use these models.

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
    train.py                      fine-tune yolov8n / yolo11n / yolo26n / yolo26s
    runs/<model>/                 one folder per trained checkpoint
  testing/
    test.py                       scores leaf val and held-out COCO128
    predict.py                    draws boxes on new photos
    runs/<model>-leaf/            Ultralytics leaf plots per model
    runs/<model>-coco128/         Ultralytics COCO128 plots per model
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
| Start from YOLO26 `n` or `s` | Ultralytics already trained both on full COCO. The backbone already knows edges, shapes, and texture. The leaf set is 443 photos, so a medium/large model would memorize the studio paper |
| Train nano variants of v8, v11, and v26 | Same data and recipe, separate run folders, so the report can compare families fairly on a weak PC |
| Prefer `yolo11n` on weak PCs | Currently the strongest leaf score among the nano models, still ~5.5 MB |
| Keep `yolo26s` for quality | Small is stronger when a GPU is available. All checkpoints stay under `training/runs/` |
| Leaf as class `80` | Classes `0`–`79` keep the COCO names. Ultralytics remaps the pretrained classification head by name, so the `person` row is not thrown away when `leaf` is added |
| `--freeze 10` | The early backbone stays as COCO learned it. The neck and head learn the new class |
| AdamW at `0.001` | A small step. The default COCO learning rate would move the pretrained weights too far on this set |
| Mild augmentation | Rotate up to 10°, vertical flip 0.2, horizontal flip 0.5, mosaic 0.8, mosaic off for the last 10 epochs. The photos are one leaf on colored paper, so heavy augmentation would invent scenes this set does not contain |
| Early stop, patience 20 | Stops when the mixed validation mAP50-95 has not improved for 20 epochs. `best.pt` is that best epoch, not the last epoch |

The number that says leaf detection worked is the **`leaf` row** from `testing/test.py`, not the mixed `all` row during training. The mixed validation set mixes held-out COCO128 with leaf photos, so COCO class noise can stop training early even when leaf is already strong.

### Commands

```bash
python training/train.py --model yolov8n.pt
python training/train.py --model yolo11n.pt
python training/train.py --model yolo26n.pt
python training/train.py --model yolo26s.pt
```

`train.py` writes each run under `training/runs/<model-stem>/`.

Typical GPU train flags:

```bash
python training/train.py \
  --data dataset/combined/coco128_leaf/data.yaml \
  --model yolo11n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --freeze 10
```

Use GPU `0` by default. If the 3060 runs out of memory: `--batch 8`. On a 2-core machine that must train locally: `--device cpu --batch 4 --workers 0`.

### Measured runs (this machine)

Hardware: NVIDIA GeForce RTX 3060. Stack: Python 3.12, PyTorch 2.15 + CUDA, Ultralytics 8.4. Train rows: 460. Full report files: [`results/summary.md`](../results/summary.md).

| Model | Checkpoint | Size | Leaf mAP50 | Leaf mAP50-95 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| **yolo11n** | `training/runs/yolo11n/weights/best.pt` | 5.5 MB | **0.980** | **0.813** | Best nano leaf score; use on a 2-core laptop |
| yolo26s | `training/runs/yolo26s/weights/best.pt` | 20.4 MB | 0.930 | 0.528 | Stronger YOLO26 when a GPU is available |
| yolo26n | `training/runs/yolo26n/weights/best.pt` | 5.5 MB | 0.817 | 0.514 | YOLO26 nano |
| yolov8n | `training/runs/yolov8n/weights/best.pt` | 6.5 MB | 0.525 | 0.285 | Classic YOLOv8 nano |

Early stopping watched the mixed COCO+leaf validation score. Leaf-only testing above is the score that matters for this project.

## 5. Test and report files

Score one model:

```bash
python testing/test.py --model yolo11n
```

Rebuild every report folder at once (writes `results/<model>/` plus `results/summary.md`):

```bash
python results/evaluate_models.py
```

| Model | Leaf val (85) | Held-out COCO128 (26) | Report folder |
| --- | --- | --- | --- |
| yolo11n | P 0.987, R 0.965, **mAP50 0.980**, **mAP50-95 0.813** | mAP50 0.298 | `results/yolo11n/` |
| yolo26s | P 0.893, R 0.879, **mAP50 0.930**, **mAP50-95 0.528** | mAP50 0.484 | `results/yolo26s/` |
| yolo26n | P 0.706, R 0.906, **mAP50 0.817**, **mAP50-95 0.514** | mAP50 0.457 | `results/yolo26n/` |
| yolov8n | P 0.462, R 0.906, **mAP50 0.525**, **mAP50-95 0.285** | mAP50 0.354 | `results/yolov8n/` |

The leaf row is the leaf-detection result. The COCO128 score only shows whether the original classes still fire on images the trainer did not use. Ultralytics plots stay under `testing/runs/<model>-leaf/` and `testing/runs/<model>-coco128/`.

### Predict on a 2-core laptop

```bash
python testing/predict.py --model yolo11n --device cpu --imgsz 416 --source path/to/photo.jpg
```

Defaults prefer `yolo11n`, `cpu`, and `imgsz=416`. Boxed images go to `testing/runs/predict/yolo11n/`. The leaf class is class **80** and prints as `leaf`.

On GPU, the larger YOLO26 small model is still available:

```bash
python testing/predict.py --model yolo26s --device 0 --imgsz 640 --source path/to/photo.jpg
```

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
