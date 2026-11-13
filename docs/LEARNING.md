# Learning this project

This file is the teaching guide. Commands, folder layout, and the measured run live in [PROJECT.md](PROJECT.md). Read this first if you want to know what the pieces mean, then use `PROJECT.md` when you actually run them.

One fact to keep in front of every table below: this repository has fine-tuned **one** checkpoint, `yolo26s.pt`. The leaf scores in this file are that run. YOLOv8 and YOLO11 appear because they are the natural comparison set on the same Ultralytics API. They have not been trained on these leaves, so there is no winner among them yet.

## 1. The task

Each photo is one pressed leaf on colored paper. The job is to draw a rectangle around that leaf and name the rectangle `leaf`.

That job is **object detection**. A detector answers two questions at once: where is the object, and what class is it. A classifier would only answer “this whole photo is a leaf,” which these photos already are. Species names such as Pinus or Quercus are used when the photos are split into train and validation. The model is not asked to name the species.

A useful detector here finds the specimen, keeps the leaflets of a compound leaf inside one box, and leaves the colored paper outside the box. Thin needles (Pinus) and thin lobes (Papaver) are the photos where a loose box is easy to spot by eye.

## 2. The pipeline, in the order you run it

```text
leaf.zip
  -> boxes from the mask or the paper color
  -> a check that every box is valid
  -> COCO128 photos plus the leaf photos, leaf rewritten to class 80
  -> fine-tune yolo26s.pt
  -> score the held-out leaf photos, then the held-out COCO photos
  -> draw boxes on a new photo
```

### Label

`dataset/labeling/label_from_zip.py` builds the boxes. It does not call YOLO.

| Photos | How the box is made |
| --- | --- |
| 340 photos that have a file in `dataset/raw/leaf/BW/` | The black-and-white file is a mask. The box is the extent of the white pixels |
| 103 photos with no mask | The paper color is measured from the image border. The box surrounds every pixel that is not that paper |

MobileSAM (`mobile_sam.pt`) can draw boxes on the unmasked photos if you pass `--method sam`. On this colored-paper set it is slower, and the border method is already tight enough. YOLO is the model you train afterward.

The script writes a leaf-only dataset:

| Output | What it is |
| --- | --- |
| `dataset/labeled/leaf/images/train` | 358 photos |
| `dataset/labeled/leaf/images/val` | 85 photos. Every species is in both splits |
| `dataset/labeled/leaf/labels/` | one `.txt` file per photo |
| `dataset/labeled/leaf/preview/` | the same photos with the box drawn, for your eyes |
| `dataset/labeled/leaf/data.yaml` | one class, index `0`, name `leaf` |

A label line looks like this:

```text
0 0.597222 0.427083 0.391667 0.431250
```

The first number is the class. The next four are the box center x, center y, width, and height, each divided by the image size so they fall between 0 and 1. Training reads `labels/`. The green rectangles in `preview/` are only for checking.

`dataset/labeling/check_labels.py` then fails the set if an image has no box, the box leaves the photo, the box is a speck, or the box covers the whole frame. Open `preview/` as well. A good box touches the outer pixels of the specimen.

### Build the training set

`dataset/combined/build_coco128_leaf.py` downloads COCO128 (128 everyday photos, about 7 MB) and builds `dataset/combined/coco128_leaf/`, which is the folder training reads.

COCO is a large public detection set with 80 classes (`person`, `car`, `dog`, and so on). COCO128 is a 128-image slice of it. The starting checkpoint was trained by Ultralytics on the **full** COCO set. This project only continues training on the small slice plus the leaves, which is enough to run on one machine and is the set this project was asked to use. It is not large enough to keep all 80 COCO classes reliable after fine-tuning. The class this run is built to learn is **leaf**.

Three details matter:

- Leaf photos are **copied** into `coco128_leaf/images/`. A symlink would be resolved back to `labeled/leaf/`, and Ultralytics would then read the leaf-only label files beside those photos. Those files still say class `0`. In the 81-class set, class `0` is `person`, so the leaves would be taught as people.
- Leaf boxes are rewritten from class `0` to class **`80`**. Classes `0`–`79` keep the COCO names. Ultralytics matches the pretrained classification head by name, so the `person` row stays `person` when `leaf` is added at the end.
- 80% of COCO128 (102 photos) is listed with the 358 leaf training photos: 460 train rows. The other 26 COCO photos and the 85 leaf validation photos are held out for testing.

### Train, test, predict

`training/train.py` loads `yolo26s.pt` and fine-tunes it. Weights and the curve go to `training/runs/coco128-leaf/`. The file to keep is `weights/best.pt`.

`testing/test.py` scores that file twice: once on `leaf_val.yaml` (the 85 leaf photos) and once on `coco128_val.yaml` (the 26 held-out COCO photos). The leaf row is the leaf-detection result.

`testing/predict.py` draws boxes on a new photo and writes them under `testing/runs/predict/`. In this checkpoint the leaf class is still class **80**, and the printed name is `leaf`.

## 3. What YOLO is

YOLO (You Only Look Once) is a **one-stage** detector. One forward pass through the network looks at the whole image and proposes boxes. A two-stage detector would first search for candidate regions and then classify each region. One stage is why these models are fast enough for video and for a single desktop GPU.

For every candidate box the network predicts:

- where the box sits (center x, center y, width, height)
- how confident it is that an object is there
- a score for each class

After training, low-confidence boxes are dropped. Overlapping boxes for the same object are usually merged by non-maximum suppression (NMS): keep the strongest box and discard the weaker ones that cover almost the same pixels. YOLO26 can also run an end-to-end head that emits boxes without that extra NMS step. The default path still uses NMS. This project calls the normal Ultralytics `predict` and `val` paths, so it uses that default.

The network itself has three parts you will see named in training logs:

| Part | Role |
| --- | --- |
| Backbone | Early layers that find edges, texture, and shape. These are the features COCO already taught |
| Neck | Mixes fine detail from early layers with the broader view from later layers, so a box can be both tight and aware of the whole leaf |
| Head | Turns those features into boxes and class scores. This is what has to learn the new name `leaf` |

You do not train that from random weights. `train.py` refuses a `.yaml` architecture file for that reason. It starts from a `.pt` checkpoint Ultralytics already trained on full COCO, then continues training on your 460 rows. That continuation is **supervised fine-tuning**: every training photo already has a correct box, and the loss pushes the network’s boxes toward those labels.

The leaf set is 443 photos of one specimen on paper. The backbone already knows edges and texture. Fine-tuning mostly has to attach the new class and adjust the boxes to this studio look.

## 4. Why the training recipe looks like this

`training/train.py` uses the same settings on every run so a later comparison measures the model, not a changed recipe.

| Choice | What it does here |
| --- | --- |
| Start from `yolo26s.pt` | COCO features are already in the file. The small size matches a 443-photo studio set. Section 6 is the full size argument |
| Leaf as class `80` | The 80 COCO head rows stay matched by name |
| `--freeze 10` | The first 10 layers stay as COCO learned them. The neck and head learn `leaf` |
| AdamW at `lr0=0.001` | A small step. The default COCO learning rate would drag the pretrained weights too far on this set |
| Mild augmentation | Rotate up to 10°, vertical flip 0.2, horizontal flip 0.5, mosaic 0.8, mosaic off for the last 10 epochs. These photos are one leaf on paper, so strong augmentation would invent scenes the set does not contain |
| Early stop, patience 20 | Training stops when mixed validation mAP50-95 has not improved for 20 epochs. `best.pt` is that best epoch, not the last epoch |
| `imgsz` 640, `batch` 16 | 640 is the size the COCO checkpoints were compared at. Batch 16 fits an RTX 3060 12 GB for the small model. Use `--batch 8` if the card runs out of memory |

Augmentation here means randomly changing a training photo so the same leaf is seen rotated, flipped, or pieced into a mosaic with other photos. The label box is moved with the pixels. The model then has to find the leaf in more than one pose.

### What this machine measured

| Item | Value |
| --- | --- |
| Hardware | NVIDIA GeForce RTX 3060, 12 GB |
| Stack | Python 3.12, PyTorch 2.15 + CUDA, Ultralytics 8.4 |
| Train rows | 460 (102 COCO128 + 358 leaf) |
| Epochs run | 21, then early stop |
| Checkpoint kept | `training/runs/coco128-leaf/weights/best.pt`, the mixed-val peak at **epoch 1** |

An epoch is one pass through the training rows. Early stopping watched the **mixed** COCO+leaf validation score. Epoch 1 won that mixed score, so that checkpoint was saved even though training continued to epoch 21. Leaf-only testing of that file is the number that matters for this project. Mixed mAP can fall because the 80 COCO classes get noisy on 128 photos, while the leaf class is already in good shape.

| Set | Images | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| Leaf validation | 85 | 0.893 | 0.879 | 0.930 | 0.528 |
| Held-out COCO128 | 26 | 0.523 | 0.447 | 0.484 | 0.260 |

Section 7 explains how to read those columns. A check on a held-out Quercus photo predicted `leaf` at about 0.9 confidence.

## 5. Three generations, one API

Ultralytics ships several YOLO generations. You load any of them with the same call:

```python
from ultralytics import YOLO
model = YOLO("yolo26s.pt")  # or yolov8s.pt, or yolo11s.pt
```

The filename is the generation and the size together. YOLO11 and YOLO26 have **no `v`** in the filename. YOLOv8 still has one: `yolov8s.pt`, `yolo11s.pt`, `yolo26s.pt`.

| Generation | Released | Checkpoint this project would load | What changed for a user of this repo |
| --- | --- | --- | --- |
| YOLOv8 | January 2023 | `yolov8s.pt`, `yolov8m.pt` | The widely documented Ultralytics detector. Same `train` / `val` / `predict` calls |
| YOLO11 | 2024 | `yolo11s.pt`, `yolo11m.pt` | Higher COCO accuracy than YOLOv8 at a similar or smaller size |
| YOLO26 | January 2026 | `yolo26s.pt`, `yolo26m.pt` | Current Ultralytics detector. Higher COCO accuracy again, a lighter head, and an optional NMS-free end-to-end head. The nano model is reported up to about 43% faster on CPU ONNX than YOLO11n |

YOLO26’s own training recipe (a MuSGD optimizer, a different loss schedule) is how **Ultralytics** produced `yolo26s.pt`. This project does not repeat that recipe. It loads the finished file and fine-tunes with AdamW, frozen early layers, and the settings in `train.py`.

This repo uses YOLO26s for a simple reason: `requirements.txt` asks for `ultralytics>=8.4`, and YOLO26s is the current small COCO checkpoint in that stack. The backbone already matches the fine-tune path (add `leaf` as class 80, freeze 10 layers). There was no side-by-side training of YOLOv8, YOLO11, and YOLO26 on these leaves. A later plan listed six fine-tunes, `s` and `m` of each generation, and that plan is still pending in the code. Until those runs exist, COCO tables below describe the **starting checkpoints**, and the leaf table in section 4 describes the **only finished model**.

## 6. Five sizes, and why this project starts at `s`

Each generation comes in five sizes. The design is the same. Width and depth change, so the file has more or fewer parameters (numbers the optimizer can adjust) and more or fewer FLOPs (roughly, how much math one image costs).

| Suffix | Name | When it is the right first try |
| --- | --- | --- |
| `n` | nano | A phone, a small board, or a CPU that must stay quick |
| `s` | small | A single desktop GPU and a small, simple dataset. This project |
| `m` | medium | The same GPU when `s` finds the object and the boxes are still loose |
| `l` | large | A large, varied dataset, or offline jobs where accuracy outranks speed |
| `x` | extra large | The last step when `l` is still short and you can pay the compute |

### Official COCO scores for the starting checkpoints

These numbers are **not** leaf scores. Ultralytics measured them on the full COCO validation set at 640 pixels, before any leaf fine-tune. Source pages:

- [YOLO26](https://docs.ultralytics.com/models/yolo26)
- [YOLO11](https://docs.ultralytics.com/models/yolo11)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8)

YOLO26 detection, the family this project starts from:

| Model | mAP50-95 on COCO | Parameters (millions) | FLOPs (billions) | CPU ONNX (ms) | T4 TensorRT (ms) |
| --- | --- | --- | --- | --- | --- |
| `yolo26n` | 40.9 | 2.4 | 5.5 | 38.9 | 1.7 |
| `yolo26s` | 48.6 | 9.5 | 20.9 | 87.2 | 2.5 |
| `yolo26m` | 53.1 | 20.4 | 68.4 | 220.0 | 4.7 |
| `yolo26l` | 55.0 | 24.8 | 86.8 | 286.2 | 6.2 |
| `yolo26x` | 57.5 | 55.7 | 194.4 | 525.8 | 11.8 |

Parameter and FLOP counts are for the fused model. Speed is the NMS-free head. The `.pt` file you fine-tune still contains the full training architecture.

The same COCO mAP50-95 at the two sizes worth comparing here:

| Size | YOLOv8 | YOLO11 | YOLO26 |
| --- | --- | --- | --- |
| small | `yolov8s` 44.9 (11.2M params, 28.6B FLOPs) | `yolo11s` 47.0 (9.4M, 21.6B) | `yolo26s` 48.6 (9.5M, 20.9B) |
| medium | `yolov8m` 50.2 (25.9M, 78.9B) | `yolo11m` 51.5 (20.1M, 68.1B) | `yolo26m` 53.1 (20.4M, 68.4B) |

YOLOv8 speed in the official table was measured on different hardware (A100 TensorRT) than the YOLO11 and YOLO26 T4 numbers, so those milliseconds are not a fair race. Compare speed by timing the checkpoints yourself, on this 3060, with the same image size.

On COCO, the accuracy steps for YOLO26 get smaller as the model grows: n to s is +7.7 mAP, s to m is +4.5, m to l is +1.9, l to x is +2.5. The compute jumps the other way. s to m is about three times the FLOPs. COCO has 80 varied classes. A single leaf on paper is an easier visual problem, so a small model often lands close to a medium one.

### Why `s` is the checkpoint in `train.py`

The leaf training split is 358 photos, all one specimen on studio paper. A larger network has more spare capacity. Spare capacity on a set this narrow tends to memorize the paper color and the lighting instead of learning a leaf outline that survives a new photo. `yolo26s` is large enough to use the COCO features and small enough to fine-tune on a 12 GB 3060 at batch 16.

`yolo26m` is the next experiment, with the same freeze, learning rate, and augmentation. If the card runs out of memory, rerun that one model at `--batch 8` and write the batch in the results row. `n` is the experiment when you care about CPU speed more than the last bit of box tightness. `l` and `x` are a poor first step on 443 studio photos: the COCO gain from m to l is small, and the extra weights have little new leaf appearance to learn from.

## 7. How to read a score, and how to decide

`testing/test.py` prints a row per class. For this project, read the **`leaf` row**. The `all` row during training mixes held-out COCO with leaves, so it is the early-stopping signal, not the leaf score.

Walk one prediction before the averages:

1. The model emits a box and a **confidence**, a number from 0 to 1. `predict.py` defaults to keeping boxes at or above 0.25.
2. **IoU** (intersection over union) is the overlap between that box and the true box. Intersection area divided by the area of the two boxes combined. 1.0 is a perfect overlap. 0.5 is a loose but recognizable overlap.
3. A detection counts as a **true positive** at IoU 0.5 when the class is right and the overlap is at least 0.5. A box on paper, or a second box on a leaf that already has a match, is a false positive. A leaf with no match is a false negative.
4. **Precision** is the fraction of kept boxes that were right. 0.893 on the leaf val set means about 89% of the leaf boxes the model kept were correct matches.
5. **Recall** is the fraction of true leaves that were found. 0.879 means about 88% of the 85 validation leaves got a match.
6. Precision and recall move together as you slide the confidence cutoff. Plotting them gives a precision-recall curve. **Average precision (AP)** is the area under that curve for one class. **mAP** is the mean of those areas across classes. With one class, the leaf AP and the leaf mAP are the same number.

Two mAP columns, two questions:

| Metric | Question it answers | This leaf run |
| --- | --- | --- |
| mAP50 | Did you find the leaf with a box that overlaps the truth by at least 50%? | 0.930 |
| mAP50-95 | Is the box still a match when the required overlap rises from 0.50 to 0.95 in steps of 0.05, averaged? | 0.528 |

mAP50 of 0.930 with mAP50-95 of 0.528 means the model usually finds the leaf, and the box is only moderately tight. The gap between those two numbers is the interesting one. A later run that lifts mAP50 from 0.93 to 0.95 but leaves mAP50-95 at 0.53 has mostly learned to find a leaf it already finds. A run that lifts mAP50-95 has learned a tighter box.

### Picking a better model

Compare two checkpoints only when the val photos, the image size, and the test script are the same. Then use this order:

1. **Leaf mAP50-95.** This is the primary score. Higher is a better leaf model on this set.
2. **Leaf mAP50.** Use it to see whether a drop in mAP50-95 is “missed the leaf” or “found it, loose box.”
3. **Precision and recall.** A model can raise precision by staying quiet and missing leaves, or raise recall by drawing extra boxes. Read them as a pair.
4. **The pictures.** Open `testing/runs/` and a few `predict.py` images, especially Pinus and Papaver. A metric can rise while a thin leaf grows a box that includes a wide margin of paper.
5. **Speed and memory on this machine.** Record seconds per validation pass, or images per second, and the batch that fit. A medium model that gains 0.01 mAP50-95 and no longer fits the 3060 is a worse tool for this project.
6. **Held-out COCO128 mAP.** This says whether `person` and the other COCO classes still fire on photos the trainer did not use. It is a forgetting check. It is not the leaf score, and 26 photos cannot prove the 80 classes are healthy.

Do not rank models by the mixed validation number that early stopping watched. On the finished run that number peaked at epoch 1, while the leaf-only test of the saved file is the result quoted above.

COCO mAP from section 6 does not transfer as a leaf ranking. It only tells you which file is the stronger pretrained start. A newer, larger COCO model can still do worse on 358 studio leaves if it memorizes the paper.

## 8. How to experiment

Change one thing at a time. Otherwise a higher score has no cause. Keep a row for every run:

| Column | What to write down |
| --- | --- |
| Model file | `yolo26s.pt`, `yolo26m.pt`, `yolov8s.pt`, … |
| What changed | The single knob, or “baseline” |
| Epochs run | The log, not the `--epochs` cap |
| Best epoch | The epoch number stored in `best.pt` |
| Batch and image size | Especially if you had to drop the batch |
| Leaf P, R, mAP50, mAP50-95 | From `testing/test.py` |
| COCO128 mAP50, mAP50-95 | The forgetting check |
| What the pictures showed | Tight box, paper included, missed needles, extra boxes |

`train.py` already exposes the knobs. Give each run its own `--name` so it does not collide with `training/runs/coco128-leaf/`.

### First comparison: size, same generation

Train nano and medium with the recipe unchanged. Score each `best.pt` by passing `--weights`.

```bash
python training/train.py --model yolo26n.pt --name yolo26n
python training/train.py --model yolo26m.pt --name yolo26m

python testing/test.py --weights training/runs/yolo26n/weights/best.pt
python testing/test.py --weights training/runs/yolo26m/weights/best.pt
```

`test.py` writes both evaluations under `testing/runs/leaf` and `testing/runs/coco128` and will overwrite those folders on the next call. Copy the printed metrics into your table, or copy the run folder aside, before you start the next model. If `yolo26m` runs out of memory:

```bash
python training/train.py --model yolo26m.pt --name yolo26m --batch 8
```

Put the leaf mAP50-95 of `yolo26n` and `yolo26m` next to the existing `yolo26s` numbers (0.930 / 0.528). The size that wins is the one with the higher leaf mAP50-95 whose pictures still hug the specimen and whose batch still fits the GPU.

### Second comparison: generation, same size

Once size is settled, change only the starting file. For the small size:

```bash
python training/train.py --model yolov8s.pt --name yolov8s
python training/train.py --model yolo11s.pt --name yolo11s
python testing/test.py --weights training/runs/yolov8s/weights/best.pt
python testing/test.py --weights training/runs/yolo11s/weights/best.pt
```

The existing `coco128-leaf` run is the YOLO26s arm of that comparison. Keep its leaf row. The generation with the higher leaf mAP50-95, at the same batch and image size, is the better starting checkpoint **for these leaves**. A higher official COCO mAP does not settle that question by itself.

Medium versions, if you want the six-model grid, are `yolov8m.pt`, `yolo11m.pt`, and `yolo26m.pt` with the same flags.

### Third experiment: unfreeze, smaller step

The script’s help text already points at a second stage. After a frozen run looks sane, train further with the backbone unlocked and a smaller learning rate, from the `best.pt` you just made:

```bash
python training/train.py \
  --model training/runs/coco128-leaf/weights/best.pt \
  --name yolo26s-unfreeze \
  --freeze 0 \
  --lr0 0.0003
```

Use a new `--name`. Judge it with the leaf row again. If mAP50-95 rises and the previews stay tight, the backbone had something left to adapt. If the score rises on the 85 val photos but new photos get worse, the unlocked network memorized the studio set. Val photos were held out of training, and they are still the same paper and the same camera. A photo from a different room is a stronger check. `predict.py` is how you look:

```bash
python testing/predict.py --model training/runs/yolo26s-unfreeze/weights/best.pt --source path/to/photo.jpg
```

### Knobs worth touching only after those three

Each of these is its own run, still compared on leaf mAP50-95 and on the pictures:

| Knob | Flag | What you are asking |
| --- | --- | --- |
| Image size | `--imgsz 768` or `512` | Do the thin needles need more pixels, or is 640 already enough? |
| Freeze depth | `--freeze 0` on a fresh COCO checkpoint, versus `--freeze 10` | Did the frozen backbone help, or was it holding the boxes back? |
| Learning rate | `--lr0 0.0003` on a frozen run | Was 0.001 already as gentle as this set needs? |
| Epoch cap | `--epochs 40` | Does patience 20 always stop you first anyway? |

Leave mosaic, flips, and rotation alone until a size and a generation have a clean pair of scores. Changing augmentation at the same time makes the table unreadable.

### A small loop you can repeat

1. Pick one question (“does medium beat small on leaf mAP50-95?”).
2. Train with one flag changed and a fresh `--name`.
3. Run `testing/test.py --weights .../best.pt` and copy both rows.
4. Draw boxes on two easy leaves and two thin ones.
5. Write the row, including the best epoch and the batch.
6. Keep the checkpoint only if leaf mAP50-95 is up and the boxes still leave the paper outside.

That loop is the whole decision procedure. The official COCO table tells you which files are worth putting into it. The leaf validation table tells you which file learned this dataset.
