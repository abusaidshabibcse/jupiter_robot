# Source pack for the report and the slides

Upload this whole `report_pack` folder. It has this file and eight figures. Nothing else is required.

Write two separate deliverables from this file only. Do both. The report is not a substitute for the slides.

1. **Report.** A detailed academic report in plain first-person lab-report voice. Use the section order at the bottom. Keep every number exactly as printed here. Four decimal places in score tables. Section 8.3, “Which model performed best, and why,” must appear in full. Do not bury the winner in a table caption.
2. **PowerPoint.** A 15-slide widescreen (16:9) deck, built from the slide script in section 14. This is a spoken defense, not the report pasted onto slides. Use the on-slide lines as written. Put the named figure on that slide. Put the “Say this” lines in the speaker notes, not on the slide. The winner slide (slide 12) is the visual peak: the number 0.9461 must be the largest text on that slide.

Rules:

- Models in the results are only **YOLO26s** and **YOLO26m**. Do not add YOLOv8, YOLO11, nano, large, or extra-large rows. Those checkpoints were not scored in this run.
- Do not invent scores, epoch counts, hardware, or outdoor tests. If a fact is not in this file, leave it out.
- The ranking number is **leaf mAP50-95** on the 85 held-out leaf photos. It is not the mixed training score and not the number printed on the precision-recall curve.
- Figure files are in `figures/`. Use the captions in the Figures section. Do not redraw them.
- Date the work **17 November 2026**.

## Cover

- University of Jinan
- Computer Science & Technology
- Title: Leaf detection for a solar panel watch
- Subtitle: Fine-tuning YOLO26s and YOLO26m on the UCI Leaf dataset
- Name: Md. Abu Said Shabib
- Student ID: 202624100003
- Teacher: Jeffery Tan
- Date: 17 November 2026

## 1. Abstract

A leaf on a solar panel shades the cells and holds dirt and water. This project builds the detector for a small desktop program that watches a camera and says when a leaf is there.

The photographs are the UCI Leaf set: 443 pressed specimens, 40 species, one box per photo, class name `leaf`. Of those photos, 340 already had a black-and-white mask, so the box is the white pixels. The other 103 had no mask. Their box comes from the paper color at the border, then a check with SAM2.1-L that keeps the box only when it still covers the whole specimen. A checker accepted all 443 boxes. The split is 358 training photos and 85 validation photos, with every species on both sides.

Those leaves were copied into a training set with COCO128. Leaf became class 80 so the 80 COCO classes kept their names. The trainer saw 460 rows (358 leaves + 102 COCO128 photos). The other 26 COCO128 photos were held out as a forgetting check.

Two COCO-pretrained checkpoints were fine-tuned on that set, with separate recipes, for 100 epochs, image size 640, batch 16, on an RTX 3060 12 GB. `best.pt` is the best epoch of **leaf** validation, not the last epoch and not the mixed COCO-plus-leaf score.

On the 85 held-out leaves, YOLO26s reached leaf mAP50-95 **0.9461** in a **20.43 MB** file (best epoch 76). YOLO26m reached **0.9245** in a **44.26 MB** file (best epoch 64). YOLO26s is the model to ship: higher strict score, about half the file size. The detector was not trained on solar-panel photos and does not name the species.

## 2. Introduction

Shade on part of a module cuts the power of that string. The practical goal is a program that watches a camera, draws a box, and speaks a short warning so someone can clean the panel.

Object detection returns a list of rectangles, each with a class name and a confidence between 0 and 1. YOLO (“You Only Look Once”) predicts all boxes in one pass, which is fast enough for a live camera. Both checkpoints start from weights Ultralytics trained on COCO (Common Objects in Context). COCO has 80 everyday classes, including person, car, and dog.

This project does three things. It labels the 443 UCI photographs with one box each. It fine-tunes YOLO26s and YOLO26m on those leaves plus COCO128 and compares them. It describes the camera program that turns a detection into a spoken alert.

It does not do several things a reader might assume. The detector was not trained on photos of solar panels. It does not name the plant species. It does not measure how much power a panel lost.

## 3. Why YOLO26s and YOLO26m

The job has two stages. First the model has to learn a tight box around one pressed leaf. Later the same file has to run on a camera, often on the CPU while the GPU is busy. That argues for a one-stage detector that already knows edges, and for a file small enough to keep.

Training did not start from random weights. A 358-photo studio set cannot teach edges, texture, and shape from nothing. Each run loads a finished COCO checkpoint (a `.pt` file) and continues training. That continuation is supervised fine-tuning: every training photo already has a correct box.

Leaf is added as class **80**, after the 80 COCO names. Ultralytics matches the pretrained classification head by name, so the `person` row stays `person`. If the leaf files had stayed at class 0, the model would have been taught that a leaf is a person.

YOLO26 is the current Ultralytics detector (January 2026). At the same size letter its published COCO accuracy is higher than YOLOv8 and YOLO11, and the head is lighter. Those published COCO numbers are **not** this project’s leaf scores. They only say which file is a stronger pretrained start:

| Starting file | Published COCO mAP | Parameters | FLOPs | Role in this project |
| --- | ---: | ---: | ---: | --- |
| `yolo26s.pt` | 48.6 | 9.5M | 20.9B | The small model. Fits a 12 GB RTX 3060 at batch 16. File after fine-tuning: 20.43 MB |
| `yolo26m.pt` | 53.1 | 20.4M | 68.4B | The medium model. The controlled question is whether extra capacity tightens the leaf box |

Source of the COCO column: Ultralytics YOLO26 model page. Our measured leaf numbers are in section 8.

Why these two, and not the rest of the family:

- **Small** is large enough to use the COCO features and small enough for this set. Spare capacity on 358 studio leaves tends to memorize the paper color and the lighting instead of the leaf outline.
- **Medium** is the next size, trained on purpose as the comparison. The recipe gives it a smaller step (learning rate 0.0006 instead of 0.001), freezes 8 layers instead of 10, and uses less mosaic (0.5 instead of 0.6), so a worse score cannot be blamed on a reckless learning rate.
- **Nano** was not trained. It is the experiment when CPU speed matters more than the last bit of box tightness.
- **Large and extra-large** were not trained. The published COCO gain from medium to large is small, and the extra weights have little new leaf appearance to learn from on 443 photos.
- **YOLOv8 and YOLO11** were not part of this scored run. They are older generations on the same Ultralytics API. This report does not rank them.

The result of that question is section 8.3. YOLO26s won. The published COCO table above is only the reason these two files were worth training. It is not the ranking.

## 4. Hardware and software

Training and every scored test ran on one desktop.

| Part | What it is |
| --- | --- |
| Processor | Intel Core i5-12500 |
| Memory | 32 GB RAM |
| Graphics | NVIDIA GeForce RTX 3060, 12 GB |
| Role of the GPU | Training and the scored tests (image size 640, batch 16) |
| Role of the CPU | The camera program defaults to CPU and image size 416, so it can run when the GPU is busy |

| Piece | Version or name |
| --- | --- |
| Language | Python 3.12 |
| Deep learning | PyTorch 2.15 with CUDA |
| Detector library | Ultralytics 8.4 |
| Label check for unmasked photos | SAM2.1-L (`sam2.1_l.pt`), used only as a labeler, not as the detector |
| Models trained and scored | YOLO26s, YOLO26m |
| Speech in the app design | spd-say (the same line is printed if speech is missing) |

File names have no `v`: `yolo26s.pt`, `yolo26m.pt`. The letter `s` is small and `m` is medium.

## 5. Dataset

### 5.1 UCI Leaf

Source: UCI Machine Learning Repository, Leaf dataset, dataset 288. https://archive.ics.uci.edu/dataset/288/leaf

The archive covers 40 plant species. The folder used here holds **443** photographs. Each photo is one pressed specimen on colored paper. The task is one rectangle, class name `leaf`. The file name still contains the species. The label line does not, so the model is never asked to name the plant.

| Item | Count |
| --- | --- |
| Species | 40 |
| Leaf photographs | 443 |
| With a black-and-white mask | 340 |
| Without a mask | 103 |
| Leaf training photos | 358 |
| Leaf validation photos | 85 |
| Species present in both splits | 40 (none missing from either side) |
| Validation photos per species | 1 to 3 |
| Label check | PASS, 0 problems |

The 103 photos with no mask are Acer negundo, Taxus, Papaver, Polypodium, Pinus, Fraxinus, Aesculus, Chelidonium, Schinus, and Fragaria. Several of these are compound, lobed, or needle-like. A box around one leaflet would be the wrong label.

### 5.2 How the boxes were made

No detection model drew the original boxes.

1. **Mask photos (340).** The zip contains a black-and-white TIFF. The box is the extent of the white pixels. These boxes were kept through every later step.
2. **Unmasked photos (103).** The paper color is measured from the image border. The box surrounds every pixel that is not that paper. A compound leaf keeps its leaflets inside one box. The stem stays inside the box when it is part of the specimen.
3. **Checker.** `check_labels.py` rejects a photo with no box, a box outside the frame, a speck, or a box that covers the whole frame. The run used here printed `status: PASS`, 358 train boxes, 85 val boxes, expected class `0` named `leaf`, 0 problems.
4. **Hybrid check with SAM2.1-L.** SAM was not allowed to replace the 340 mask boxes. On the other photos it was allowed to replace the paper-color box only when the SAM box covered about the same area as the paper-color box. A SAM mask the size of one leaflet was rejected and the paper-color box was kept (`color-over-sam`). If SAM missed the leaf entirely, the fallback was a fine-tuned YOLO26m box, and then the paper-color box. The point of this rule is the whole specimen, not a higher score from a tight box around one leaflet.

An earlier SAM run that boxed individual leaflets scored leaf mAP50-95 **0.937** for both models, and the previews were wrong on compound leaves. That run was discarded. The score in section 8 is the whole-specimen labels, not that 0.937 run.

A label line looks like this:

```text
0 0.597222 0.427083 0.391667 0.431250
```

The first number is the class. The next four are center x, center y, width, and height, each divided by the image size so they fall between 0 and 1. In the leaf-only files the class is 0. Training does not read those files directly.

Figure `02_label_papaver.jpg` is one unmasked Papaver training photo after this process. One green box contains the lobes and the stem. It does not cut the specimen into leaflets.

### 5.3 COCO128 and the class-80 copy

COCO128 is a 128-image slice of COCO, about 7 MB, with the 80 everyday classes. It is mixed in so the pretrained head is not trained only on leaves.

| Item | Count |
| --- | --- |
| COCO128 photos listed with the leaf training photos | 102 |
| COCO128 photos held out | 26 |
| Training rows the trainer sees | 460 (358 leaf + 102 COCO128) |
| Leaf validation photos (the project score) | 85 |
| Combined validation list (leaf val + held-out COCO) | 111 |

Three details matter:

- Leaf photos are **copied** into the combined folder. A symlink would point back at the leaf-only labels, which still say class 0. In the 81-class set, class 0 is `person`, so the leaves would be taught as people.
- The copied leaf boxes are rewritten from class 0 to class **80**. Classes 0–79 keep the COCO names. The name list ends with `leaf`.
- `best.pt` is chosen on `leaf_val.txt` (the 85 leaf photos only). The mixed list of 111 photos is not the score that saves the checkpoint. An earlier YOLO26s run that watched the mixed score saved epoch 1, while the leaf box was still loose (leaf mAP50-95 about **0.528**). That is why this run watches the leaf score instead. The 0.528 figure is that earlier recipe. It is not a third model in the table.

Figure `01_training_labels.jpg` is the combined training set the model actually saw. The tall bar at class 80 is the 358 training leaves. The bar at class 0 is `person` from the COCO128 photos. The overlaid rectangles sit in the middle of the frame and are large, which matches one pressed specimen. The scatter of centers is around the middle of the photo. This figure is a picture of the labels, not of the model’s score.

## 6. Training

Both runs started from the COCO file (`yolo26s.pt` or `yolo26m.pt`), used AdamW, image size 640, and batch 16. Both finished at batch 16. The out-of-memory retry at batch 8 was not needed. Both ran the full **100** epochs. Early stopping was armed (patience 40 for s, 45 for m) and did not fire. The saved file is still the best leaf epoch, not epoch 100.

Shared settings: weight decay 0.0005, final learning-rate fraction 0.01, warmup 2 epochs, rotate up to 10°, vertical flip 0.2, horizontal flip 0.5, mosaic turned off for the last 20 epochs. These photos are one leaf on paper, so stronger scene augmentation would invent pictures the set does not contain.

| | YOLO26s | YOLO26m |
| --- | --- | --- |
| Starting file | `yolo26s.pt` | `yolo26m.pt` |
| Epoch cap | 100 | 100 |
| Epochs run | 100 | 100 |
| Patience | 40 | 45 |
| Best epoch (leaf val) | 76 | 64 |
| Frozen layers | first 10 | first 8 |
| AdamW learning rate | 0.001 | 0.0006 |
| Mosaic | 0.6 | 0.5 |
| Saved file | 20.43 MB | 44.26 MB |
| Leaf mAP50-95 of that file | 0.9461 | 0.9245 |

The first frozen layers keep the COCO edge and texture features. The neck and the head learn `leaf`. The medium model freezes fewer layers and takes a smaller step, on purpose.

At the end of epoch 100 the training log’s leaf mAP50-95 was **0.9192** (s) and **0.9154** (m). Both are below the saved peaks, which is why the last epoch is not the file we keep.

Figures `03_yolo26s_curves.png` and `04_yolo26m_curves.png` are those 100 epochs. Box, classification, and distribution-focal losses fall. Precision, recall, and mAP50 rise within the first part of training and then sit high. mAP50-95 climbs more slowly and keeps moving past epoch 50. That slow climb is the box getting tighter. The saved epoch is the peak of the mAP50-95 panel (epoch 76 on the small model, epoch 64 on the medium model).

## 7. How to read the scores

- **Precision:** of the boxes the model kept, how many were correct.
- **Recall:** of the true leaves, how many were found.
- **IoU:** overlap of the predicted box and the true box. 1 is a perfect overlap.
- **mAP50:** mean average precision when a box counts as correct at 50% overlap.
- **mAP50-95:** the same idea averaged from 50% overlap up to 95% overlap. This is the stricter number. This project ranks models by it.

The project score is the `leaf` row on the **85** held-out leaf photos, at image size 640, on GPU 0. It is not the curve in the training figure, and it is not the COCO128 row.

The precision-recall figure prints **mAP50** in the legend (0.984 for YOLO26s). That is the easier overlap. Do not caption that legend as 0.9461.

## 8. Results

Scored from the saved `best.pt` files on 17 November 2026. Image size 640, GPU 0, batch 16.

### 8.1 Leaf validation, 85 photos (the project score)

| Model | Precision | Recall | mAP50 | mAP50-95 | Weights | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26s | 0.9878 | 0.9882 | 0.9841 | 0.9461 | 20.43 MB | 76 |
| YOLO26m | 0.9757 | 0.9882 | 0.9814 | 0.9245 | 44.26 MB | 64 |

YOLO26s is the winner on the strict score, by 0.0216 mAP50-95, and its file is about half the size (20.43 MB against 44.26 MB). Recall is the same to four decimals (0.9882): both models find the leaf. The gap is how tight the box is, which is what mAP50-95 measures. The medium model’s extra weights did not buy a tighter leaf box on this set.

The training log at epoch 64 for YOLO26m shows leaf mAP50-95 0.9275. The separate test of the saved file, which is the number in the table, is 0.9245. Use 0.9245 in the report. For YOLO26s the test number 0.9461 matches the epoch-76 log (0.9460).

Figure `05_yolo26s_leaf_pr.png`: precision-recall curve of YOLO26s on the 85 leaf photos. The legend reads leaf 0.984, which is mAP50. The curve stays near precision 1 until recall is almost 1. The ranking number for this same run is mAP50-95 0.9461, which is stricter and is not printed on the figure.

Figure `06_yolo26s_leaf_predictions.jpg`: sixteen of those validation photos with the YOLO26s boxes. The labels read `leaf` at about 0.9. The box follows the specimen, including lobed maple leaves and small leaves, and leaves the colored paper outside. The file names contain the species. The model only outputs `leaf`. This grid is a qualitative check, not the 0.9461 score.

Figure `07_yolo26m_leaf_predictions.jpg`: the same kind of grid for YOLO26m on validation photos. Boxes also read `leaf` at about 0.9. Use it to show that the medium model finds the specimen too. The table, not this picture, is what says the small model’s boxes are tighter.

Figure `08_yolo26s_leaf_confusion.png`: normalized confusion matrix for the YOLO26s leaf test. The axes list every class the model carries (COCO classes 0–79 plus leaf) because the head still has those names. The test photos contain only leaves. The cell that matters is leaf predicted as leaf. Empty COCO rows are expected. A mark in the background column is a box drawn where no leaf was labeled. This is not a species classifier.

### 8.2 Held-out COCO128, 26 photos (the forgetting check)

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO26s | 0.5094 | 0.3160 | 0.3760 | 0.2072 |
| YOLO26m | 0.7006 | 0.2798 | 0.3499 | 0.1963 |

COCO scores are much lower than the leaf scores for both models. Fine-tuning on 443 studio leaves plus 102 COCO photos is enough to learn `leaf` and not enough to keep all 80 COCO classes sharp. YOLO26s kept a slightly higher COCO128 mAP50-95 (0.2072 against 0.1963) and a higher COCO recall (0.3160 against 0.2798). This row is a forgetting check. It is not the project score. Twenty-six photos cannot prove the 80 classes are healthy. It does show that YOLO26s did not win the leaf score by forgetting COCO more than YOLO26m did.

### 8.3 Which model performed best, and why

**YOLO26s performed best.** It is the file the camera program should load: `yolo/runs/yolo26s/weights/best.pt`, the epoch-76 checkpoint, 20.43 MB.

The ranking rule is one number: leaf mAP50-95 on the 85 held-out leaf photos. YOLO26s scored **0.9461**. YOLO26m scored **0.9245**. The gap is **0.0216**.

Why that gap is the right reason to call a winner:

- Both models find the leaf. Recall is **0.9882** for both. A model that missed leaves would lose here. Neither did.
- YOLO26s is slightly more careful about the boxes it keeps: precision **0.9878** against **0.9757**. mAP50 is close: **0.9841** against **0.9814**. At the easy 50% overlap, they are nearly tied.
- mAP50-95 averages correctness from 50% overlap up to 95% overlap. That is the box getting tighter, not the model merely noticing a leaf. YOLO26s wins on that stricter test. The prediction grids both say `leaf` at about 0.9, so the pictures alone do not show the winner. The table does.
- The medium model was not handicapped. It trained for the same 100 epochs, the same image size 640, and the same batch 16. Its learning rate was lower (0.0006 against 0.001), its mosaic was milder (0.5 against 0.6), and it froze fewer layers (8 against 10), so a loss cannot be blamed on a step that was too large. The extra weights still did not tighten the box. On 358 studio leaves, spare capacity is more likely to memorize the paper color and the lighting than to learn a better outline.
- The file is about half the size: **20.43 MB** against **44.26 MB**. The watch runs on the CPU at image size 416. A smaller file that is also more accurate is the one to ship. Size would not have saved a weaker detector. Here the smaller file is the stronger detector.
- On the 26 held-out COCO128 photos, YOLO26s is also slightly ahead (mAP50-95 **0.2072** against **0.1963**, recall **0.3160** against **0.2798**). That check does not pick the winner. It only says the leaf win did not come from throwing the COCO classes away.

What “best” does not mean:

- It does not mean the stronger published COCO model. Before fine-tuning, YOLO26m has the higher COCO mAP (53.1 against 48.6). After fine-tuning on these leaves, the small model is better. The leaf table overrules the COCO brochure.
- It does not mean YOLO26m fails. It finds the same share of leaves. It is the second-best file, and it is the one we would not load.
- It does not mean the last epoch. Epoch 100 is worse for both (training-log leaf mAP50-95 0.9192 for s, 0.9154 for m). The winner is epoch 76 of YOLO26s.

## 9. Application

The program is a solar-panel watch. It loads the YOLO26s `best.pt` (COCO classes 0–79 plus leaf as class 80). It can read a camera index, a photo, or a video. It looks up each detection by the class name from `model.names`, not by a hard-coded id, so `leaf` stays class 80.

- On start it says “Solar panel watch started.”
- A class must stay in view for 3 frames before it counts.
- It speaks once, then stays quiet for that class until the class leaves and comes back, with at least 8 seconds between repeats. The same sentence is printed.
- Defaults: confidence 0.25, image size 416, device CPU. Press q to quit the preview.
- Speech uses spd-say. If speech is missing, the line is only printed.

Other COCO classes (chair, laptop, and the rest) are drawn and not spoken.

| What it sees | What it says |
| --- | --- |
| leaf | There is a leaf on the solar panel. Clean it. |
| bird, potted plant, umbrella, kite, bottle, cup, bowl, backpack, handbag, suitcase, book, sports ball, frisbee, banana, apple, orange | There is a {name} on the solar panel. Clean it. |
| person | There is a person in this area. |
| cat, dog, horse, sheep, cow, bear | There is a {name} in this area. It can damage the solar panel. |
| baseball bat, knife, scissors, skateboard | There is a {name} near the solar panel. It can damage the glass. |
| bicycle, car, motorcycle, bus, truck | There is a {name} in this area. Keep it clear of the solar panel. |

YOLO26s is the file to load because it won leaf mAP50-95 (0.9461 against 0.9245) and because 20.43 MB is the better size for a CPU preview at image size 416. The scored tests used image size 640 on the GPU. The live preview uses 416 on the CPU so the watch can run when the GPU is busy. That speed choice is not a second accuracy table. No camera-on-a-panel score was measured.

## 10. Limitations

- The 85 validation leaves are the same paper and the same camera style as the training photos. A panel in sun, rain, or at a steep angle is a harder test that this score does not cover.
- The detector was not trained on solar-panel photos.
- COCO classes, including person, are weaker after the leaf fine-tune. The 26-photo COCO128 check is only a warning, not a full COCO evaluation.
- A SAM run that boxed leaflets looked better in a single number (0.937) and was wrong on compound leaves. The labels used here refuse that kind of box. The score and the pictures have to be read together.
- An earlier YOLO26s recipe that picked `best.pt` from mixed validation saved epoch 1 and only reached leaf mAP50-95 about 0.528. Watching the leaf validation score is what fixed that. The 0.528 run is not the file in section 8.
- The app speaks from class names. It does not measure how much power the panel lost.
- The model does not classify species.
- No mAP was saved for a 40-photo hand-labeled pilot. That pilot is not a result in this report.

## 11. Conclusion

Labeling all 443 UCI leaf photos (340 mask boxes kept, 103 whole-specimen boxes from the paper color, SAM2.1-L allowed only when it still covered the specimen) and fine-tuning two COCO checkpoints produced a clear comparison. `best.pt` was chosen on the 85 leaf photos, over 100 epochs, so the saved file is the tight leaf box rather than an early mixed-validation peak.

YOLO26s performed best: leaf mAP50-95 **0.9461** against **0.9245**, in **20.43 MB** against **44.26 MB**. Recall is the same, so the win is a tighter box, not a model that finds more leaves. The medium model had a milder recipe and still lost, which is what we expect when extra weights have only 358 studio leaves to learn from. The camera program should load the YOLO26s checkpoint from epoch 76. The next real test is photos of leaves on actual panels, which this dataset does not contain.

## 12. References

1. UCI Machine Learning Repository. Leaf dataset (dataset 288). https://archive.ics.uci.edu/dataset/288/leaf
2. Ultralytics documentation. YOLO26. https://docs.ultralytics.com/models/yolo26
3. COCO: Common Objects in Context. Source of the 80 pretrained classes and of the COCO128 subset. https://cocodataset.org

## 13. Figures to place

| File | Put it in | Caption to use |
| --- | --- | --- |
| `figures/01_training_labels.jpg` | Dataset, after the class-80 explanation | Training-label report for the combined set (358 leaves + COCO128). The tall bar at class 80 is leaf. The bar at class 0 is person. The rectangles are the boxes drawn on top of each other. This is a picture of the labels, not of the score. |
| `figures/02_label_papaver.jpg` | How the boxes were made | One unmasked Papaver training photo. The green box contains the whole specimen, lobes and stem. Papaver is one of the 103 photos with no mask. |
| `figures/03_yolo26s_curves.png` | Training | YOLO26s training, 100 epochs. Losses fall. mAP50-95 is the bottom-right panel and peaks at epoch 76 (0.9461 on the saved test). |
| `figures/04_yolo26m_curves.png` | Training | YOLO26m training, 100 epochs. Same reading. The leaf mAP50-95 peak is epoch 64. The saved test of that file is 0.9245. |
| `figures/05_yolo26s_leaf_pr.png` | Leaf results | Precision-recall curve for YOLO26s on the 85 held-out leaves. The legend 0.984 is mAP50. The strict score is mAP50-95 0.9461. |
| `figures/06_yolo26s_leaf_predictions.jpg` | Leaf results | Sixteen validation photos with YOLO26s boxes, labeled leaf at about 0.9. Qualitative check, not the 0.9461 score. |
| `figures/07_yolo26m_leaf_predictions.jpg` | Leaf results | Validation photos with YOLO26m boxes, also leaf at about 0.9. The table is what ranks the two models. |
| `figures/08_yolo26s_leaf_confusion.png` | Leaf results | Normalized confusion matrix for the YOLO26s leaf test. Axes list all 81 class names. The test photos contain only leaves, so the cell that matters is leaf predicted as leaf. |

## 14. PowerPoint script

Build a real 15-slide PowerPoint beside the report. Widescreen 16:9. White background, one dark green accent for headings and for the winning number. No clip art, no stock photos, no gradients, no word art. Footer on every slide except the title: `Md. Abu Said Shabib  ·  Leaf detection  ·  n / 15`.

On-slide text is short. If a line below is marked **On slide**, it may appear as written. If it is marked **Say this**, it goes only in the speaker notes. Do not paste speaker notes onto the slide. Each content slide has at most six lines. The figures are the files in `figures/`. Do not redraw a chart when the file exists.

Do not put YOLOv8 or YOLO11 on any slide. Do not show 0.528 as if it were the result. That number appears once, on the training slide, as the mistake this recipe avoided.

### Slide 1 — Title

**On slide**

- University of Jinan
- Computer Science & Technology
- Leaf detection for a solar panel watch
- Fine-tuning YOLO26s and YOLO26m on the UCI Leaf dataset
- Md. Abu Said Shabib
- Student ID: 202624100003
- Teacher: Jeffery Tan
- 17 November 2026

**Say this:** One leaf on a panel shades the string. I trained two detectors. The small one won, and it is the file I would load.

### Slide 2 — The problem

**On slide**

- A leaf shades the cells and holds dirt and water
- The program watches a camera, draws a box, and speaks a warning
- It learns one class: leaf
- It does not name the species
- It was not trained on solar-panel photos
- It does not measure lost power

**Say this:** Detection returns a rectangle, a class name, and a confidence. YOLO does that in one pass, which is fast enough for a camera.

### Slide 3 — Why these two models

**On slide**

- Start from a COCO checkpoint, not from random weights
- 358 training leaves cannot teach edges from nothing
- Leaf is class 80, so person stays person
- YOLO26s: published COCO mAP 48.6, 9.5M parameters, about 20 MB
- YOLO26m: published COCO mAP 53.1, 20.4M parameters, about 44 MB
- Question: does the extra size tighten the leaf box?

**Say this:** Those COCO numbers are Ultralytics’ brochure, not our leaf score. I trained both so the answer would be measured. Nano, large, and extra-large were not trained. YOLOv8 and YOLO11 are not in this comparison.

### Slide 4 — Machine

**On slide**

- Intel Core i5-12500, 32 GB RAM
- NVIDIA RTX 3060, 12 GB
- Training and the score: GPU, image size 640, batch 16
- The watch: CPU, image size 416
- Python 3.12, PyTorch 2.15, Ultralytics 8.4
- Both models finished at batch 16

**Say this:** SAM2.1-L was used only to check labels. It is not the detector.

### Slide 5 — How many photos

**On slide**

- UCI Leaf, 40 species, 443 photographs
- 340 boxes from a black-and-white mask
- 103 boxes from the paper color
- Train 358, validation 85, every species on both sides
- Plus COCO128: 102 in training, 26 held out
- The trainer saw 460 rows

**Say this:** The 85 leaf photos are the project score. The 26 COCO photos are only a forgetting check.

### Slide 6 — How a box was made

**On slide:** figure `figures/02_label_papaver.jpg`, and these lines beside it

- Mask photos: the box is the white pixels, and those 340 boxes were kept
- No mask: the box surrounds whatever is not the paper color
- One specimen, one box, including leaflets and the stem
- Checker: PASS, 0 problems
- SAM2.1-L could replace a box only when it still covered the whole specimen

**Say this:** A leaflet-sized SAM box was rejected. An earlier run that kept those leaflet boxes scored 0.937 and looked wrong. That run is not this result. The green box on the Papaver photo is the labeling goal.

### Slide 7 — What training actually saw

**On slide:** figure `figures/01_training_labels.jpg` full width, and one line under it

- Class 80 is leaf. The photos were copied, not symlinked. Class 0 would have taught the leaf as a person.

**Say this:** The tall bar at class 80 is the 358 training leaves. The bar at class 0 is person from COCO128. The rectangles are the label boxes, not the model’s predictions.

### Slide 8 — The two recipes

**On slide:** `figures/03_yolo26s_curves.png` on the left, `figures/04_yolo26m_curves.png` on the right

| | YOLO26s | YOLO26m |
| --- | --- | --- |
| Epochs run | 100 | 100 |
| Best epoch | 76 | 64 |
| Freeze | 10 | 8 |
| Learning rate | 0.001 | 0.0006 |
| Mosaic | 0.6 | 0.5 |

**Say this:** `best.pt` is the best leaf epoch, not epoch 100. Epoch 100 is worse. An older recipe saved epoch 1 from the mixed score and only reached about 0.528 on leaves. This run watches the leaf score so that does not happen again. The bottom-right panel of each curve is the number that climbs slowly. That climb is the box getting tighter.

### Slide 9 — How the winner is chosen

**On slide**

- Rank by leaf mAP50-95 on the 85 held-out photos
- mAP50 means the box overlaps the truth by at least half
- mAP50-95 also requires a tight box
- Precision: of the boxes we kept, how many were right
- Recall: of the true leaves, how many we found
- The legend 0.984 on the curve is mAP50, not the ranking score

**Say this:** Same recall means both models find the leaf. The stricter score is what separates a loose box from a tight one.

### Slide 10 — The scores

**On slide**

| Model | Precision | Recall | mAP50 | mAP50-95 | File |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO26s | 0.9878 | 0.9882 | 0.9841 | 0.9461 | 20.43 MB |
| YOLO26m | 0.9757 | 0.9882 | 0.9814 | 0.9245 | 44.26 MB |

**Say this:** Read the mAP50-95 column. Do not stop at recall. Recall is identical. Image size 640, GPU, batch 16, scored on 17 November 2026.

### Slide 11 — Proof, before the verdict

**On slide:** `figures/05_yolo26s_leaf_pr.png` on the left, `figures/06_yolo26s_leaf_predictions.jpg` on the right. One caption line:

- YOLO26s on the 85 leaf photos. The curve’s 0.984 is mAP50. Boxes read leaf at about 0.9.

**Say this:** The grid is a check that the box sits on the specimen and leaves the paper outside. Species names are in the file names. The model only says leaf. The confusion matrix lists all 81 class names because the head still has them. The cell that matters is leaf predicted as leaf.

### Slide 12 — YOLO26s performed best

This is the slide the talk turns on. Make **0.9461** the largest text on the slide. Under it, in smaller type, **YOLO26m 0.9245**.

**On slide**

- Winner: YOLO26s, epoch 76, 20.43 MB
- Same recall, 0.9882. The win is a tighter box, +0.0216 mAP50-95
- Medium had the gentler recipe and still lost
- Extra weights did not help on 358 studio leaves
- Half the file size, and the better score
- COCO check also slightly favors the small model (0.2072 against 0.1963)

**Say this:** Before training, the medium model was the stronger COCO detector, 53.1 against 48.6. After this fine-tune, the leaf table overrules that. YOLO26m is not a failure. It finds the same share of leaves. It is not the file I would load. The pictures of both models look similar at confidence 0.9. The table is the decision, not the grid.

### Slide 13 — What the watch says

**On slide**

- Load YOLO26s, because it won and because it is smaller
- Confidence 0.25, image size 416, CPU
- Leaf: “There is a leaf on the solar panel. Clean it.”
- Person: “There is a person in this area.”
- Animal: “There is a {name} in this area. It can damage the solar panel.”
- Vehicle: “There is a {name} in this area. Keep it clear of the solar panel.”

**Say this:** A class must stay for 3 frames. It speaks once, then waits at least 8 seconds. Chair and laptop are drawn and not spoken. No panel-camera score was measured. The 0.9461 number is the studio validation set.

### Slide 14 — Limits

**On slide**

- Validation leaves use the same paper and the same camera as training
- No solar-panel photos in the score
- COCO classes are weaker after the leaf fine-tune
- The model does not name the species
- The watch does not measure lost power
- The next test is a real panel

**Say this:** I will not claim the 0.9461 number is a panel result. It is 85 held-out studio leaves.

### Slide 15 — Close

**On slide**

- 443 leaves labeled, one box each, checker PASS
- YOLO26s leaf mAP50-95 0.9461 in 20.43 MB
- YOLO26m leaf mAP50-95 0.9245 in 44.26 MB
- Best model: YOLO26s, because the box is tighter and the file is smaller
- UCI Leaf dataset 288. Ultralytics YOLO26. COCO.

**Say this:** I would ship the small checkpoint from epoch 76. Thank you.

## Report section order

1. Abstract
2. Introduction
3. Why YOLO26s and YOLO26m
4. Hardware and software
5. Dataset
6. How the labels were made
7. Training
8. How to read the scores
9. Results
10. Which model performed best, and why
11. Application
12. Limitations
13. Conclusion
14. References
