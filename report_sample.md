University of Jinan
Computer Science & Technology
Leaf detection for a solar panel watch
Fine-tuning YOLOv8, YOLO11, and YOLO26 on the UCI Leaf dataset
Name: Md. Abu Said Shabib
Student ID: 202624100003
Teacher: Jeffery Tan
Date: November 16, 2026
Contents
Abstract	2
Introduction	2
Hardware and software	2
Dataset	3
Method	4
Stage 1	4
Stage 2	5
Stage 3	5
How to read the scores	6
Results	7
Application	10
Limitations	11
Conclusion	11
References	12
Abstract
A leaf lying on a solar panel shades the cells and holds dirt and water, so I wanted a small program that watches a camera and says when one is there. This report describes the detector behind it. I worked in three stages. A first run on 40 hand-labeled photos was not good enough to use, and no score was saved for it. I then labeled the full UCI Leaf set of 443 photographs, using mask boxes for 340 photos and a paper-color border method for the other 103. In the third stage I fine-tuned six COCO-pretrained YOLO checkpoints (YOLOv8, YOLO11 and YOLO26, small and medium) on those leaves plus COCO128, with one shared recipe. On 85 held-out leaf photos, YOLO11s reached the highest leaf mAP50-95, 0.8717, in a 19.23 MB file. The spoken camera app still loads YOLO26s, whose leaf mAP50-95 is 0.5283. The detector was not trained on solar-panel photos and does not name species.
Introduction
A leaf on a solar panel shades the cells and holds dirt and water. Shade on part of a module cuts the power of that string. The practical goal of this project is a small desktop program that watches a camera, draws a box around what it sees, and speaks a short warning so that someone can clean the panel.
The program rests on object detection: a model looks at a picture and returns a list of rectangles, each with a class name and a confidence between 0 and 1. I used YOLO (“You Only Look Once”), a family of detectors that predict all boxes in a single pass over the image, which makes them fast enough for a live camera. Each model I used starts from weights pretrained on COCO (Common Objects in Context), a large photo collection whose 80 classes include person, car and dog.
This project does three things. It labels the UCI Leaf photographs with one box each, it fine-tunes six pretrained checkpoints on those leaves and compares them on the same recipe, and it describes a camera program that turns detections into spoken alerts.
It does not do several things that a reader might assume. The detector was not trained on photos of solar panels. It does not name the plant species; the only class it learns is leaf. It does not measure how much power a panel lost.
Hardware and software
Training and every scored test ran on one desktop computer. The hardware sentence is: Intel Core i5-12500, 32 GB RAM, NVIDIA GeForce RTX 3060 12 GB.
Table 1. Training computer
Part
What it is
Processor
Intel Core i5-12500
Memory
32 GB RAM
Graphics
NVIDIA GeForce RTX 3060, 12 GB
Role of the GPU
Training and the scored tests (image size 640, batch 16)
Role of the CPU
The solar-panel camera app defaults to CPU, image size 416, so it can run when the GPU is busy
Table 2. Software
Piece
Version or name
Language
Python 3.12
Deep learning
PyTorch 2.15 with CUDA
Detector library
Ultralytics 8.4
Models trained
YOLOv8s, YOLOv8m, YOLO11s, YOLO11m, YOLO26s, YOLO26m
Speech in the app
spd-say (prints the same line if speech is missing)

A note on file names: YOLOv8 keeps the v (yolov8s.pt, yolov8m.pt), while YOLO11 and YOLO26 do not (yolo11s.pt, yolo26s.pt). The letter s is the small checkpoint and m is the medium checkpoint. No nano, large or extra-large sizes were trained.
Dataset
Leaf dataset
The source is the UCI Machine Learning Repository Leaf dataset (https://archive.ics.uci.edu/dataset/288/leaf). The archive covers 40 plant species. The full image folder used here holds 443 leaf photographs. Each is one pressed specimen on colored paper. The task is object detection: one rectangle, class name leaf. File names still contain the species, but the label line does not, so the model is never asked to name it.
    • 340 photos already had a black-and-white mask. The box is the extent of the white pixels. No detection model was used.
    • 103 photos had no mask (Acer negundo, Taxus, Papaver, Polypodium, Pinus, Fraxinus, Aesculus, Chelidonium, Schinus, Fragaria). The paper color is measured from the image border, and the box surrounds every pixel that is not that paper, so a compound leaf keeps its leaflets. No detection model was used.
    • The split keeps every species on both sides: 358 training photos and 85 validation photos. The 85 validation photos give the leaf score.
    • A checker rejected a photo with no box, a box outside the frame, a speck, or a box that covered the whole frame.
MobileSAM (mobile_sam.pt) can be run on the 103 unmasked photos, but it was not the labeler for this scored set. On this colored-paper set it is slower and not tighter than the border method.
COCO128
The second dataset is mixed in at training time. COCO128 is a 128-image slice of COCO (everyday photos with person, car, dog and 77 other classes), about 7 MB. Of those photos, 102 are listed with the 358 leaf training photos, so the trainer sees 460 training rows. The other 26 are held out. Their score is a forgetting check, not the leaf score. COCO128 is large enough to run the combined pipeline on one machine, but not large enough to keep all 80 COCO classes sharp after fine-tuning.
Class numbers
In the leaf-only files, leaf is class 0. In the combined files, the leaf photos are copied (not symlinked) and leaf is rewritten to class 80. Classes 0 to 79 keep the COCO names. A symlink would have sent Ultralytics back to the leaf-only labels, which still say class 0, and class 0 in the 81-class set is person.
Table 3. Dataset counts
Item
Count
Species in the UCI archive
40
Leaf photographs in the full folder
443
With a mask (box from white pixels)
340
Without a mask (box from paper color)
103
Leaf training photos / validation photos
358 / 85
COCO128 photos in training / held out
102 / 26
Training rows seen by the trainer
460

Method
The work ran in three stages. The first stage failed on purpose-built evidence I can show but not score. The second built the full label set. The third trained and compared six checkpoints.
Stage 1
The first experiment used 40 leaf photos from different plant categories in the UCI set, labeled by hand, with one class (leaf) and about one box per photo. A YOLO model was trained on that small set. The result was not good enough to use: forty studio leaves do not cover pose, species shape, or a new background.
No precision, recall or mAP number was saved for this 40-image run, so none is reported here. The two figures below are charts of the labels, not charts of a score.

Figure 1. Ultralytics label report from the first experiment. One class only: leaf. The red block on the left is the class count. The axis runs from 0 to 40, so this set has about 40 leaf boxes (40 photos, one leaf each, taken from different plant categories in the UCI Leaf archive). The nested rectangles on the right are those boxes drawn on top of each other in normalized image coordinates. The two scatter plots show where the box centers sit (x vs y) and how wide and tall the boxes are (width vs height). Centers and sizes are spread out. With only 40 examples, that spread is too thin to train a reliable detector. This picture is why the first training run was not good enough. It is a picture of the labels, not of the model’s score. No precision, recall, or mAP was saved for that run.

Figure 2. The matching correlogram for the same 40 boxes. Each small heatmap compares two box values (x, y, width, height). The bars along the right and top are histograms of those values. Read it as “the 40 boxes do not share one size or one place in the frame.”
Stage 2
Stage 2 labeled the full set of 443 photos.
    1. Photos with a black-and-white mask (340) kept the mask box.
    2. Photos without a mask (103) used the paper-color border method described in the Dataset section.
    3. The checker then confirmed every box.
SAM was not used to label this set. The boxes come from the masks or from the paper color, and no detection model made any of them.
Stage 3
The leaf photos were copied into a combined dataset with COCO128, with leaf stored as class 80. Training started from COCO-pretrained weights, not from random weights, and every checkpoint used the same settings.
    • Frozen layers: the first 10 layers stay fixed so the early COCO edge and texture features stay put. The neck and the head learn leaf. Ultralytics remaps the pretrained classification head by name, so the person row is not thrown away when leaf is added.
    • Optimizer: AdamW, learning rate 0.001, weight decay 0.0005.
    • Length: epoch cap 80, early-stop patience 20 (training stops after 20 epochs without improvement).
    • Size: image size 640, batch 16. All six finished at batch 16 on the 12 GB RTX 3060. The script would retry a medium model at batch 8 if it ran out of memory; that retry was not needed.
    • Augmentation: rotate up to 10°, vertical flip 0.2, horizontal flip 0.5, mosaic 0.8, mosaic off for the last 10 epochs. These photos are one leaf on paper, so stronger scene augmentation would invent pictures the set does not contain.
Which epoch is `best.pt`? It is the best epoch of the mixed COCO-plus-leaf validation mAP50-95, not the last epoch and not a leaf-only score. This matters for YOLO26s, as the results show.
Table 4. Training runs
Model
Starting file
Epochs run
Best epoch (mixed val)
Saved weights
YOLOv8s
yolov8s.pt
44
24
22.57 MB
YOLOv8m
yolov8m.pt
67
47
52.11 MB
YOLO11s
yolo11s.pt
41
21
19.23 MB
YOLO11m
yolo11m.pt
44
24
40.63 MB
YOLO26s
yolo26s.pt
21
1
20.42 MB
YOLO26m
yolo26m.pt
31
11
44.25 MB

Every run stopped early. YOLO26s stopped at the mixed-validation peak of epoch 1, which is why its leaf boxes stay loose even though it still finds most leaves.

Figure 3. Training curves for the YOLO11s run, which stopped at epoch 41. The three loss plots fall. The precision, recall, and mAP plots are the mixed COCO128-plus-leaf validation used to pick best.pt. That mixed mAP50-95 stays near 0.19. The leaf-only score of 0.8717 is a separate test and is not the right-hand axis of this figure.
How to read the scores
    • Precision: of the boxes the model kept, how many were correct.
    • Recall: of the true leaves, how many were found.
    • IoU (intersection over union): the overlap of the predicted box and the true box. 1 is a perfect overlap.
    • mAP50: mean average precision when a box counts as correct at 50% overlap.
    • mAP50-95: the same idea averaged from 50% overlap up to 95% overlap. It is the stricter number, and the one this project uses to rank the six models.
The project score is the `leaf` row from the 85 held-out leaf photos. It is not the mixed “all” row that the trainer prints during training, and not the curve in Figure 3.
Results
All six checkpoints were evaluated at image size 640, on GPU 0, with batch 16. Table 5 is the project score. Rows are in training order, not rank.
Table 5. Leaf validation, 85 photos (the project score)
Model
Precision
Recall
mAP50
mAP50-95
Weights
YOLOv8s
0.9880
0.9721
0.9834
0.7408
22.57 MB
YOLOv8m
0.9960
0.9882
0.9933
0.8145
52.11 MB
YOLO11s
0.9768
0.9905
0.9944
0.8717
19.23 MB
YOLO11m
0.9667
0.9176
0.9728
0.8170
40.63 MB
YOLO26s
0.8925
0.8794
0.9300
0.5283
20.42 MB
YOLO26m
0.9627
0.9882
0.9925
0.7531
44.25 MB

YOLO11s is the leaf winner on the strict score, with mAP50-95 0.8717 and mAP50 0.9944, and it is also the smallest file (19.23 MB). YOLOv8m (0.8145) and YOLO11m (0.8170) are close to each other and behind YOLO11s. The medium files are about twice as large and did not buy a better leaf box on this set. YOLOv8s finds the leaf (mAP50 0.9834) but draws a looser box (mAP50-95 0.7408). YOLO26m sits in the same band on the strict score (0.7531) with a higher mAP50 (0.9925).
YOLO26s usually finds the leaf (mAP50 0.9300) but its box is only moderately tight (mAP50-95 0.5283). It is the weakest leaf result in the table. The reason is in Table 4: mixed validation peaked at epoch 1, so best.pt is a very early checkpoint.
Table 6. Held-out COCO128, 26 photos (the forgetting check)
Model
Precision
Recall
mAP50
mAP50-95
YOLOv8s
0.5858
0.2643
0.3337
0.1832
YOLOv8m
0.6213
0.3186
0.3193
0.1829
YOLO11s
0.7931
0.1618
0.2651
0.1745
YOLO11m
0.6330
0.2523
0.2777
0.1500
YOLO26s
0.5230
0.4472
0.4843
0.2595
YOLO26m
0.5189
0.3142
0.3585
0.1987

COCO scores fell for every model. Fine-tuning on 443 studio leaves plus 102 COCO photos is enough to learn leaf, and not enough to keep all 80 COCO classes sharp. YOLO26s kept the most: the highest COCO128 mAP50-95 (0.2595) and recall 0.4472. YOLO11s, the best leaf model, has the lowest COCO128 recall of the six (0.1618).

Figure 4. Precision-recall curve for YOLO11s on the 85 held-out leaf photos. This is the leaf test for the best model in the comparison (leaf mAP50-95 0.8717, mAP50 0.9944). It is not a training-loss plot.

Figure 5. Normalized confusion matrix for that same YOLO11s leaf test. The task is one class, leaf, plus background. It is not a species classifier. The axes list every class name the model carries (COCO classes 0 to 79 plus leaf), but the test photos contain only leaves: the dark cell is leaf predicted as leaf, and the faint cells in the background column are boxes drawn where no leaf was labeled.

Figure 6. Sixteen validation specimens with the boxes YOLO11s predicted. Most labels read leaf at about 0.8 to 0.9. One small leaf at the top left is a miss, with a second box near 0.3. The file names on the photos include species; the model only outputs leaf. This grid is a qualitative check, not the 0.8717 score.

Figure 7. Precision-recall curve for YOLO26s on the same 85 leaf photos. This is the checkpoint the camera app loads. Leaf mAP50 is 0.9300 and leaf mAP50-95 is 0.5283, so the model finds the leaf more often than it draws a tight box. Compare it with the YOLO11s curve in Figure 4.

Figure 8. A qualitative check on a photo taken outside the UCI paper-background set, so it is not one of the 85 validation scores and it is not tied to one row of the comparison table. The photo shows a large dark-green leaf in front of a towel, with other leaves behind it. The model drew blue boxes labeled leaf: 0.38 on the large leaf, and 0.39 and 0.26 on other foliage.
Application
The program loads the YOLO26s best.pt (COCO classes 0 to 79 plus leaf as class 80). It can read a camera index, a photo, or a video, and it looks up each detection by the class name from model.names, not by a hard-coded id, so leaf stays class 80.
    • On start it says “Solar panel watch started.”
    • A class must stay in view for 3 frames before it counts.
    • It speaks once, then stays quiet for that class until the class leaves and comes back, with at least 8 seconds between repeats. The same sentence is printed.
    • Defaults: confidence 0.25, image size 416, device CPU. Press q to quit the preview.
    • Speech uses spd-say. If speech is missing, the line is only printed.
Other COCO classes (chair, laptop and the rest) are drawn and not spoken.
Table 7. What the app says
What it sees
What it says
leaf
There is a leaf on the solar panel. Clean it.
bird, potted plant, umbrella, kite, bottle, cup, bowl, backpack, handbag, suitcase, book, sports ball, frisbee, banana, apple, orange
There is a {name} on the solar panel. Clean it.
person
There is a person in this area.
cat, dog, horse, sheep, cow, bear
There is a {name} in this area. It can damage the solar panel.
baseball bat, knife, scissors, skateboard
There is a {name} near the solar panel. It can damage the glass.
bicycle, car, motorcycle, bus, truck
There is a {name} in this area. Keep it clear of the solar panel.

The app loads YOLO26s because that is the checkpoint wired into solar_panel/. In this comparison it is a weaker leaf detector than YOLO11s (leaf mAP50-95 0.5283 against 0.8717), because its best.pt was taken from mixed-validation epoch 1. YOLO11s would be the checkpoint to ship if the choice followed leaf mAP50-95 and file size.
Run from the repository root:
source .venv/bin/activate
python solar_panel/watch.py
python solar_panel/watch.py --source yolo/testing/sample.mp4
Limitations
    • Stage 1 has charts but no saved mAP.
    • Validation leaves are the same paper and the same camera style as training. A panel photo in sun, rain, or at a steep angle is a harder test that this score does not cover.
    • The detector was not trained on solar-panel photos.
    • COCO classes, including person, are weaker after the leaf fine-tune. YOLO11s, the best leaf model, has the lowest COCO128 recall of the six (0.1618).
    • The camera app loads YOLO26s, whose leaf boxes are the loosest in Table 5 (mAP50-95 0.5283), because best.pt was taken from mixed-validation epoch 1.
    • SAM did not label the scored set and did not raise any score.
    • A separate outdoor park clip (303 frames, 1920×1080) was run with an earlier YOLO26s check. It shows trees, grass and a path, not a solar panel and not a single pressed leaf. The model drew no leaf boxes (0 of 303 frames). It marked potted plant on 13 frames and car on 1 frame, both at about 0.28 confidence. The detector knows a specimen leaf, not a tree canopy.
    • The app speaks from class names. It does not measure how much power the panel lost.
    • The model does not classify species.
Conclusion
A 40-image pilot was too small. Labeling the full 443-photo UCI leaf set (mask boxes kept, paper-color boxes on the 103 unmasked photos) and fine-tuning six COCO checkpoints on those leaves plus COCO128 produced a real comparison. On 85 held-out leaves, YOLO11s reached mAP50-95 0.8717 in a 19.23 MB file and is the model this comparison would ship.
The camera program still loads YOLO26s, which finds most leaves (mAP50 0.9300) but with looser boxes (mAP50-95 0.5283), and which kept more of the COCO128 check. The next real test is photos of leaves on actual panels, which this dataset does not contain.
References
    [1] UCI Machine Learning Repository. Leaf dataset (dataset 288). https://archive.ics.uci.edu/dataset/288/leaf
    [2] Ultralytics documentation. YOLOv8. https://docs.ultralytics.com/models/yolov8
    [3] Ultralytics documentation. YOLO11. https://docs.ultralytics.com/models/yolo11
    [4] Ultralytics documentation. YOLO26. https://docs.ultralytics.com/models/yolo26
    [5] COCO: Common Objects in Context. Source of the 80 pretrained classes and of the COCO128 subset. https://cocodataset.org