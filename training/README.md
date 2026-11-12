# Training

`train.py` fine-tunes `yolo26s.pt` on `dataset/combined/coco128_leaf/data.yaml`.

```bash
python training/train.py
```

Logs and weights are written to `training/runs/coco128-leaf/`. The checkpoint to keep is `weights/best.pt`.
