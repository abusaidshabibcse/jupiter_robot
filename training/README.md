# Training

`train.py` fine-tunes pretrained `yolo26s.pt` on `dataset/combined/coco128_leaf/data.yaml`.

```bash
source .venv/bin/activate
python training/train.py
```

## Technique

- Supervised fine-tune of a COCO-pretrained YOLO26s checkpoint
- Leaf is class `80` so COCO classes `0`–`79` keep their pretrained head rows
- First 10 layers frozen; AdamW at learning rate `0.001`
- Mild geometric augmentation (rotate, flips, mosaic)

Full rationale and measured scores are in [../docs/PROJECT.md](../docs/PROJECT.md).

## Output

Logs and weights go to `training/runs/coco128-leaf/`. Keep `weights/best.pt`.

On this machine the run stopped at epoch 21 by early stopping. The saved `best.pt` is the mixed-validation peak (epoch 1). Leaf-only testing of that file scored **mAP50 0.930** and **mAP50-95 0.528** on the 85 leaf validation photos.
