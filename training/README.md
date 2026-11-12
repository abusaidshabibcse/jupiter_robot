# Training

Fine-tune a COCO-pretrained YOLO26 checkpoint on `dataset/combined/coco128_leaf/data.yaml`.

## Models

| Run folder | Checkpoint | Best for | Size |
| --- | --- | --- | --- |
| `runs/yolo26n/` | `yolo26n.pt` → `weights/best.pt` | **2-core / CPU laptop** | ~5 MB class |
| `runs/yolo26s/` | `yolo26s.pt` → `weights/best.pt` | GPU / better boxes | ~20 MB |
| `runs/coco128-leaf/` | legacy alias of the s-model | older docs/clones | ~20 MB |

`yolo26n` is the nano YOLO26 model. It is the lightest current Ultralytics YOLO26 detector and the one meant for a weak CPU. `yolo26s` is kept for higher accuracy when a GPU is available.

## Commands

```bash
source .venv/bin/activate

# CPU-friendly nano (default)
python training/train.py --model yolo26n.pt

# Stronger small model
python training/train.py --model yolo26s.pt
```

On a 2-core machine that must train locally:

```bash
python training/train.py --model yolo26n.pt --device cpu --batch 4 --workers 0
```

## Technique

- Supervised fine-tune of a COCO-pretrained checkpoint
- Leaf is class `80` so COCO classes `0`–`79` keep their pretrained head rows
- First 10 layers frozen; AdamW at learning rate `0.001`
- Mild geometric augmentation (rotate, flips, mosaic)

Full rationale and measured scores: [../docs/PROJECT.md](../docs/PROJECT.md).

## Output

Each run writes `training/runs/<model-stem>/weights/best.pt`. Keep that file for testing and laptop use.
