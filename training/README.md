# Training

Fine-tune a COCO-pretrained YOLO26 checkpoint on `dataset/combined/coco128_leaf/data.yaml`.

## Models

Each family gets its own run folder under `training/runs/<name>/weights/best.pt`.

| Run folder | Base weights | Best for |
| --- | --- | --- |
| `yolov8n/` | `yolov8n.pt` | Classic YOLOv8 nano (CPU-friendly) |
| `yolo11n/` | `yolo11n.pt` | YOLO11 nano (CPU-friendly) |
| `yolo26n/` | `yolo26n.pt` | YOLO26 nano (CPU-friendly default) |
| `yolo26s/` | `yolo26s.pt` | YOLO26 small (stronger GPU model) |
| `coco128-leaf/` | legacy YOLO26s path | older clones |

Report-ready numbers for every model are under [`../results/`](../results/).

## Commands

```bash
source .venv/bin/activate

python training/train.py --model yolov8n.pt
python training/train.py --model yolo11n.pt
python training/train.py --model yolo26n.pt
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
