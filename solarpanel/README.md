# Solar panel watch

Live camera check that uses the fine-tuned **YOLO26s** weights in `yolo/runs/yolo26s/weights/best.pt`. When the camera sees something that can shade, soil, or damage the sheet, it speaks a short instruction. Everything else the model knows is ignored.

| Class | Why it matters | What it says |
| --- | --- | --- |
| person | Someone is at the array | There is a person at the solar panel. Please stay clear of the array. |
| dog | Weight can crack the glass | There is a dog on the solar panel. Please move it away so it does not crack the glass. |
| cat | Claws can scratch the glass | There is a cat on the solar panel. Please remove it before it scratches the glass. |
| bird | Droppings etch the glass and shade cells | There is a bird on the solar panel. Please clear it so it does not leave droppings. |
| leaf | A leaf shades cells and can cause a hot spot | There is a leaf on the solar panel. Please clean it. |

Two or more of the same class uses the plural line, for example leaves or people. If several different classes are in one frame, they are spoken together, person first.

```bash
source .venv/bin/activate
python solarpanel/monitor.py
python solarpanel/monitor.py --source photo.jpg --save --no-speak
```

`--source 0` is the default webcam. Press `q` in the preview to stop. Voice uses `spd-say`. The same alert waits 12 seconds before it repeats; a new class speaks right away.
