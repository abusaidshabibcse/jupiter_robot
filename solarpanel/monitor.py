#!/usr/bin/env python3
"""Watch a camera and speak when something harmful is on the solar panel.

Uses the fine-tuned YOLO26s weights in yolo/runs/yolo26s/weights/best.pt.
Only person, dog, cat, bird, and leaf are announced. Other classes are ignored.

    python solarpanel/monitor.py
    python solarpanel/monitor.py --source 0
    python solarpanel/monitor.py --source photo.jpg --save

Press q in the preview window to stop a live camera.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import alerts


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "yolo" / "runs" / "yolo26s" / "weights" / "best.pt"
OUT_DIR = Path(__file__).resolve().parent / "out"


def parse_source(raw: str):
    if raw.isdigit():
        return int(raw)
    return raw


def speak(text: str, enabled: bool) -> None:
    print(text)
    if not enabled:
        return
    speaker = shutil.which("spd-say")
    if speaker is None:
        print("Voice unavailable: spd-say is not installed.")
        return
    # Drop a new line if the previous one is still playing.
    proc = getattr(speak, "_proc", None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
    speak._proc = subprocess.Popen(  # type: ignore[attr-defined]
        [speaker, "--wait", text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def should_repeat(signature: tuple, now: float, state: dict, cooldown: float) -> bool:
    if not signature:
        state["signature"] = ()
        return False
    if signature != state["signature"] or now - state["spoken_at"] >= cooldown:
        state["signature"] = signature
        state["spoken_at"] = now
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or a photo or video path.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cooldown", type=float, default=12.0, help="Seconds before the same alert repeats.")
    parser.add_argument("--no-speak", action="store_true", help="Print the alert without voice.")
    parser.add_argument("--no-show", action="store_true", help="Skip the preview window.")
    parser.add_argument("--save", action="store_true", help="Save annotated frames under solarpanel/out/.")
    args = parser.parse_args()

    weights = args.weights
    if not weights.is_file():
        raise SystemExit(
            f"Weights not found: {weights}\nTrain first with: python yolo/train_yolo26s.py"
        )

    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {int(k): v for k, v in model.names.items()}
    keep = alerts.class_ids(names)
    missing = [name for name in alerts.ORDER if name not in names.values()]
    if missing:
        raise SystemExit(f"Weights are missing alert classes: {', '.join(missing)}")

    source = parse_source(args.source)
    live = isinstance(source, int)
    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using weights: {weights}")
    print("Watching for: " + ", ".join(alerts.ORDER))

    results = model.predict(
        source=source,
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        classes=keep,
        verbose=False,
    )

    state = {"signature": (), "spoken_at": 0.0}
    window = "Solar panel"
    shown = False

    for index, result in enumerate(results):
        found = []
        if result.boxes is not None and len(result.boxes):
            found = result.boxes.cls.cpu().numpy().astype(int).tolist()
        counts = alerts.counts_from_boxes(found, names)
        text = alerts.message_for(counts)
        signature = tuple((name, counts[name]) for name in alerts.ORDER if counts.get(name))
        if text and should_repeat(signature, time.monotonic(), state, args.cooldown):
            speak(text, enabled=not args.no_speak)

        frame = result.plot()
        if args.save:
            dest = OUT_DIR / f"frame_{index:05d}.jpg"
            import cv2

            cv2.imwrite(str(dest), frame)
            if not live:
                print(dest)

        if not args.no_show and live:
            import cv2

            cv2.imshow(window, frame)
            shown = True
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if shown:
        import cv2

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
