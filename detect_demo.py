#!/usr/bin/env python3
"""Edge Security Demo - "the villain": plain object/person detection, NO identity.

Same camera, same edge idea, but the system only knows WHAT is in the frame
(person, phone, laptop, bottle ...), never WHO. Everyone is just "person".
Run it next to intrusion_demo.py to show why detection alone is not security.

Needs: pip install ultralytics opencv-python   (YOLOv8n weights auto-download first run)
"""
import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
AMBER, WHITE, GREY, BLACK = (0, 170, 255), (255, 255, 255), (150, 150, 150), (0, 0, 0)


def text(img, s, org, color=WHITE, scale=0.55):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + 5), BLACK, -1)
    cv2.putText(img, s, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None, help="video file (default: webcam)")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", default="yolov8n.pt", help="any ultralytics weights")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=320, help="smaller = faster on CPU")
    p.add_argument("--persons-only", action="store_true", help="start with person class only (key O toggles)")
    p.add_argument("--no-gui", action="store_true", help="headless run (testing)")
    return p.parse_args(argv)


class Detector:
    def __init__(self, model, conf, imgsz):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("Missing dependency: pip install ultralytics")
        self.model, self.conf, self.imgsz = YOLO(model), conf, imgsz
        self.names = self.model.names

    def __call__(self, frame, persons_only=False):
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, classes=[0] if persons_only else None,
                                 verbose=False)[0]
        return [(tuple(int(v) for v in b.xyxy[0]), self.names[int(b.cls[0])], float(b.conf[0])) for b in res.boxes]


def draw(frame, dets, fps, persons_only):
    out = frame.copy()
    for (x1, y1, x2, y2), name, conf in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), AMBER, 2)
        ly = y1 - 8 if y1 > 100 else min(y2 + 22, out.shape[0] - 30)  # keep label on-screen, clear of HUD
        text(out, f"{name} {conf:.2f}", (x1, ly), AMBER)
    counts = Counter(n for _, n, _ in dets)
    summary = ", ".join(f"{v} {k}" for k, v in counts.most_common()) or "nothing detected"
    h = out.shape[0]
    text(out, "DETECTION ONLY - no identity, no authorisation", (10, 24), AMBER)
    text(out, f"FPS {fps:4.1f}  |  {'persons only' if persons_only else 'person + objects'}  |  {summary}", (10, 52), WHITE, 0.5)
    cv2.putText(out, "q quit | o persons-only/all objects | s snapshot", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, GREY, 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    det = Detector(args.model, args.conf, args.imgsz)
    cap = cv2.VideoCapture(args.source if args.source else args.cam)
    if not args.source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("Camera not opening. Try --cam 1; on macOS allow Camera access for Terminal/IDE.")
    persons_only, fps, last = args.persons_only, 0.0, time.time()
    win = "Edge Security - Detection Only (no identity)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.source:
                frame = cv2.flip(frame, 1)
            dets = det(frame, persons_only)
            now = time.time()
            inst = 1 / max(now - last, 1e-3)
            fps, last = (0.9 * fps + 0.1 * inst if fps else inst), now
            out = draw(frame, dets, fps, persons_only)
            if args.no_gui:
                continue
            cv2.imshow(win, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("o"):
                persons_only = not persons_only
            elif key == ord("s"):
                path = ROOT / f"detect_{datetime.now():%H%M%S}.jpg"
                cv2.imwrite(str(path), out)
                print(f"saved {path.name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
