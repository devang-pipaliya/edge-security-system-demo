#!/usr/bin/env python3
"""Edge Security Demo - authorised vs unauthorised person detection (laptop webcam).

Pipeline:  camera -> [motion gate] -> face detect (YuNet) -> face embed (SFace)
           -> match against enrolled faces -> track + persistence -> alert

The laptop plays the edge device. On a Pi/Hailo only FaceEngine changes.
Only needs: pip install opencv-python numpy   (models auto-download on first run)
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
DET_FILE = "face_detection_yunet_2023mar.onnx"
REC_FILE = "face_recognition_sface_2021dec.onnx"
DET_URL = ZOO + "face_detection_yunet/" + DET_FILE
REC_URL = ZOO + "face_recognition_sface/" + REC_FILE

GREEN, RED, AMBER, WHITE, GREY = (60, 200, 60), (50, 50, 230), (0, 170, 255), (255, 255, 255), (150, 150, 150)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_model(name, url):
    path = MODELS / name
    if path.exists() and path.stat().st_size > 100_000:
        return str(path)
    MODELS.mkdir(exist_ok=True)
    print(f"Downloading {name} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        path.unlink(missing_ok=True)
        sys.exit(f"Could not download {name}: {e}\nDownload it manually from {url} into {MODELS}/")
    return str(path)


class FaceEngine:
    """Swap this class for a Hailo/TFLite implementation on the Pi."""

    def __init__(self, det_thresh=0.7):
        self.det = cv2.FaceDetectorYN.create(ensure_model(DET_FILE, DET_URL), "", (320, 320), det_thresh, 0.3, 20)
        self.rec = cv2.FaceRecognizerSF.create(ensure_model(REC_FILE, REC_URL), "")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame)
        return [] if faces is None else list(faces)

    def embed(self, frame, face):
        return self.rec.feature(self.rec.alignCrop(frame, face))

    def similarity(self, a, b):
        return self.rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE)


class Gallery:
    """Enrolled (authorised) people -> list of face embeddings."""

    def __init__(self, engine, folder):
        self.engine, self.folder, self.people = engine, Path(folder), {}
        self.load()

    def load(self):
        self.people = {}
        if not self.folder.exists():
            return
        for p in sorted(self.folder.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            name = p.parent.name if p.parent != self.folder else p.stem
            img = cv2.imread(str(p))
            faces = self.engine.detect(img) if img is not None else []
            if not faces:
                print(f"  skipped {p.name}: no face found")
                continue
            face = max(faces, key=lambda f: f[2] * f[3])
            self.people.setdefault(name, []).append(self.engine.embed(img, face))

    def add(self, name, feat):
        self.people.setdefault(name, []).append(feat)

    def identify(self, feat, threshold):
        best_name, best = None, -1.0
        for name, feats in self.people.items():
            s = max(self.engine.similarity(feat, f) for f in feats)
            if s > best:
                best_name, best = name, s
        return (best_name if best >= threshold else None), best


class Tracker:
    """Nearest-centroid tracker; smooths identity over recent frames."""

    def __init__(self, history=7, max_age=1.0):
        self.tracks, self.next_id, self.history, self.max_age = [], 1, history, max_age

    def update(self, detections, now):
        """detections: list of dict(box=(x,y,w,h), name=str|None, score=float)."""
        for t in self.tracks:
            t["matched"] = False
        for d in detections:
            x, y, w, h = d["box"]
            cx, cy = x + w / 2, y + h / 2
            best, best_dist = None, w * 1.5
            for t in self.tracks:
                dist = np.hypot(t["cx"] - cx, t["cy"] - cy)
                if not t["matched"] and dist < best_dist:
                    best, best_dist = t, dist
            if best is None:
                best = {"id": self.next_id, "hist": deque(maxlen=self.history), "unknown_streak": 0, "alerted": False}
                self.next_id += 1
                self.tracks.append(best)
            best.update(cx=cx, cy=cy, box=d["box"], score=d["score"], seen=now, matched=True)
            best["hist"].append(d["name"])
            best["unknown_streak"] = best["unknown_streak"] + 1 if d["name"] is None else 0
            votes = Counter(best["hist"]).most_common(1)[0][0]
            best["name"] = votes
        self.tracks = [t for t in self.tracks if now - t["seen"] < self.max_age]
        return [t for t in self.tracks if t["matched"]]


class MotionGate:
    """MOG2 on a small frame; sleeps the detector when the scene is static."""

    def __init__(self, min_pixels=300, hold=3.0):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=True)
        self.min_pixels, self.hold, self.last_motion, self.mask = min_pixels, hold, 0.0, None

    def active(self, frame, now):
        small = cv2.resize(frame, (320, 180))
        mask = self.bg.apply(small)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # drop shadows (value 127)
        self.mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(self.mask) > self.min_pixels:
            self.last_motion = now
        return now - self.last_motion < self.hold


class Recorder:
    """Pre-roll ring buffer + post-roll clip writer."""

    def __init__(self, folder, pre=3.0, post=3.0):
        self.folder, self.pre, self.post = folder, pre, post
        self.buf = deque()  # (t, frame)
        self.writer, self.until = None, 0.0

    def push(self, frame, now):
        self.buf.append((now, frame.copy()))
        while self.buf and now - self.buf[0][0] > self.pre:
            self.buf.popleft()
        if self.writer:
            self.writer.write(frame)
            if now >= self.until:
                self.writer.release()
                self.writer = None

    def start(self, path, fps, now):
        if self.writer:
            self.until = now + self.post
            return
        h, w = self.buf[-1][1].shape[:2]
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 5), (w, h))
        for _, f in self.buf:
            self.writer.write(f)
        self.until = now + self.post

    def close(self):
        if self.writer:
            self.writer.release()


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.engine = FaceEngine(args.det_conf)
        self.gallery = Gallery(self.engine, args.authorized)
        self.tracker = Tracker()
        self.gate = MotionGate(args.motion_pixels, args.hold) if args.motion_gate else None
        self.recorder = Recorder(Path(args.alerts))
        self.armed = True
        self.frames = self.skipped = self.alerts = 0
        self.last_alert = -1e9
        self.alert_banner_until = 0.0
        self.fps, self._t = 0.0, time.time()
        self.enroll_left, self.enroll_next = 0, 0.0
        Path(args.alerts).mkdir(parents=True, exist_ok=True)
        Path(args.authorized).mkdir(parents=True, exist_ok=True)

    # -- operator actions
    def start_enroll(self, n=8):
        self.enroll_left, self.enroll_next = n, 0.0

    def toggle_gate(self):
        self.gate = None if self.gate else MotionGate(self.args.motion_pixels, self.args.hold)

    # -- one frame
    def process(self, frame, now):
        a = self.args
        self.frames += 1
        inst = 1 / max(now - self._t, 1e-3)
        self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else 10.0
        self._t = now
        awake = (self.gate.active(frame, now) if self.gate else True) or self.enroll_left > 0
        tracks = []
        if awake:
            dets, faces = [], self.engine.detect(frame)
            for f in faces:
                feat = self.engine.embed(frame, f)
                name, score = self.gallery.identify(feat, a.threshold)
                box = tuple(int(v) for v in f[:4])
                dets.append({"box": box, "name": name, "score": score, "feat": feat, "conf": float(f[14])})
            self._enroll(frame, now, dets)
            tracks = self.tracker.update(dets, now)
        else:
            self.skipped += 1

        out = frame.copy()
        alert_track = None
        for t in tracks:
            known = t["name"] is not None
            color = GREEN if known else RED
            x, y, w, h = t["box"]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = f"AUTHORISED: {t['name']}" if known else "UNAUTHORISED"
            if self.gallery.people:
                label += f" ({t['score']:.2f})"
            ly = y - 8 if y > 100 else y + h + 22  # keep label clear of the HUD
            self._text(out, label, (x, ly), color)
            if not known and t["unknown_streak"] >= a.min_frames and not t["alerted"]:
                alert_track = t
        if alert_track and self.armed and now - self.last_alert > a.cooldown:
            alert_track["alerted"] = True
            self._alert(out, alert_track, now)

        self._hud(out, tracks, awake, now)
        self.recorder.push(out, now)
        return out

    def _enroll(self, frame, now, dets):
        if self.enroll_left <= 0 or now < self.enroll_next:
            return
        if len(dets) == 1 and dets[0]["conf"] > 0.85:
            self.gallery.add(self.args.name, dets[0]["feat"])
            d = Path(self.args.authorized) / self.args.name
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{datetime.now():%H%M%S_%f}.jpg"), frame)
            self.enroll_left -= 1
            self.enroll_next = now + 0.4
            print(f"enrolled {self.args.name}: {self.enroll_left} samples left")

    def _alert(self, out, track, now):
        self.alerts += 1
        self.last_alert = now
        self.alert_banner_until = now + 3.0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = Path(self.args.alerts) / f"intruder_{stamp}.jpg"
        cv2.imwrite(str(snap), out)
        self.recorder.start(Path(self.args.alerts) / f"intruder_{stamp}.mp4", self.fps, now)
        event = {"time": datetime.now().isoformat(timespec="seconds"), "event": "unauthorised_person",
                 "best_match_score": round(float(track["score"]), 3), "snapshot": snap.name}
        with open(Path(self.args.alerts) / "events.jsonl", "a") as fh:
            fh.write(json.dumps(event) + "\n")
        self.notify(event)

    def notify(self, event):
        """Hook for MQTT / Telegram / webhook."""
        sys.stdout.write("\a")
        print(f"ALERT {event}")
        sys.stdout.flush()

    @staticmethod
    def _text(img, text, org, color, scale=0.6):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x, y = org
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + 5), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def _hud(self, out, tracks, awake, now):
        h, w = out.shape[:2]
        if now < self.alert_banner_until:
            if int(now * 4) % 2 == 0:
                cv2.rectangle(out, (0, 0), (w - 1, h - 1), RED, 10)
            msg = "INTRUDER ALERT - UNAUTHORISED PERSON"
            tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            self._text(out, msg, (max((w - tw) // 2 - 4, 4), h - 40), RED, 0.8)
        n_auth = sum(1 for t in tracks if t["name"])
        n_unk = len(tracks) - n_auth
        status = ("ARMED" if self.armed else "DISARMED", GREEN if self.armed else AMBER)
        mode = "always on" if not self.gate else ("MOTION: DETECTOR ON" if awake else "IDLE: DETECTOR SLEEPING")
        self._text(out, f"{status[0]}  |  {mode}", (10, 24), status[1])
        skipped = 100 * self.skipped / max(self.frames, 1)
        self._text(out, f"FPS {self.fps:4.1f}  inference skipped {skipped:3.0f}%  authorised {n_auth}  unknown {n_unk}  "
                        f"alerts {self.alerts}", (10, 52), WHITE, 0.5)
        if self.enroll_left > 0:
            self._text(out, f"ENROLLING '{self.args.name}': look at camera ({self.enroll_left} left)", (10, 80), AMBER)
        elif not self.gallery.people:
            self._text(out, "No authorised faces yet - press E to enroll yourself", (10, 80), AMBER)
        cv2.putText(out, "q quit | e enroll | a arm/disarm | m motion gate | x clear enrolled", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)

    def clear_enrolled(self):
        self.gallery.people = {}
        for p in Path(self.args.authorized).rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                p.unlink()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None, help="video file path (default: webcam)")
    p.add_argument("--cam", type=int, default=0, help="webcam index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--name", default="Presenter", help="name used when pressing E to enroll")
    p.add_argument("--authorized", default=str(ROOT / "authorized"), help="folder of authorised faces")
    p.add_argument("--alerts", default=str(ROOT / "alerts"))
    p.add_argument("--threshold", type=float, default=0.363, help="cosine match threshold (higher = stricter)")
    p.add_argument("--det-conf", type=float, default=0.7, help="face detection confidence")
    p.add_argument("--min-frames", type=int, default=5, help="consecutive unknown frames before alert")
    p.add_argument("--cooldown", type=float, default=8.0, help="seconds between alerts")
    p.add_argument("--motion-gate", action="store_true", help="sleep detector when scene is static (key M toggles)")
    p.add_argument("--motion-pixels", type=int, default=300)
    p.add_argument("--hold", type=float, default=3.0, help="seconds to keep detector on after motion stops")
    p.add_argument("--no-gui", action="store_true", help="headless run (testing)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    pipe = Pipeline(args)
    print(f"Authorised people loaded: {', '.join(pipe.gallery.people) or 'none'}")
    cap = cv2.VideoCapture(args.source if args.source else args.cam)
    if not args.source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("Camera not opening. Try --cam 1; on macOS allow Camera access for Terminal/IDE.")
    win = "Edge Security - Intrusion Detection"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.source:
                frame = cv2.flip(frame, 1)  # mirror view feels natural on stage
            out = pipe.process(frame, time.time())
            if args.no_gui:
                continue
            cv2.imshow(win, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                pipe.start_enroll()
            elif key == ord("a"):
                pipe.armed = not pipe.armed
            elif key == ord("m"):
                pipe.toggle_gate()
            elif key == ord("x"):
                pipe.clear_enrolled()
    finally:
        pipe.recorder.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
