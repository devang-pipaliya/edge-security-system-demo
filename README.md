# Edge Security System: Real-Time Intrusion Detection Camera

Live-demo projects for the talk *"Edge Security Systems: Assembling a Real-Time Intrusion Detection Camera"*.
The laptop webcam plays the edge device. Three ways to run it:

| # | Project | What it shows | Run |
|---|---|---|---|
| 1 | **Authorised vs unauthorised** (`intrusion_demo.py`) | Recognises enrolled people (green) and flags strangers (red) with alert, snapshot, clip and log | `python intrusion_demo.py` |
| 2 | **Detection only, "the villain"** (`detect_demo.py`) | Detects person + objects but cannot tell who anyone is | `python detect_demo.py` |
| 3 | **Web version** (`web/`) | Same recognition in the browser with a persons list. Nothing is uploaded, audience can try it | `cd web && python3 -m http.server 8000` |

## Setup (once)
Needs Python 3.10+ (tested on 3.13) and a webcam.
```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # projects 1 and 3 (project 3 needs no Python packages)
pip install -r requirements-detect.txt   # only for project 2 (adds ultralytics + PyTorch, large download)
```
First run downloads models: ~40 MB for project 1, ~6 MB (YOLOv8n) for project 2. Run each once on the
venue Wi-Fi/laptop **before the talk** so they are cached. Project 3 bundles its models (~8 MB).

---

## 1. Authorised vs unauthorised: `intrusion_demo.py`
```bash
python intrusion_demo.py
```
1. A window opens. Nobody is enrolled yet, so your face is a red **UNAUTHORISED** box and an alert fires.
2. Press **E** and look at the camera for ~4 s to enroll yourself (name via `--name`, default `Presenter`).
3. Your box turns green `AUTHORISED: Presenter`. Have someone else step in: red box, flashing border, beep.

**Keys:** `q` quit | `e` enroll | `a` arm/disarm | `m` motion gate on/off | `x` clear enrolled

**Output:** alerts go to `alerts/` (`intruder_*.jpg`, `intruder_*.mp4` with 3 s before + 3 s after, `events.jsonl`).
Enrolled faces are kept in `authorized/<name>/`. You can also drop a clear front-facing photo at
`authorized/<name>.jpg`.

| Flag | Purpose |
|---|---|
| `--cam 1` | Different camera (USB, Continuity Camera) |
| `--source file.mp4` | Run on a video file instead of the webcam |
| `--name Devang` | Name used when pressing `E` (default `Presenter`) |
| `--threshold 0.45` | Stricter face matching (default `0.363`, higher = stricter) |
| `--min-frames 8` | Consecutive unknown frames before an alert (default 5) |
| `--motion-gate` | Start with the motion gate on (detector sleeps when the scene is static) |
| `--cooldown 8` | Seconds between alerts |
| `--no-gui` | Headless run (testing) |

## 2. Detection only, "the villain": `detect_demo.py`
```bash
python detect_demo.py
```
Amber boxes for `person` and everyday objects (phone, laptop, bottle...). Every person is just "person":
it knows *what* is there, never *who*. **Keys:** `q` quit | `o` persons-only / all objects | `s` snapshot.
Flags: `--cam`, `--source`, `--persons-only`, `--conf 0.4`, `--imgsz 320` (smaller = faster on CPU), `--no-gui`.

## 3. Web version: `web/`
```bash
cd web
python3 -m http.server 8000 --bind 127.0.0.1    # then open http://localhost:8000
```
Cameras only work on a secure origin: use **`http://localhost:8000`** (or `127.0.0.1`), never `0.0.0.0`
or a LAN IP (those show "Camera API unavailable"), and do not open `index.html` via `file://`.
Click **Start camera**, allow the permission, then use the panels on the right:

- **Authorised persons:** enroll by name (8 samples), remove, export/import as JSON.
  Faces are stored only in that browser's localStorage.
- **Persons in view / Session roster:** who is in frame now, and everyone seen this session.
- **Alert log:** snapshot, time and a Save link, plus beep and flashing banner on intruders.
- **Zones:** draw green and red polygons on the video (`+ Green zone` / `+ Red zone`, click the corners, then click
  the first point, double-click or press Enter to finish; Backspace undoes a point, Esc cancels; `Quick split` makes a
  left-green / right-red demo layout). Zones are remembered in the browser. The rules:

  | Person is in the... | Enrolled | Not enrolled |
  |---|---|---|
  | **Green zone** | Authorised | Unauthorised, alert |
  | **Red zone** (restricted) | Not allowed, alert (tick *Enrolled people may enter red zone* to allow) | Unauthorised, alert |
  | **Outside both zones** | Shown, no alert | Shown, no alert |
  | **No zones drawn** | Authorised | Unauthorised, alert (whole frame monitored) |

  Where zones overlap, red wins. A person is placed by the centre of their face box.
- **Detection only** toggle (the "villain" view), **Match tolerance** (lower = stricter), **Alert after N frames**.

**Public link (GitHub Pages):** repo Settings -> Pages -> Source: *GitHub Actions*. The workflow
`.github/workflows/pages.yml` then publishes `web/` on every push to `main`
(`https://<user>.github.io/<repo>/`). Any static host works (Cloudflare Pages, Netlify).

Speed depends on the device: WebGL (GPU) when available, otherwise a much slower CPU fallback.

---

## Troubleshooting
| Problem | Fix |
|---|---|
| Camera won't open | `--cam 1`. **macOS:** System Settings -> Privacy & Security -> Camera, enable your terminal/IDE, restart it |
| Webcam fails on stage | `--source backup.mp4` runs the same pipeline on a recorded video |
| Model download fails (project 1) | Put `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` from [opencv_zoo](https://github.com/opencv/opencv_zoo) into `models/` |
| Stranger accepted as authorised | `--threshold 0.45` (web: lower the Match tolerance) |
| You get flagged unauthorised | Enroll again in the current lighting, `--threshold 0.30` |
| Alert flickers | `--min-frames 8` (web: raise Alert after N frames) |
| Low FPS | Keep the default 640x480; project 2: `--imgsz 320` or `--persons-only` |
| Web page: "Camera API unavailable" | Open `http://localhost:8000`, not `http://0.0.0.0:8000` or a LAN IP. Hosted links need HTTPS (GitHub Pages has it) |
| Web page: "Could not load models" | Serve it over http(s) (`python3 -m http.server`), not `file://` |

## How it works
camera -> (optional MOG2 motion gate) -> YuNet face detection -> SFace embedding -> match vs enrolled faces
-> tracker + identity smoothing -> N consecutive unknown frames -> alert (snapshot, clip, log, beep).
The web version follows the same steps with face-api.js (TinyFaceDetector + 128-d embeddings).
Porting project 1 to a Pi/Hailo means replacing the `FaceEngine` class only.

## Limits to mention
- Indoor, well-lit, frontal faces only. Not tested for night/IR/rain.
- A printed photo or screen can fool it (no liveness check). Demo, not a certified security product.
- Alerts are a beep plus files. MQTT/Telegram would go in `Pipeline.notify()` in `intrusion_demo.py`.

## Repo layout
```
intrusion_demo.py        project 1: authorised vs unauthorised (OpenCV YuNet + SFace)
detect_demo.py           project 2: detection only (YOLOv8)
web/                     project 3: browser version (static; models + face-api.js bundled)
requirements.txt         projects 1 and 3 (Python side)
requirements-detect.txt  adds ultralytics for project 2
```
