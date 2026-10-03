# Eye Control Mouse

> Hands-free mouse control using a standard webcam and eye tracking. All video processing happens locally.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Control your computer's mouse cursor with your eyes and an ordinary webcam. Intended for accessibility, hands-free computing and experimentation.

**Status: beta.** The full pipeline (camera → face/iris landmarks → calibration → cursor movement → clicks) is implemented, but end-to-end accuracy and latency have not been benchmarked yet. Use `python scripts/evaluate.py` to measure them on your own setup.

## Features

- Gaze-based cursor control from a regular webcam (MediaPipe Face Landmarker, 478 landmarks including irises)
- Fullscreen 9-point calibration with a stability indicator, optional auto-confirm, and cancel with `Esc` (keeps the previous calibration)
- Affine, polynomial and RBF gaze-to-screen mapping
- Kalman (default) or EWMA smoothing
- Click modes: blink, dwell, left/right wink
- Save and load calibration
- Display-scaling aware (Windows high-DPI)
- Runs on Windows, macOS and Linux (Python 3.9+)

## Privacy

All video frames are processed in memory on your device and are never saved or sent anywhere. The only network access is a one-time download of the MediaPipe face landmark model (~3.7 MB) the first time the app starts; see [INSTALL.md](INSTALL.md#face-landmark-model) to provide the model file yourself instead.

## Quick Start

```bash
git clone https://github.com/sohaibbasharat1384/eye-control-mouse.git
cd eye-control-mouse

python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate

pip install -e .
python scripts/verify_installation.py
eyemouse
```

Then:

1. **Calibrate** – click **Start Calibration** (or press `C`). Look at the center of each target; when its ring turns green press `Space`. Tick **Auto-confirm when gaze is stable** to skip pressing `Space`. `Esc` cancels.
2. **Track** – click **Enable Tracking** (or press `Space`). The cursor now follows your gaze.
3. **Click** – choose a **Click Mode** (start with *None* until tracking feels right).

Tips: even lighting on your face, camera at eye level about 50–70 cm away, keep your head still while calibrating, and sit the way you will use it.

See [USAGE.md](USAGE.md) for the full guide and [INSTALL.md](INSTALL.md) for platform-specific setup and troubleshooting.

## Keyboard Shortcuts

| Key | Main window | Calibration overlay |
|-----|-------------|---------------------|
| `Space` | Enable/disable tracking | Confirm the current point |
| `C` | Start calibration | – |
| `Q` | Quit | – |
| `Esc` | – | Cancel calibration |

## Command Line

```bash
eyemouse                 # run the app
eyemouse --camera 1      # use another camera
eyemouse --debug         # verbose logging
python -m eyemouse       # same as eyemouse
```

## How It Works

```
Camera thread ──frames──▶ Processing thread (QThread) ──signals──▶ GUI thread
                             │
                             ├─ MediaPipe Face Landmarker → eye, iris and face landmarks
                             ├─ gaze feature: each iris relative to its own eye corners
                             ├─ calibrated mapping → screen coordinates
                             ├─ Kalman / EWMA smoothing
                             └─ cursor move (PyAutoGUI) + blink / wink / dwell clicks
```

## Project Layout

```
src/eyemouse/
  app.py           entry point and command line
  gui.py           PyQt6 window, processing thread, calibration overlay
  capture.py       threaded camera capture
  detector.py      MediaPipe Face Landmarker wrapper (model download/lookup)
  tracker.py       gaze tracker and gaze-to-screen mappers
  calibration.py   calibration workflow, save/load, evaluation
  clicker.py       mouse control and blink/wink/dwell detection
  utils.py         filters, eye aspect ratio, stability buffer, metrics
scripts/           demos, accuracy evaluation, installation check
tests/             pytest suite
```

## Development

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, the manual test checklist and how to build a standalone executable.

## License

MIT License – see [LICENSE](LICENSE).

## Citation

If you use this in research, please cite:

```
@software{eyemouse2025,
  title={Eye Control Mouse: Webcam-based Gaze Tracking},
  year={2025}
}
```

## Acknowledgments

- [MediaPipe](https://developers.google.com/mediapipe) by Google for face and iris landmarks
- OpenCV community
- PyQt6 project
