# Installation Guide

## Prerequisites

- **Python** 3.9 or newer (tested with 3.14 on Windows)
- **Webcam**: built-in or USB
- **OS**: Windows 10/11, macOS, or Linux with an X11 session (see [Linux](#linux))

Pre-built installers and a PyPI package are not available yet; install from source.

## Install from Source

```bash
# 1. Clone
git clone https://github.com/sohaibbasharat1384/eye-control-mouse.git
cd eye-control-mouse

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install
pip install -e .
```

`pip install -e .` installs the dependency ranges declared in `pyproject.toml` and the `eyemouse` command. To reproduce the exact versions the project is tested with, run `pip install -r requirements.txt` first. For development tools (pytest, linters, PyInstaller) use `pip install -r requirements-dev.txt`.

### Run

```bash
eyemouse                      # installed command
python -m eyemouse            # as a module
python src/eyemouse/app.py    # directly from the checkout
```

### Verify

```bash
python scripts/verify_installation.py   # checks dependencies and camera access
python scripts/demo_basic.py            # camera preview with eye/iris landmarks
pytest                                  # test suite (needs requirements-dev.txt)
python -c "import eyemouse; print(eyemouse.__version__)"
```

## Face Landmark Model

Face and iris landmarks come from MediaPipe's Face Landmarker, which needs a model file (`face_landmarker.task`, ~3.7 MB). The app looks for it in this order:

1. The path in the `EYEMOUSE_FACE_MODEL` environment variable
2. `src/eyemouse/models/face_landmarker.task` (a copy shipped with the package)
3. The per-user cache directory:
   - Windows: `%LOCALAPPDATA%\eyemouse\`
   - macOS: `~/Library/Caches/eyemouse/`
   - Linux: `$XDG_CACHE_HOME/eyemouse/` (default `~/.cache/eyemouse/`)

If none exists, it is downloaded once into the cache directory from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`.
For offline machines, download that file elsewhere and either copy it into one of the locations above or point `EYEMOUSE_FACE_MODEL` at it.

## Platform Notes

### Windows

**Camera permission**: Settings → Privacy & security → Camera → enable "Let apps access your camera" and "Let desktop apps access your camera".

Display scaling (125%, 150%, ...) is supported; calibration targets and cursor positions are converted automatically.

### macOS

macOS asks for camera access on first run. To grant it manually: System Settings → Privacy & Security → Camera, and enable the terminal you start the app from.

Moving and clicking the mouse also needs **Accessibility** permission: System Settings → Privacy & Security → Accessibility → enable the same terminal. Without it the cursor will not move.

### Linux

- **Camera access**: your user must be in the `video` group:
  ```bash
  sudo usermod -a -G video $USER   # then log out and back in
  ```
- **Cursor control requires X11.** PyAutoGUI cannot move the pointer in a native Wayland session; log in with an X11 session ("GNOME on Xorg" or similar).
- **Qt runtime library**: PyQt6's X11 platform plugin needs `libxcb-cursor0`:
  ```bash
  sudo apt-get install libxcb-cursor0      # Debian/Ubuntu
  sudo dnf install xcb-util-cursor         # Fedora
  ```
- Install all Python packages with pip inside the virtual environment; do not mix in distribution packages such as `python3-opencv` or `python3-pyqt6`.

## Troubleshooting

### Camera not detected

1. Check the camera works in another application and that no other app is using it.
2. Try another camera index: `eyemouse --camera 1` (or pick it in the **Camera** box).
3. Check camera permissions (see above).

### `ImportError` / crashes when importing `cv2`

MediaPipe depends on `opencv-contrib-python`. Having `opencv-python` (or `opencv-python-headless`) installed in the same environment as well makes the packages overwrite each other's `cv2`. Fix:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-contrib-python
```

### `module 'mediapipe' has no attribute 'solutions'`

You are running an old copy of eyemouse with a new MediaPipe. Current versions use the Face Landmarker task API; update your checkout and reinstall with `pip install -e .`.

### "Could not download the face landmark model"

The first start needs internet access to fetch the model. Download it manually and set `EYEMOUSE_FACE_MODEL` (see [Face Landmark Model](#face-landmark-model)).

### Low FPS

First find out whether the camera or the processing is the limit. Close Eye Control Mouse and run:

```bash
python scripts/benchmark_camera.py
```

It reports the raw capture FPS for each camera backend and format, the face detection time per frame, and which of the two is the bottleneck. Face detection normally takes well under 20 ms per frame on a modern CPU, so a low FPS usually comes from the camera:

1. **Lighting**: many webcams lower their frame rate in dim light to expose longer. Brighter, even light often raises the FPS.
2. **Virtual machines**: a webcam passed through to a VM (for example "VirtualBox Webcam") is often limited to a few FPS by the virtualization layer. Run the app on the host system for usable frame rates.
3. **Other applications** using the camera or a lot of CPU.

The FPS shown under the video preview is the rate of fully processed frames; run with `--debug` for more detail.

## Uninstall

```bash
pip uninstall eyemouse
```

Also delete the cached model directory listed in [Face Landmark Model](#face-landmark-model) and any `calibration_data.json` you saved.
