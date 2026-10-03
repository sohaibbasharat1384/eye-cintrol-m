# Usage Guide

1. [Setup](#setup)
2. [The Main Window](#the-main-window)
3. [Calibration](#calibration)
4. [Tracking](#tracking)
5. [Click Modes](#click-modes)
6. [Settings](#settings)
7. [Settings Storage](#settings-storage)
8. [Tips](#tips)
9. [Troubleshooting](#troubleshooting)
10. [Scripts](#scripts)
11. [Python API](#python-api)

## Setup

- **Lighting**: bright, even light on your face; avoid a window behind you.
- **Camera**: at eye level, 50–70 cm away, lens clean.
- **Seating**: a stable position you can keep; calibration is only valid for roughly the head position you calibrated in.
- **Display**: calibration and cursor control use the primary monitor.

Start the app with `eyemouse` (see [INSTALL.md](INSTALL.md)). The first start downloads the face landmark model once.

## The Main Window

- **Left**: camera preview (mirrored) with eye contours (green), iris centers (yellow) and the combined gaze point (magenta). Below it: status line, FPS, processing latency and face detection state.
- **Right**:
  - *Tracking Control*: camera selection, **Enable Tracking**, **Click Mode**
  - *Calibration*: **Start Calibration**, **Auto-confirm when gaze is stable**, progress, **Save** / **Load**
  - *Settings*: smoothing, blink threshold, dwell time

If the status says "No face detected", improve the lighting, move closer and center your face.

## Calibration

Calibration is required before tracking: it learns how your eye position maps to screen positions.

1. Click **Start Calibration** or press `C`.
2. A dark fullscreen overlay shows one target at a time (9 in total, in a 3×3 grid).
3. Look at the center of the target and hold still. The outer ring turns **green** when enough stable samples are collected.
4. Press `Space` to confirm the point. Pressing it while the ring is still red shows "Not stable yet" and keeps the point.
5. After the last point the overlay closes and the calibration error is shown.

**Auto-confirm**: with **Auto-confirm when gaze is stable** ticked, each point is confirmed automatically once the ring turns green, so no key press is needed. After a new target appears the app waits about 0.8 s, so your eyes have reached it, before it starts judging stability.

**Cancel**: press `Esc` (or close the overlay) at any time. If you were already calibrated, the previous calibration is kept.

**Calibration error**: the reported error is how closely the fitted mapping reproduces the 9 calibration points, in screen pixels. It is a quick sanity check, not a measurement of real accuracy; use `scripts/evaluate.py` for that.

| Error | Meaning |
|-------|---------|
| < 50 px | Excellent |
| 50–100 px | Good |
| 100–150 px | Acceptable, consider recalibrating |
| > 150 px | Poor, recalibrate |

**Saved automatically**: every successful calibration is saved to `calibration.json` in your settings folder (see [Settings Storage](#settings-storage)) and restored the next time the app starts, so you do not have to calibrate every session. **Save** and **Load** write and re-read that file manually. A saved calibration is only useful with the same screen resolution, seating position, camera position and lighting, so recalibrate when those change. Loading fails if the screen resolution differs, or if the file was saved by an older version that used a different gaze feature (recalibrate in that case).

Recalibrate after moving your chair, monitor or camera, changing lighting, or putting on/removing glasses.

## Tracking

- Click **Enable Tracking** or press `Space`. The button turns green and the cursor follows your gaze.
- Press `Space` (with the app window focused) or click the button again to stop.
- If you enable tracking without a calibration, the app offers to calibrate first. Without calibration the cursor only moves in a small area around the screen center.

`Space`, `C` and `Q` only work while the Eye Control Mouse window has focus. Once the cursor follows your gaze, other windows easily get the focus, so use the **emergency stop** `Ctrl+Alt+Space`: it works in any application, disables tracking and cancels a running calibration. The key is shown under the Enable Tracking button; if it shows as unavailable (see [INSTALL.md](INSTALL.md)), keep a physical mouse at hand.

To use a different key: `eyemouse --stop-hotkey "<ctrl>+<shift>+s"` (pynput syntax: modifiers and special keys in angle brackets, joined with `+`).

Pipeline per frame: camera image → MediaPipe face landmarks → gaze feature (each iris position relative to its own eye corners, averaged over both eyes; unaffected by head tilt and sideways movement) → calibrated mapping → smoothing → cursor move.

While one or both eyes are closed (eye aspect ratio below the blink threshold) the cursor stays where it was: the iris cannot be seen reliably then, and this keeps blink and wink clicks on the spot you were looking at.

## Click Modes

Pick a mode in **Click Mode**. Start with **None** until cursor movement feels right.

### None
No automatic clicks; click with a physical mouse or keyboard.

### Blink
A left click on a **deliberate, long blink**: close both eyes for about half a second (at least 0.4 s, at most 2 s) and open them again; the click happens when they reopen. Clicks are at least 0.5 s apart.

Normal involuntary blinks (100–300 ms) and closing only one eye do not click, and keeping your eyes closed longer than 2 s (resting them) is ignored.

### Dwell
A left click when the cursor stays within 30 px of the same spot for the **Dwell Time** (default 600 ms). The next dwell can start 0.5 s after a click, so holding your gaze on one spot clicks repeatedly.

While you dwell, a green ring around the spot fills up clockwise; the click happens when it is complete. The ring does not block clicks and disappears when you look away.

### Wink (L/R)
Close one eye while keeping the other open for at least 3 frames: left eye → left click, right eye → right click. Winks are at least 0.6 s apart. Harder to perform consistently, and may work less well with glasses.

## Settings

### Smoothing (1–100, default 75)
How much the cursor movement is smoothed. **Higher values make the cursor steadier but slower to follow your gaze; lower values make it faster but more jittery.** The scale is logarithmic, so each step changes the feel by a similar amount: around 50 the cursor settles in a few frames, at 100 it takes about a second.

- Cursor jittery → raise the value
- Cursor lagging → lower the value

### Blink Threshold (0.10–0.40, default 0.21)
An eye counts as closed when its eye aspect ratio (EAR, eye height relative to width; about 0.25–0.35 when open) falls below this value.

- **Higher** threshold → eyes count as closed sooner → clicks trigger more easily (more accidental clicks)
- **Lower** threshold → a firmer, fuller blink is needed

This slider affects Blink mode only; Wink mode uses its own fixed threshold (0.21).

### Dwell Time (300–1500 ms, default 600)
How long the cursor must stay within 30 px to trigger a dwell click. Longer means fewer accidental clicks but slower interaction.

## Settings Storage

The camera, click mode, slider values and the auto-confirm option are saved when you close the app and restored at the next start. They live, together with the calibration, in:

- Windows: `%APPDATA%\eyemouse\` (`settings.json`, `calibration.json`)
- macOS: `~/Library/Application Support/eyemouse/`
- Linux: `$XDG_CONFIG_HOME/eyemouse/` (default `~/.config/eyemouse/`)

Delete `settings.json` to return to the defaults. `--camera N` on the command line overrides the saved camera.

## Tips

- Calibrate carefully and don't rush; the mapping is only as good as the calibration samples.
- Blink normally during calibration; don't force your eyes open.
- Take regular breaks to avoid eye fatigue (for example every 20 minutes).
- Use eye tracking together with a physical mouse/keyboard at first.
- Precision is limited; this is not suitable for pixel-exact work such as drawing.

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| "No face detected" | Better lighting, face centered, closer to the camera |
| Calibration ring never turns green | Keep your head still, improve lighting, sit closer |
| Cursor jittery | Raise the smoothing value; recalibrate |
| Cursor lags behind | Lower the smoothing value |
| Cursor offset or drifting | Recalibrate in your current position |
| Accidental blink clicks | Lower the blink threshold, or use Dwell |
| Blink clicks hard to trigger | Raise the blink threshold |
| Cursor does not move at all (macOS) | Grant Accessibility permission (see [INSTALL.md](INSTALL.md#macos)) |
| Low FPS | Close other apps using the camera/CPU; improve lighting |

## Scripts

```bash
python scripts/verify_installation.py   # check dependencies and camera
python scripts/demo_basic.py            # camera preview with landmarks, no mouse control
python scripts/demo_calibration.py      # calibration + virtual cursor window, no mouse control
python scripts/evaluate.py              # calibration followed by accuracy test points
python scripts/evaluate.py --test-points 6
python scripts/benchmark_camera.py      # where FPS is lost: camera capture vs. processing
```

`evaluate.py` reports the mean, standard deviation and maximum error in pixels on separate test points, plus an approximate visual angle (assuming a 24" monitor at 60 cm).

## Python API

```python
from eyemouse.capture import CameraCapture
from eyemouse.detector import FaceDetector
from eyemouse.tracker import GazeTracker

camera = CameraCapture(camera_id=0)
camera.start()
detector = FaceDetector()
tracker = GazeTracker(screen_width=1920, screen_height=1080)

try:
    while True:
        result = camera.read(timeout=0.1)
        if result is None:
            continue
        frame, timestamp = result
        detection = detector.process(frame)
        if detection is not None:
            cursor_pos = tracker.process(detection)  # (x, y) in screen pixels
finally:
    camera.stop()
    detector.close()
```

Without a mapper (`tracker.set_mapper(...)`, from a completed `CalibrationManager`) the tracker falls back to an uncalibrated mapping.

`CalibrationManager(screen_width, screen_height, num_points=9 | 12 | 16, mapping_method="affine" | "polynomial" | "rbf")` controls the grid size and mapping model.
