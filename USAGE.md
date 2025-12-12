# Usage Guide

Complete guide to using Eye Control Mouse.

## Table of Contents

1. [First Time Setup](#first-time-setup)
2. [Calibration](#calibration)
3. [Tracking Modes](#tracking-modes)
4. [Click Detection](#click-detection)
5. [Settings & Tuning](#settings--tuning)
6. [Tips for Best Results](#tips-for-best-results)
7. [Command Line Usage](#command-line-usage)

## First Time Setup

### 1. Environment Setup

For optimal performance:

- **Lighting**: Good, even lighting on your face. Avoid backlighting (window behind you)
- **Camera**: Position camera at eye level, 50-70cm away
- **Seating**: Stable seating position, minimize head movement during use
- **Display**: Single monitor recommended for first use

### 2. Launch Application

```bash
eyemouse
```

The main window shows:
- Left: Camera preview with detected landmarks
- Right: Control panel with settings

### 3. Verify Face Detection

- You should see green landmarks on your eyes
- Face bounding box (green rectangle)
- Magenta dot showing gaze point
- Confidence score should be > 0.8

If no face detected:
- Check camera permissions
- Improve lighting
- Move closer to camera
- Center your face in the frame

## Calibration

**Required before first use!**

### Standard 9-Point Calibration

1. Click **"Start Calibration"** button
2. Look at the red circle that appears
3. Keep looking until "STABLE" message appears
4. Press **SPACE** to confirm
5. Repeat for all 9 points
6. Calibration complete - you'll see the average error

**Tips for accurate calibration:**
- Keep your head still during calibration
- Look directly at each point center
- Don't move on to next point until stable
- Blink normally - don't force eyes open
- Sit in the same position you'll use the system

### Calibration Quality

- **< 50px error**: Excellent - precise control
- **50-100px error**: Good - general mouse control
- **100-150px error**: Acceptable - may need adjustment
- **> 150px error**: Poor - recalibrate

### Save/Load Calibration

After successful calibration:

**Save**: Click "Save" button (saves to `calibration_data.json`)
**Load**: Click "Load" button to restore previous calibration

This allows you to skip calibration next time if:
- Same lighting conditions
- Same seating position
- Same camera angle

## Tracking Modes

### Enable/Disable Tracking

- Click **"Enable Tracking"** button (or press **SPACE**)
- Your gaze now controls the cursor
- Button turns green when active
- Click again (or press **SPACE**) to pause

### How It Works

1. Camera captures your face (30 FPS)
2. MediaPipe detects iris positions
3. Kalman filter smooths the signal
4. Calibration mapping converts to screen coordinates
5. Cursor moves to predicted location

## Click Detection

Choose click mode from dropdown:

### 1. None
- No automatic clicking
- Use physical mouse/keyboard for clicks
- Recommended for initial practice

### 2. Blink
- Close both eyes briefly to click
- Default threshold: 0.21 (adjustable)
- Debounce time: 500ms (prevents double-clicks)

**Usage:**
- Natural blink = click
- Adjust threshold if too sensitive/insensitive
- Higher threshold = requires stronger blink

### 3. Dwell
- Keep cursor still for set duration
- Default: 600ms
- Visual feedback shows progress (circle fills)

**Usage:**
- Move cursor over target
- Hold still - you'll see progress indicator
- Click triggers automatically
- Good for accessibility, no physical action needed

**Settings:**
- **Dwell Time**: 300-1500ms (how long to wait)
- **Dwell Radius**: 10-50px (how much movement allowed)

### 4. Wink (L/R)
- Left wink = left click
- Right wink = right click
- Requires closing one eye while keeping other open

**Usage:**
- More control than blink (separate left/right)
- Harder to perform consistently
- May not work well with glasses

## Settings & Tuning

### Smoothing (0-100)

Controls cursor smoothness vs responsiveness.

- **Low (10-30)**: Fast response, jittery
- **Medium (30-50)**: Balanced (recommended)
- **High (50-100)**: Very smooth, sluggish

**When to adjust:**
- Cursor too jittery → Increase smoothing
- Cursor too slow → Decrease smoothing
- Default: 30

### Blink Threshold (0.10-0.40)

Eye Aspect Ratio threshold for blink detection.

- **Lower**: More sensitive, may false-trigger
- **Higher**: Requires stronger blink

**When to adjust:**
- Accidental clicks → Increase threshold
- Hard to trigger → Decrease threshold
- Default: 0.21

### Dwell Time (300-1500ms)

How long cursor must stay still.

- **Shorter**: Faster but more accidental clicks
- **Longer**: More deliberate, slower workflow

**When to adjust:**
- Too many accidental clicks → Increase
- Too slow → Decrease
- Default: 600ms

## Tips for Best Results

### Optimal Performance

1. **Lighting**: Bright, even lighting. Use desk lamp if needed
2. **Camera**: Clean lens, 720p or better
3. **Position**: Stable head position, ~60cm from screen
4. **Background**: Plain background helps (not required)

### Improving Accuracy

1. **Recalibrate** if you:
   - Move your chair
   - Change lighting
   - Adjust monitor position
   - Put on/remove glasses

2. **Adjust smoothing**: Start high, decrease gradually

3. **Multi-monitor**: May need per-monitor calibration

### Reducing Fatigue

1. **Take breaks**: Rest eyes every 20 minutes
2. **Blink regularly**: System works fine with normal blinking
3. **Use hybrid mode**: Physical mouse + eye tracking
4. **Adjust brightness**: Reduce screen brightness

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Low FPS | Close other apps, improve lighting |
| Cursor jumps | Increase smoothing, recalibrate |
| No face detected | Check camera, improve lighting |
| Poor accuracy | Recalibrate, check positioning |
| False clicks | Increase thresholds |
| Cursor drift | Recalibrate, keep head still |

## Command Line Usage

### Basic Commands

```bash
# Run with default settings
eyemouse

# Enable debug logging
eyemouse --debug

# Use different camera
eyemouse --camera 1

# Show help
eyemouse --help
```

### Demo Scripts

**Basic Demo** (landmarks only):
```bash
python scripts/demo_basic.py
```

**Calibration Demo** (full calibration flow):
```bash
python scripts/demo_calibration.py
```

### Evaluation Script

Measure accuracy:
```bash
python scripts/evaluate.py

# Custom test points
python scripts/evaluate.py --test-points 16
```

This runs full calibration + test point evaluation and outputs:
- Mean error (pixels)
- Standard deviation
- Max error
- Quality assessment

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_utils.py

# Verbose output
pytest tests/ -v
```

## Advanced Usage

### Python API

Use components programmatically:

```python
from eyemouse.capture import CameraCapture
from eyemouse.detector import FaceDetector
from eyemouse.tracker import GazeTracker

# Initialize
camera = CameraCapture()
camera.start()

detector = FaceDetector()
tracker = GazeTracker(1920, 1080)

# Process loop
while True:
    frame, timestamp = camera.read()
    detection = detector.process(frame)

    if detection:
        cursor_pos = tracker.process(detection)
        # Use cursor_pos...
```

See source code for full API documentation.

### Custom Calibration Points

Edit `calibration.py` to use custom grid:

```python
manager = CalibrationManager(
    screen_width=1920,
    screen_height=1080,
    num_points=16,  # Use 4x4 grid
    mapping_method="rbf"  # Try RBF instead of polynomial
)
```

### Custom Click Detection

Implement custom click detector in `clicker.py`:

```python
class CustomDetector:
    def detect(self, detection_result):
        # Your logic here
        return should_click
```

## Keyboard Shortcuts

- **SPACE**: Pause/Resume tracking (or confirm calibration)
- **C**: Start calibration
- **Q**: Quit application

## Next Steps

1. Practice with tracking disabled first
2. Calibrate carefully
3. Start with "None" click mode
4. Gradually enable click detection
5. Tune settings to your preference

## Getting Help

- Check [README.md](README.md) for overview
- See [INSTALL.md](INSTALL.md) for installation issues
- Report bugs on [GitHub Issues](https://github.com/your-repo/issues)
- Read [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

---

**Remember**: Eye tracking takes practice! Start slow, calibrate carefully, and adjust settings to your comfort.
