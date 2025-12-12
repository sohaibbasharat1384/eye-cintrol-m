# Eye Control Mouse

> **Production-ready hands-free mouse control using webcam and eye tracking. 100% local processing.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Control your computer's mouse cursor using only your eyes and a standard webcam. Perfect for accessibility, hands-free computing, or just trying something cool!

## Features

- Real-time gaze-based cursor control (20-30+ FPS)
- Multiple click modes: blink, dwell, wink
- 9-point calibration with RBF/polynomial mapping
- Kalman filtering for smooth tracking
- Works with glasses, varied lighting, head movement
- Cross-platform: Windows, macOS, Linux

## Privacy & Safety

All video processing happens locally on your device. No data is sent to any server. Webcam frames are never saved unless you explicitly enable recording.

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for 5-minute guide!

```bash
# Clone and install
git clone https://github.com/your-repo/eye-control-mouse.git
cd eye-control-mouse
pip install -r requirements.txt

# Run
python src/eyemouse/app.py

# Or install and run
pip install -e .
eyemouse
```

## Installation

### Prerequisites
- Python 3.9-3.11
- Webcam
- Windows/macOS/Linux

### From Source
```bash
git clone <repo-url>
cd eye-control-mouse
pip install -r requirements.txt
python src/app.py
```

### Binary Installers
- Windows: Download `EyeMouse-Setup.exe` from releases
- macOS: Download `EyeMouse.app` from releases

## Usage

1. **Launch**: Run the application
2. **Calibrate**: Click "Calibrate" and follow the 9-point calibration
3. **Enable**: Click "Enable Tracking" to start controlling the mouse
4. **Click Mode**: Choose blink/dwell/wink in settings

### Keyboard Shortcuts
- `Space`: Pause/Resume tracking
- `C`: Recalibrate
- `Q`: Quit

## Configuration

Adjust in the GUI:
- **Smoothing**: Lower = faster response, Higher = more stable
- **Click Threshold**: Blink/wink sensitivity
- **Dwell Time**: Duration to trigger dwell click (default 600ms)
- **Dwell Radius**: Pixel radius for dwell detection (default 30px)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low FPS | Close other apps, ensure good lighting |
| Cursor jumpy | Increase smoothing, recalibrate |
| No face detected | Check webcam, improve lighting, center face |
| False clicks | Increase blink threshold, use dwell mode |
| Cursor drift | Recalibrate, keep head still during calibration |

## Performance Metrics

Typical performance on modern laptop (CPU-only):
- Latency: 40-70ms end-to-end
- FPS: 25-35
- Accuracy: ~30-50px average error after calibration

## Architecture

```
Camera Thread → Frame Queue → Processing Thread → Gaze Queue → Main Thread
                                     ↓
                            MediaPipe FaceMesh/Iris
                                     ↓
                            Kalman Filter + Mapping
                                     ↓
                            Click Detection + PyAutoGUI
```

## Development

```bash
# Run tests
pytest tests/

# Run with debug logging
python src/app.py --debug

# Evaluate accuracy
python scripts/evaluate.py
```

## Documentation

Comprehensive guides available:

- **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
- **[INSTALL.md](INSTALL.md)** - Detailed installation for all platforms
- **[USAGE.md](USAGE.md)** - Complete user manual with tips
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Developer guide and code style
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical architecture overview
- **[TESTING_CHECKLIST.md](TESTING_CHECKLIST.md)** - QA procedures
- **[IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)** - Full implementation analysis
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current project status

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines:
1. Fork the repo
2. Create feature branch
3. Add tests
4. Submit PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, testing, and development setup.

## License

MIT License - see [LICENSE](LICENSE) file

## Citation

If you use this in research, please cite:
```
@software{eyemouse2025,
  title={Eye Control Mouse: Webcam-based Gaze Tracking},
  year={2025}
}
```

## Acknowledgments

- MediaPipe by Google for iris tracking
- OpenCV community
- PyQt6 project
