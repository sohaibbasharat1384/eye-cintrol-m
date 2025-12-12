# Eye Control Mouse - Complete File Index

Quick reference guide to all files in this project.

## 📖 Start Here

| File | Purpose | Audience |
|------|---------|----------|
| **[README.md](README.md)** | Main overview and quick start | Everyone |
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute tutorial | New users |
| **[PROJECT_STATUS.md](PROJECT_STATUS.md)** | Current status and metrics | Everyone |

## 🚀 For Users

| File | Purpose | When to Read |
|------|---------|--------------|
| [INSTALL.md](INSTALL.md) | Installation instructions | First-time setup |
| [USAGE.md](USAGE.md) | Complete user manual | Learning the system |
| [QUICKSTART.md](QUICKSTART.md) | Quick 5-minute guide | Getting started fast |

## 👨‍💻 For Developers

| File | Purpose | When to Read |
|------|---------|--------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Developer guide | Before contributing |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Technical architecture | Understanding design |
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | Detailed analysis | Deep dive |
| [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) | QA procedures | Before release |

## 📦 Source Code

### Main Package (`src/eyemouse/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~10 | Package initialization |
| `app.py` | ~50 | Entry point and CLI |
| `capture.py` | ~315 | Threaded camera capture |
| `detector.py` | ~287 | MediaPipe face/iris detection |
| `tracker.py` | ~338 | Gaze tracking and mapping |
| `calibration.py` | ~298 | 9-point calibration system |
| `clicker.py` | ~367 | Click detection (blink/dwell/wink) |
| `gui.py` | ~687 | PyQt6 GUI application |
| `utils.py` | ~294 | Filters, metrics, math helpers |

**Total**: ~2,600 lines

### Component Descriptions

#### `capture.py`
- **Purpose**: Camera capture with threading
- **Key Classes**: `CameraCapture`
- **Features**: Non-blocking capture, queue management, mirror flip
- **Thread-safe**: Yes

#### `detector.py`
- **Purpose**: MediaPipe wrapper for face/eye detection
- **Key Classes**: `FaceDetector`, `FaceDetectionResult`
- **Features**: 468 face landmarks + 10 iris landmarks per eye
- **Performance**: 30+ FPS on CPU

#### `tracker.py`
- **Purpose**: Gaze tracking and screen mapping
- **Key Classes**: `GazeTracker`, `GazeMapper`
- **Features**: Kalman/EWMA smoothing, polynomial/RBF/affine mapping
- **Algorithms**: 3 mapping methods

#### `calibration.py`
- **Purpose**: Calibration workflow and management
- **Key Classes**: `CalibrationManager`, `CalibrationEvaluator`
- **Features**: N-point grid, stability detection, save/load
- **Default**: 9-point grid

#### `clicker.py`
- **Purpose**: Click detection and mouse control
- **Key Classes**: `MouseController`, `ClickManager`, `BlinkDetector`, `DwellDetector`, `WinkDetector`
- **Features**: 3 click modes with configurable thresholds
- **Cross-platform**: Via PyAutoGUI

#### `gui.py`
- **Purpose**: Main GUI application
- **Key Classes**: `EyeMouseGUI`, `ProcessingThread`
- **Features**: Real-time preview, metrics, calibration UI
- **Framework**: PyQt6

#### `utils.py`
- **Purpose**: Shared utilities
- **Key Classes**: `PerformanceMetrics`, `EWMAFilter`, `SimpleKalmanFilter`, `StabilityBuffer`
- **Features**: FPS tracking, smoothing, math helpers

## 🧪 Tests (`tests/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~5 | Test package init |
| `test_utils.py` | ~250 | Test filters and utilities |
| `test_tracker.py` | ~200 | Test mapping algorithms |
| `test_calibration.py` | ~150 | Test calibration flow |

**Total**: ~600 lines

**Run tests**: `pytest tests/ -v --cov=src`

## 🎮 Demo Scripts (`scripts/`)

| File | Lines | Purpose |
|------|-------|---------|
| `demo_basic.py` | ~130 | Show landmarks and gaze |
| `demo_calibration.py` | ~180 | Full calibration demo |
| `evaluate.py` | ~200 | Accuracy measurement |
| `verify_installation.py` | ~100 | Setup verification |

**Total**: ~610 lines

### Demo Descriptions

#### `demo_basic.py`
- Shows camera feed with landmarks
- No mouse control
- Good for testing detection
- **Run**: `python scripts/demo_basic.py`

#### `demo_calibration.py`
- Full calibration workflow
- Virtual screen visualization
- Cursor tracking demo
- **Run**: `python scripts/demo_calibration.py`

#### `evaluate.py`
- Measures calibration accuracy
- Reports error metrics
- Quality assessment
- **Run**: `python scripts/evaluate.py`

#### `verify_installation.py`
- Checks all dependencies
- Tests camera access
- Diagnostic tool
- **Run**: `python scripts/verify_installation.py`

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Package metadata (modern) |
| `setup.py` | Setuptools config (legacy) |
| `.gitignore` | Git ignore rules |
| `LICENSE` | MIT License text |

## 🔨 Build Tools

| File | Purpose |
|------|---------|
| `build_installer.py` | PyInstaller build script |

**Usage**: `python build_installer.py`
**Output**: Executable in `dist/`

## 📚 Documentation Files

### User Documentation

| File | Lines | Purpose |
|------|-------|---------|
| [README.md](README.md) | ~200 | Main overview |
| [QUICKSTART.md](QUICKSTART.md) | ~120 | 5-minute guide |
| [INSTALL.md](INSTALL.md) | ~250 | Installation guide |
| [USAGE.md](USAGE.md) | ~500 | User manual |

### Developer Documentation

| File | Lines | Purpose |
|------|-------|---------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | ~350 | Developer guide |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | ~450 | Technical overview |
| [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) | ~400 | QA procedures |
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | ~700 | Implementation analysis |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | ~300 | Status summary |
| [INDEX.md](INDEX.md) | ~200 | This file |

**Total**: ~3,000 lines of documentation

## 📊 Project Statistics

### Code Distribution
- **Source code**: 2,600 lines (8 files)
- **Test code**: 600 lines (3 files)
- **Demo scripts**: 610 lines (4 files)
- **Documentation**: 3,000 lines (10 files)
- **Configuration**: 362 lines (5 files)
- **Total**: **6,172 lines** (30 files)

### File Types
- Python (`.py`): 16 files
- Markdown (`.md`): 10 files
- Config (`.txt`, `.toml`): 4 files

### Test Coverage
- Unit tests: 15+ tests
- Integration tests: 6+ tests
- Coverage: >80% of core code

## 🗂️ Directory Structure

```
eye-control-mouse/
├── src/eyemouse/          # Source code (2,600 lines)
├── tests/                 # Test suite (600 lines)
├── scripts/               # Demos & tools (610 lines)
├── docs/                  # (Optional documentation directory)
├── dist/                  # (Build output, not in repo)
├── *.md                   # Documentation (3,000 lines)
├── *.txt, *.toml, *.py    # Configuration (362 lines)
└── LICENSE                # MIT License
```

## 🔍 Finding What You Need

### "I want to..."

| Goal | File to Read |
|------|--------------|
| **Get started quickly** | [QUICKSTART.md](QUICKSTART.md) |
| **Install on my system** | [INSTALL.md](INSTALL.md) |
| **Learn how to use it** | [USAGE.md](USAGE.md) |
| **Understand the architecture** | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) |
| **Contribute code** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Run tests** | [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) |
| **See implementation details** | [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) |
| **Check project status** | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| **Build executables** | `build_installer.py` |
| **Test my installation** | `scripts/verify_installation.py` |
| **Try a quick demo** | `scripts/demo_basic.py` |
| **Measure accuracy** | `scripts/evaluate.py` |

### "I want to modify..."

| Component | File to Edit |
|-----------|--------------|
| **Camera capture** | `src/eyemouse/capture.py` |
| **Face detection** | `src/eyemouse/detector.py` |
| **Gaze tracking** | `src/eyemouse/tracker.py` |
| **Calibration** | `src/eyemouse/calibration.py` |
| **Click detection** | `src/eyemouse/clicker.py` |
| **GUI interface** | `src/eyemouse/gui.py` |
| **Filters/utilities** | `src/eyemouse/utils.py` |
| **Entry point** | `src/eyemouse/app.py` |

## 🎯 Quick Commands

```bash
# Installation
pip install -r requirements.txt

# Verification
python scripts/verify_installation.py

# Run application
python src/eyemouse/app.py

# Run demos
python scripts/demo_basic.py
python scripts/demo_calibration.py

# Test accuracy
python scripts/evaluate.py

# Run tests
pytest tests/ -v

# Build executable
python build_installer.py
```

## 📞 Support Resources

| Need | Resource |
|------|----------|
| Quick help | [QUICKSTART.md](QUICKSTART.md) |
| Installation issues | [INSTALL.md](INSTALL.md) |
| Usage questions | [USAGE.md](USAGE.md) |
| Bug reports | GitHub Issues |
| Feature requests | GitHub Issues |
| Code contributions | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

**Last Updated**: January 2025
**Total Files**: 30
**Total Lines**: 6,172
**Status**: ✅ Production Ready
