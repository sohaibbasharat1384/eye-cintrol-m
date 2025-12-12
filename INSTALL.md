# Installation Guide

## Prerequisites

- **Python**: 3.9, 3.10, or 3.11
- **Webcam**: Built-in or USB webcam
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 20.04+)

## Installation Methods

### Method 1: Binary Installer (Easiest)

#### Windows
1. Download `EyeMouse-Setup.exe` from [Releases](https://github.com/your-repo/releases)
2. Run the installer
3. Follow the installation wizard
4. Launch from Start Menu or Desktop shortcut

#### macOS
1. Download `EyeMouse.dmg` from [Releases](https://github.com/your-repo/releases)
2. Open the DMG file
3. Drag EyeMouse to Applications folder
4. Launch from Applications or Spotlight

### Method 2: Install from PyPI

```bash
pip install eyemouse
```

Then run:
```bash
eyemouse
```

### Method 3: Install from Source (Developers)

#### Step 1: Clone the repository
```bash
git clone https://github.com/your-repo/eye-control-mouse.git
cd eye-control-mouse
```

#### Step 2: Create virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Install in development mode
```bash
pip install -e .
```

#### Step 5: Run the application
```bash
# Option 1: Using the installed command
eyemouse

# Option 2: Run directly
python src/eyemouse/app.py

# Option 3: As module
python -m eyemouse
```

## System-Specific Setup

### Windows

No additional setup required. Windows Defender may show a warning for the executable - this is normal for unsigned applications.

**Camera Permissions**: Windows 10/11 may require camera permissions:
1. Settings → Privacy & Security → Camera
2. Enable "Let apps access your camera"
3. Enable for Python or EyeMouse

### macOS

**Camera Permissions**: macOS will prompt for camera access on first run.

If you need to manually enable:
1. System Preferences → Security & Privacy → Camera
2. Check the box for Terminal (if running from terminal) or EyeMouse

**Gatekeeper**: For downloaded apps, you may need to:
1. Right-click the app → Open
2. Click "Open" in the security dialog

### Linux

**Camera Access**: Ensure you have access to video devices:
```bash
sudo usermod -a -G video $USER
```
Log out and back in for changes to take effect.

**Dependencies**: Install system packages:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libopencv-dev python3-opencv
sudo apt-get install python3-pyqt6

# Fedora
sudo dnf install python3-devel python3-pip
sudo dnf install opencv-python3
sudo dnf install python3-qt6
```

## Troubleshooting

### Camera not detected

1. Check if camera is working in other applications
2. Try different camera ID:
   ```bash
   eyemouse --camera 1
   ```
3. Check camera permissions (see above)

### ImportError: No module named 'cv2'

```bash
pip install opencv-python
```

### ImportError: No module named 'mediapipe'

```bash
pip install mediapipe
```

### PyQt6 errors on Linux

```bash
sudo apt-get install python3-pyqt6 python3-pyqt6.qtmultimedia
pip install --upgrade PyQt6
```

### Low FPS / Performance Issues

1. Close other applications using the camera
2. Improve lighting in your environment
3. Lower camera resolution (edit in code if needed)
4. Check CPU usage

### "Permission denied" on macOS

```bash
chmod +x /Applications/EyeMouse.app/Contents/MacOS/EyeMouse
```

## Verification

After installation, verify it works:

```bash
# Run basic demo
python scripts/demo_basic.py

# Run tests
pytest tests/

# Check version
python -c "import src.eyemouse; print(src.eyemouse.__version__)"
```

## Uninstallation

### PyPI install
```bash
pip uninstall eyemouse
```

### Source install
```bash
cd eye-control-mouse
pip uninstall eyemouse
```

### Binary install
- **Windows**: Control Panel → Programs → Uninstall
- **macOS**: Delete from Applications folder
- **Linux**: Use package manager or delete binary

## Next Steps

After installation:
1. Read [README.md](README.md) for usage instructions
2. Run calibration: Launch app and click "Start Calibration"
3. Adjust settings for your environment
4. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter issues
