# Quick Start Guide

Get up and running with Eye Control Mouse in 5 minutes!

## 1. Install (30 seconds)

```bash
# Clone the repository
git clone https://github.com/your-repo/eye-control-mouse.git
cd eye-control-mouse

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_installation.py
```

## 2. First Run (1 minute)

```bash
python src/eyemouse/app.py
```

You should see:
- Window with your camera feed
- Green landmarks on your eyes
- Control panel on the right

## 3. Quick Calibration (2 minutes)

1. Click **"Start Calibration"** button
2. Look at each red dot that appears (9 total)
3. When you see "STABLE", press **SPACE**
4. Repeat for all 9 points
5. Done! You'll see calibration accuracy

**Tip**: Keep your head still and look directly at each point.

## 4. Enable Tracking (30 seconds)

1. Click **"Enable Tracking"** button
2. Your eyes now control the cursor!
3. Try moving your gaze around the screen

## 5. Try Click Detection (1 minute)

1. Select click mode from dropdown:
   - **Blink**: Close both eyes briefly
   - **Dwell**: Hold cursor still for 600ms
2. Test clicking on buttons or icons

## That's It! 🎉

You're now controlling your mouse with your eyes.

## Common First-Time Issues

### "No face detected"
→ Check camera permissions and improve lighting

### Cursor is jumpy
→ Increase smoothing slider (right panel)

### Can't trigger clicks
→ Adjust blink threshold slider

### Poor accuracy
→ Recalibrate in better lighting

## Next Steps

- Read [USAGE.md](USAGE.md) for detailed guide
- Try demos: `python scripts/demo_basic.py`
- Run evaluation: `python scripts/evaluate.py`
- Adjust settings to your preference

## Quick Commands

```bash
# Run application
eyemouse

# Basic demo (no mouse control)
python scripts/demo_basic.py

# Calibration demo
python scripts/demo_calibration.py

# Measure accuracy
python scripts/evaluate.py

# Run tests
pytest tests/

# Verify setup
python scripts/verify_installation.py
```

## Keyboard Shortcuts

- **SPACE**: Pause/Resume tracking
- **C**: Start calibration
- **Q**: Quit

## Tips for Best Experience

1. ✅ Good, even lighting on your face
2. ✅ Camera at eye level, 50-70cm away
3. ✅ Stable seating position
4. ✅ Calibrate carefully (don't rush)
5. ✅ Start with smoothing = 30-40
6. ✅ Practice without clicks first

## Getting Help

- **Installation**: See [INSTALL.md](INSTALL.md)
- **Usage**: See [USAGE.md](USAGE.md)
- **Issues**: Open [GitHub Issue](https://github.com/your-repo/issues)

---

**Ready to go deeper?** Check out [USAGE.md](USAGE.md) for the complete guide!
