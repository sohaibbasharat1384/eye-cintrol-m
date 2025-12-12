# Testing & Validation Checklist

Complete checklist for testing the Eye Control Mouse system before deployment.

## Automated Tests

### Unit Tests

```bash
pytest tests/ -v
```

- [ ] `test_utils.py`: Performance metrics, filters, EAR, normalization
- [ ] `test_tracker.py`: Gaze mapping (affine, polynomial, RBF)
- [ ] `test_calibration.py`: Calibration flow, evaluation

**Expected**: All tests pass, >80% coverage

### Integration Tests

- [ ] Realistic calibration scenario (9-point with noise)
- [ ] Mapping accuracy on test data
- [ ] End-to-end gaze tracking pipeline

## Manual Testing

### 1. Installation

- [ ] Fresh install from requirements.txt works
- [ ] Verify installation script passes
- [ ] Command `eyemouse` launches application

**Test on**: Windows, macOS, Linux (if available)

### 2. Camera Access

- [ ] Camera opens successfully
- [ ] Video preview displays correctly
- [ ] Camera permissions requested (macOS/Windows 10+)
- [ ] Can switch between multiple cameras
- [ ] Graceful error if no camera

### 3. Face Detection

**Setup**: Good lighting, face centered

- [ ] Face detected within 2 seconds
- [ ] Green bounding box drawn
- [ ] Eye landmarks visible
- [ ] Iris centers tracked (cyan dots)
- [ ] Confidence > 0.8

**Edge cases**:
- [ ] Works with glasses
- [ ] Works with different ethnicities/face shapes
- [ ] Handles partial occlusion (hand near face)
- [ ] Recovers from temporary face loss

### 4. Calibration System

**9-Point Standard Calibration**:

- [ ] Calibration button starts process
- [ ] Target points displayed clearly
- [ ] Stability indicator works ("STABLE" message)
- [ ] Space key confirms points
- [ ] Progress bar updates (1/9, 2/9, etc.)
- [ ] All 9 points can be completed
- [ ] Calibration error displayed
- [ ] Error < 100px with good setup

**Calibration Quality**:
- [ ] Error < 50px: Excellent
- [ ] Error 50-100px: Good
- [ ] Error 100-150px: Acceptable
- [ ] Error > 150px: Suggests recalibration

**Save/Load**:
- [ ] Save creates `calibration_data.json`
- [ ] Load restores calibration successfully
- [ ] Loaded calibration has similar accuracy

### 5. Gaze Tracking

**Basic Tracking**:
- [ ] Enable Tracking button works
- [ ] Cursor follows gaze direction
- [ ] Latency < 80ms (feels responsive)
- [ ] FPS > 20 (check metrics display)

**Smoothing**:
- [ ] Low smoothing: Fast, jittery
- [ ] High smoothing: Slow, smooth
- [ ] Slider updates in real-time

**Accuracy**:
- [ ] Can look at screen corners
- [ ] Can look at center
- [ ] Cursor stays within ~100px of gaze point

**Edge Cases**:
- [ ] Handles moderate head movement
- [ ] Doesn't drift over 30 seconds
- [ ] Recovers from face detection loss

### 6. Click Detection

**Blink Mode**:
- [ ] Natural blink triggers click
- [ ] No false positives (normal blinking)
- [ ] Threshold slider adjusts sensitivity
- [ ] Debounce prevents double-clicks

**Dwell Mode**:
- [ ] Holding still triggers click
- [ ] Visual progress indicator shows
- [ ] Time slider adjusts duration (300-1500ms)
- [ ] Moving cursor cancels dwell

**Wink Mode**:
- [ ] Left wink triggers left click
- [ ] Right wink triggers right click
- [ ] Works reliably without glasses
- [ ] (May be difficult with glasses)

**Click Accuracy**:
- [ ] Can click large buttons reliably
- [ ] Can click medium targets (50x50px)
- [ ] Minimal false positives

### 7. Performance

**Hardware Requirements**:
- [ ] Works on typical laptop (i5/i7)
- [ ] CPU usage < 40%
- [ ] Memory usage < 200MB
- [ ] No memory leaks over 5 minutes

**Frame Rate**:
- [ ] FPS displayed in UI
- [ ] FPS > 20 with face detected
- [ ] FPS stable over time

**Latency**:
- [ ] Latency displayed in UI
- [ ] Latency < 80ms average
- [ ] Feels responsive (subjective)

### 8. GUI

**Main Window**:
- [ ] Video preview scales properly
- [ ] Control panel responsive
- [ ] Metrics update in real-time
- [ ] Window can be resized

**Controls**:
- [ ] All buttons work
- [ ] All sliders work
- [ ] Dropdown menus work
- [ ] Progress bars update

**Keyboard Shortcuts**:
- [ ] Space: Pause/resume
- [ ] C: Calibrate
- [ ] Q: Quit

**Visual Feedback**:
- [ ] Status messages clear
- [ ] Error messages helpful
- [ ] Calibration progress visible
- [ ] Dwell progress shows circle

### 9. Error Handling

**Camera Errors**:
- [ ] Graceful error if camera unavailable
- [ ] Clear message to user
- [ ] Doesn't crash

**Calibration Errors**:
- [ ] Handles insufficient samples
- [ ] Handles unstable gaze
- [ ] Can restart calibration

**Tracking Errors**:
- [ ] Handles no face detected
- [ ] Handles intermittent detection
- [ ] Cursor doesn't jump wildly

### 10. Cross-Platform

**Windows**:
- [ ] Camera access works
- [ ] Mouse control works
- [ ] GUI renders correctly
- [ ] No DLL errors

**macOS**:
- [ ] Camera permissions requested
- [ ] Mouse control works
- [ ] App bundle structure correct
- [ ] No code signing issues (expected)

**Linux**:
- [ ] Camera `/dev/video0` accessible
- [ ] X11/Wayland mouse control works
- [ ] Dependencies install cleanly
- [ ] PyQt6 renders correctly

### 11. Documentation

**README**:
- [ ] Clear overview
- [ ] Quick start works
- [ ] Installation steps correct
- [ ] Screenshots/GIFs (if added)

**INSTALL.md**:
- [ ] All platforms covered
- [ ] Dependencies listed
- [ ] Troubleshooting helpful

**USAGE.md**:
- [ ] Calibration guide clear
- [ ] Settings explained
- [ ] Tips actionable

**Code Documentation**:
- [ ] All functions have docstrings
- [ ] Type hints present
- [ ] Comments explain complex logic

### 12. Demo Scripts

**demo_basic.py**:
- [ ] Shows camera feed
- [ ] Shows landmarks
- [ ] Q to quit works

**demo_calibration.py**:
- [ ] Full calibration flow
- [ ] Virtual screen visualization
- [ ] Cursor tracking visible

**evaluate.py**:
- [ ] Runs calibration
- [ ] Collects test samples
- [ ] Reports metrics correctly

### 13. Edge Cases & Stress Tests

**Long Session**:
- [ ] Run for 30 minutes
- [ ] No memory leaks
- [ ] FPS stays stable
- [ ] No crashes

**Poor Conditions**:
- [ ] Low light: Degrades gracefully
- [ ] High light/backlit: Still works
- [ ] Multiple faces: Uses first/closest

**Unusual Inputs**:
- [ ] No camera: Shows error
- [ ] Camera disconnected mid-session: Handles
- [ ] Rapid head movement: Doesn't crash

## Acceptance Criteria

System is **production-ready** if:

- ✅ All automated tests pass
- ✅ Calibration error < 100px in good conditions
- ✅ FPS > 20, latency < 80ms
- ✅ Works on Windows, macOS, Linux
- ✅ No critical bugs
- ✅ Documentation complete

## Performance Benchmarks

Run evaluation script and record:

```bash
python scripts/evaluate.py
```

**Target Metrics**:
- Mean error: < 80px
- Std deviation: < 40px
- Max error: < 150px
- Quality: "GOOD" or better

## Regression Testing

Before each release:

1. Run full test suite
2. Manual test on each platform
3. Run evaluation script
4. Check no performance regression
5. Verify documentation up-to-date

## Known Issues / Limitations

Document any known issues:

- Multi-monitor support limited
- Precision not suitable for pixel-perfect tasks
- Best results with stable head position
- May need recalibration after moving

## Sign-Off

Testing completed by: _______________

Date: _______________

Platform tested: _______________

All critical tests passed: [ ] Yes [ ] No

Ready for release: [ ] Yes [ ] No

Notes:
_______________________________________________________
_______________________________________________________
_______________________________________________________
