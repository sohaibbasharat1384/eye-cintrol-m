# Contributing to Eye Control Mouse

Thank you for considering contributing! Be respectful and inclusive; we want this to be a welcoming project for everyone.

## Reporting Bugs

Check the [issues](https://github.com/sohaibbasharat1384/eye-control-mouse/issues) first. A good report includes:

- a clear title, steps to reproduce, expected vs. actual behavior
- OS, Python version, webcam model, display scaling
- the console output of `eyemouse --debug`

## Suggesting Enhancements

Open an issue describing the feature, its use cases, and optionally an implementation idea. For substantial changes, discuss them in an issue before writing code.

## Development Setup

```bash
git clone https://github.com/sohaibbasharat1384/eye-control-mouse.git
cd eye-control-mouse
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

## Workflow

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make focused changes; add or update tests and docs.
3. Run the checks:
   ```bash
   pytest
   black src/ tests/ scripts/
   pylint src/
   mypy src/
   ```
4. Commit with a descriptive message, e.g. `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`, `refactor: ...`. Explain *why* in the body.
5. Push and open a pull request.

## Code Style

- PEP 8, line length 100 (black is configured in `pyproject.toml`)
- Type hints and docstrings (Args/Returns) on public functions and classes
- Use the `logging` module (`logger = logging.getLogger(__name__)`), not `print`, in the package
- Keep the GUI thread responsive: per-frame work belongs in `ProcessingThread`, while user-triggered state changes (such as confirming calibration points) happen in the GUI thread

## Project Structure

```
eye-control-mouse/
├── src/eyemouse/
│   ├── __main__.py      # python -m eyemouse
│   ├── app.py           # entry point, command line options, logging setup
│   ├── gui.py           # main window, processing thread, calibration overlay
│   ├── capture.py       # threaded camera capture
│   ├── detector.py      # MediaPipe Face Landmarker wrapper, model lookup/download
│   ├── tracker.py       # GazeTracker (smoothing) and GazeMapper (affine/polynomial/RBF)
│   ├── calibration.py   # calibration workflow, save/load, evaluation
│   ├── clicker.py       # mouse control, blink/wink/dwell detection
│   └── utils.py         # filters, EAR, stability buffer, metrics
├── tests/               # pytest suite
├── scripts/             # demos, evaluation, installation check
├── build_installer.py   # PyInstaller build script
├── pyproject.toml       # package metadata and tool configuration
├── requirements.txt     # pinned runtime dependencies
└── requirements-dev.txt # pinned runtime + development dependencies
```

## Testing

```bash
pytest                          # all tests, with coverage report (htmlcov/)
pytest tests/test_utils.py -v   # one file
pytest -o addopts=""            # without coverage options
```

- Unit tests must be fast and deterministic (seed any randomness).
- The tests need no camera or display. They import `eyemouse.detector`, so MediaPipe must be installed.

### Manual Test Checklist

Run through this before a release, ideally on Windows, macOS and Linux.

**Install and start**
- [ ] Fresh virtual environment: `pip install -e .` succeeds; `python scripts/verify_installation.py` passes
- [ ] `eyemouse`, `python -m eyemouse` and `eyemouse --camera N` start; `--debug` shows debug logs
- [ ] First start downloads the face model; later starts work offline

**Camera and detection**
- [ ] Preview shows eye contours, iris centers and gaze point
- [ ] Switching cameras works; selecting a missing camera shows an error and returns to the previous one
- [ ] "Face: Not detected" when leaving the frame; detection recovers when you return

**Calibration**
- [ ] Overlay shows each target at the right place, also with display scaling above 100%
- [ ] Ring turns green when stable; `Space` confirms; "Not stable yet" when pressed too early
- [ ] Auto-confirm completes all 9 points without key presses
- [ ] `Esc` cancels and keeps a previous calibration
- [ ] Completion shows the error; Save/Load round-trips `calibration_data.json`

**Tracking and clicks**
- [ ] Cursor follows the gaze across the screen corners and center; FPS and latency stay reasonable
- [ ] Smoothing slider: higher values make the cursor visibly steadier and slower, live
- [ ] Blink, Dwell and Wink modes click as described in [USAGE.md](USAGE.md); `None` never clicks
- [ ] Dwell mode: the progress ring follows the dwell point, fills up, and the click reaches the window under it
- [ ] `Space` stops tracking; `Q` quits cleanly (camera released)

## Building a Standalone Executable

```bash
pip install -r requirements-dev.txt
python build_installer.py
```

The script runs PyInstaller for the current platform and writes the result to `dist/`. The model file is not bundled; the executable downloads it into the user cache directory on first start (or uses `EYEMOUSE_FACE_MODEL`).

## Areas for Contribution

- Better gaze features and calibration accuracy
- Head-movement compensation
- Multi-monitor support
- Configurable shortcuts and persistent settings
- More tests (GUI logic, click detectors)

Thank you for helping make eye tracking more accessible!
