# Contributing to Eye Control Mouse

Thank you for considering contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful and inclusive. We want this to be a welcoming project for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug is already reported in [Issues](https://github.com/your-repo/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, webcam model)
   - Screenshots or error messages

### Suggesting Enhancements

1. Check if the enhancement is already suggested
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach (optional)

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/eye-control-mouse.git
   cd eye-control-mouse
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Make your changes**
   - Write clear, commented code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation

5. **Run tests**
   ```bash
   pytest tests/
   ```

6. **Run linters**
   ```bash
   black src/ tests/
   pylint src/
   mypy src/
   ```

7. **Commit your changes**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

   Commit message format:
   - `feat: Add new feature`
   - `fix: Fix bug in X`
   - `docs: Update documentation`
   - `test: Add tests for X`
   - `refactor: Refactor X`

8. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Development Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all functions/classes

Example:
```python
def process_frame(frame: np.ndarray, threshold: float = 0.5) -> Optional[Result]:
    """
    Process a single frame and detect faces.

    Args:
        frame: Input image as numpy array
        threshold: Detection confidence threshold (0-1)

    Returns:
        Detection result or None if no face detected
    """
    # Implementation
```

### Testing

- Write unit tests for new functions
- Write integration tests for new features
- Aim for >80% code coverage
- Tests should be fast and deterministic

### Documentation

Update documentation for:
- New features: Update README.md
- API changes: Update docstrings
- Configuration: Update relevant files
- Bug fixes: Update CHANGELOG.md (if exists)

## Project Structure

```
eye-control-mouse/
├── src/eyemouse/        # Main package
│   ├── app.py           # Entry point
│   ├── capture.py       # Camera capture
│   ├── detector.py      # Face/eye detection
│   ├── tracker.py       # Gaze tracking
│   ├── calibration.py   # Calibration system
│   ├── clicker.py       # Click detection
│   ├── gui.py           # GUI application
│   └── utils.py         # Utilities
├── tests/               # Test suite
├── scripts/             # Demo and evaluation scripts
├── docs/                # Documentation
└── requirements.txt     # Dependencies
```

## Areas for Contribution

### High Priority

- [ ] Improve calibration accuracy
- [ ] Reduce latency
- [ ] Better head movement compensation
- [ ] GPU acceleration support
- [ ] Multi-monitor support

### Medium Priority

- [ ] Additional click modes
- [ ] Configurable keyboard shortcuts
- [ ] Better error messages
- [ ] More comprehensive tests
- [ ] Performance profiling tools

### Good First Issues

- [ ] Add more unit tests
- [ ] Improve documentation
- [ ] Add example videos/GIFs
- [ ] Fix typos or formatting
- [ ] Add logging throughout codebase

## Performance Optimization Guidelines

When optimizing performance:

1. **Profile first**: Use `cProfile` or `line_profiler`
2. **Measure impact**: Before and after metrics
3. **Document trade-offs**: Performance vs accuracy
4. **Test thoroughly**: Ensure no regression

## Adding New Features

For substantial new features:

1. **Discuss first**: Create an issue to discuss the feature
2. **Design document**: For major features, write a design doc
3. **Break into steps**: Create tasks/milestones
4. **Backwards compatibility**: Avoid breaking existing functionality

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Create an issue
- **Chat**: Join our Discord/Slack (if available)

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Appreciated in the community!

Thank you for contributing to making eye tracking more accessible!
