"""Persistent user settings and data file locations."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SETTINGS_FILENAME = "settings.json"
CALIBRATION_FILENAME = "calibration.json"

# Values used when nothing is saved yet (or a saved value is invalid)
DEFAULT_SETTINGS: Dict[str, Any] = {
    "camera_id": 0,
    "click_mode": "none",        # none | blink | dwell | wink
    "smoothing": 75,             # 1-100, higher = smoother
    "blink_threshold": 21,       # slider units, EAR * 100 (10-40)
    "dwell_time_ms": 600,        # 300-1500
    "auto_confirm": False,
}

# Allowed range for numeric settings: (min, max)
_RANGES = {
    "camera_id": (0, 99),
    "smoothing": (1, 100),
    "blink_threshold": (10, 40),
    "dwell_time_ms": (300, 1500),
}
CLICK_MODES = ("none", "blink", "dwell", "wink")


def user_config_dir() -> Path:
    """Per-user directory for settings and calibration data."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "eyemouse"


def default_calibration_path(config_dir: Optional[Path] = None) -> Path:
    """Where the calibration is saved and loaded from."""
    return (config_dir or user_config_dir()) / CALIBRATION_FILENAME


def _validated(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Merge with defaults, dropping unknown keys and invalid values."""
    result = dict(DEFAULT_SETTINGS)
    for key, default in DEFAULT_SETTINGS.items():
        if key not in settings:
            continue
        value = settings[key]
        if isinstance(default, bool):
            if isinstance(value, bool):
                result[key] = value
        elif key == "click_mode":
            if value in CLICK_MODES:
                result[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            low, high = _RANGES[key]
            if low <= value <= high:
                result[key] = value
        if result[key] != value:
            logger.warning("Ignoring invalid setting %s=%r", key, value)
    return result


def load_settings(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load settings, falling back to defaults for anything missing, invalid
    or unreadable. Never raises.
    """
    path = (config_dir or user_config_dir()) / SETTINGS_FILENAME
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("settings file does not contain an object")
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)
    except Exception as e:
        logger.warning("Could not read settings from %s (%s); using defaults", path, e)
        return dict(DEFAULT_SETTINGS)
    return _validated(data)


def save_settings(settings: Dict[str, Any], config_dir: Optional[Path] = None) -> bool:
    """Save settings atomically (temp file + rename). Returns success."""
    directory = config_dir or user_config_dir()
    path = directory / SETTINGS_FILENAME
    tmp_path = path.with_suffix(".tmp")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_validated(settings), f, indent=2)
        tmp_path.replace(path)
        return True
    except Exception as e:
        logger.warning("Could not save settings to %s: %s", path, e)
        return False
