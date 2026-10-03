"""Tests for persistent settings."""

import json

from eyemouse.settings import (
    CLICK_MODES, DEFAULT_SETTINGS, SETTINGS_FILENAME,
    default_calibration_path, load_settings, save_settings,
)


def test_defaults_when_nothing_saved(tmp_path):
    assert load_settings(tmp_path) == DEFAULT_SETTINGS


def test_round_trip(tmp_path):
    settings = {
        "camera_id": 1, "click_mode": "dwell", "smoothing": 40,
        "blink_threshold": 25, "dwell_time_ms": 900, "auto_confirm": True,
    }
    assert save_settings(settings, tmp_path)
    assert load_settings(tmp_path) == settings


def test_save_creates_directory(tmp_path):
    target = tmp_path / "nested" / "eyemouse"
    assert save_settings(DEFAULT_SETTINGS, target)
    assert (target / SETTINGS_FILENAME).is_file()


def test_invalid_and_unknown_values_fall_back_to_defaults(tmp_path):
    (tmp_path / SETTINGS_FILENAME).write_text(json.dumps({
        "camera_id": -3,             # out of range
        "click_mode": "telepathy",   # unknown mode
        "smoothing": "high",         # wrong type
        "blink_threshold": 30,       # valid
        "auto_confirm": 1,           # not a bool
        "unknown_key": 123,
    }))
    loaded = load_settings(tmp_path)

    assert loaded["blink_threshold"] == 30
    for key in ("camera_id", "click_mode", "smoothing", "auto_confirm", "dwell_time_ms"):
        assert loaded[key] == DEFAULT_SETTINGS[key]
    assert "unknown_key" not in loaded


def test_corrupt_file_gives_defaults(tmp_path):
    (tmp_path / SETTINGS_FILENAME).write_text("{not json")
    assert load_settings(tmp_path) == DEFAULT_SETTINGS


def test_click_modes_match_gui_combo_order():
    """GUI combo items are None, Blink, Dwell, Wink (L/R), in this order."""
    assert CLICK_MODES == ("none", "blink", "dwell", "wink")


def test_calibration_path_is_in_config_dir(tmp_path):
    assert default_calibration_path(tmp_path).parent == tmp_path
