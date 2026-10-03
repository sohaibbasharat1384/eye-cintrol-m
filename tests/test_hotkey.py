"""Tests for the system-wide emergency stop hotkey."""

import threading

import pytest
from PyQt6.QtCore import QCoreApplication

from eyemouse.hotkey import DEFAULT_STOP_HOTKEY, GlobalStopHotkey, format_hotkey


@pytest.fixture(scope="module")
def qt_core_app():
    """A QCoreApplication (no display needed) to deliver queued signals."""
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


@pytest.mark.parametrize("hotkey, expected", [
    ("<ctrl>+<alt>+<space>", "Ctrl+Alt+Space"),
    ("<ctrl>+<shift>+s", "Ctrl+Shift+S"),
    ("<f9>", "F9"),
])
def test_format_hotkey(hotkey, expected):
    assert format_hotkey(hotkey) == expected


def test_default_hotkey_is_valid():
    from pynput import keyboard

    keyboard.HotKey.parse(DEFAULT_STOP_HOTKEY)  # raises if invalid


def test_invalid_hotkey_is_reported_not_raised(qt_core_app):
    hotkey = GlobalStopHotkey("<ctrl>+<not-a-key>")
    assert hotkey.start() is False
    assert not hotkey.is_active()
    hotkey.stop()  # safe when not running


def test_activation_from_listener_thread_reaches_main_thread(qt_core_app):
    """pynput calls back in its own thread; the slot must run in the Qt
    (main) thread so it can safely touch GUI objects."""
    hotkey = GlobalStopHotkey()
    calls = []
    hotkey.triggered.connect(lambda: calls.append(threading.current_thread()))

    worker = threading.Thread(target=hotkey._on_activate)
    worker.start()
    worker.join()

    assert calls == []  # queued, not run in the listener thread
    QCoreApplication.processEvents()
    assert calls == [threading.main_thread()]
