"""System-wide emergency stop hotkey."""

import logging
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# pynput hotkey syntax: modifiers and special keys in <angle brackets>,
# joined with "+", e.g. "<ctrl>+<alt>+<space>" or "<ctrl>+<shift>+x"
DEFAULT_STOP_HOTKEY = "<ctrl>+<alt>+<space>"


def format_hotkey(hotkey: str) -> str:
    """Human-readable form of a pynput hotkey, e.g. "Ctrl+Alt+Space"."""
    parts = []
    for part in hotkey.split("+"):
        name = part.strip().strip("<>")
        parts.append(name.upper() if len(name) == 1 else name.capitalize())
    return "+".join(parts)


class GlobalStopHotkey(QObject):
    """
    Listen for a hotkey anywhere in the system, not only while the
    application window has focus.

    Tracking moves the real mouse cursor, so once it is running the user
    may not be able to bring the window back into focus to press Space.
    pynput's listener runs in its own thread; `triggered` is a Qt signal,
    so connected slots on GUI objects run in the GUI thread.
    """

    triggered = pyqtSignal()

    def __init__(self, hotkey: str = DEFAULT_STOP_HOTKEY, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.hotkey = hotkey
        self._listener = None

    def start(self) -> bool:
        """
        Start listening.

        Returns:
            True if the hotkey is active. False if pynput is unavailable,
            the hotkey string is invalid, or the platform does not allow
            global keyboard monitoring (e.g. a Wayland session, or missing
            Accessibility permission on macOS); the reason is logged.
        """
        if self._listener is not None:
            return True

        try:
            from pynput import keyboard

            keyboard.HotKey.parse(self.hotkey)  # raises ValueError if invalid
            self._listener = keyboard.GlobalHotKeys({self.hotkey: self._on_activate})
            self._listener.daemon = True
            self._listener.start()
        except Exception as e:
            logger.warning("Emergency stop hotkey %r unavailable: %s", self.hotkey, e)
            self._listener = None
            return False

        logger.info("Emergency stop hotkey: %s", format_hotkey(self.hotkey))
        return True

    def _on_activate(self):
        """Called in pynput's listener thread."""
        logger.info("Emergency stop hotkey pressed")
        self.triggered.emit()

    def stop(self):
        """Stop listening."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def is_active(self) -> bool:
        """Whether the hotkey is currently being listened for."""
        return self._listener is not None
