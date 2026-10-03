"""Tests for per-frame cursor logic in the processing thread."""

import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication

from eyemouse.clicker import ClickManager
from eyemouse.gui import ProcessingThread

OPEN_EYE = np.array([[0, 5], [3, 3.5], [6, 3.5], [10, 5], [6, 6.5], [3, 6.5]], dtype=float)
CLOSED_EYE = np.array([[0, 5], [3, 4.75], [6, 4.75], [10, 5], [6, 5.25], [3, 5.25]], dtype=float)


class FakeEyes:
    def __init__(self, left_closed=False, right_closed=False):
        self.left = CLOSED_EYE if left_closed else OPEN_EYE
        self.right = CLOSED_EYE if right_closed else OPEN_EYE

    def get_left_eye_landmarks(self):
        return self.left

    def get_right_eye_landmarks(self):
        return self.right


class FakeTracker:
    """Returns a new position on every call, like a moving gaze."""

    def __init__(self):
        self.calls = 0

    def process(self, detection_result, dt=None):
        self.calls += 1
        return (100 * self.calls, 50 * self.calls)


class FakeMouse:
    def __init__(self):
        self.moves = []

    def move_to(self, x, y):
        self.moves.append((x, y))


@pytest.fixture
def thread():
    QCoreApplication.instance() or QCoreApplication([])
    click_manager = ClickManager(FakeMouse())
    return ProcessingThread(camera=None, detector=None, tracker=FakeTracker(),
                            click_manager=click_manager)


def test_cursor_follows_gaze_with_open_eyes(thread):
    assert thread._update_cursor(FakeEyes(), 1 / 30) == (100, 50)
    assert thread._update_cursor(FakeEyes(), 1 / 30) == (200, 100)
    assert thread.click_manager.mouse.moves == [(100, 50), (200, 100)]


@pytest.mark.parametrize("closed", [(True, True), (True, False), (False, True)])
def test_cursor_is_held_while_an_eye_is_closed(thread, closed):
    thread._update_cursor(FakeEyes(), 1 / 30)                     # (100, 50)
    for _ in range(5):
        assert thread._update_cursor(FakeEyes(*closed), 1 / 30) == (100, 50)

    # The tracker was not fed the unreliable closed-eye frames
    assert thread.tracker.calls == 1
    assert thread.click_manager.mouse.moves == [(100, 50)]

    # Tracking resumes when the eyes open again
    assert thread._update_cursor(FakeEyes(), 1 / 30) == (200, 100)


def test_closed_eyes_before_any_position_still_track(thread):
    """Without a previous position there is nothing to hold."""
    assert thread._update_cursor(FakeEyes(True, True), 1 / 30) == (100, 50)
