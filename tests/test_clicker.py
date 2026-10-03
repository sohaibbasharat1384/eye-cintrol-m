"""Tests for click detection."""

import numpy as np
import pytest

from eyemouse.clicker import BlinkDetector

FPS = 30.0

# Six-point eye contours: EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
OPEN_EYE = np.array([[0, 5], [3, 3.5], [6, 3.5], [10, 5], [6, 6.5], [3, 6.5]], dtype=float)  # 0.30
CLOSED_EYE = np.array([[0, 5], [3, 4.75], [6, 4.75], [10, 5], [6, 5.25], [3, 5.25]], dtype=float)  # 0.05


class FakeEyes:
    """Stand-in for FaceDetectionResult exposing only the eye contours."""

    def __init__(self, left_closed=False, right_closed=False):
        self.left = CLOSED_EYE if left_closed else OPEN_EYE
        self.right = CLOSED_EYE if right_closed else OPEN_EYE

    def get_left_eye_landmarks(self):
        return self.left

    def get_right_eye_landmarks(self):
        return self.right


def run_sequence(detector, pattern, start=100.0):
    """
    Feed frames to detector.detect(); pattern is a list of
    (closed: bool, seconds) segments. Returns the times clicks fired.
    """
    clicks = []
    t = start
    for closed, seconds in pattern:
        for _ in range(int(round(seconds * FPS))):
            if detector.detect(FakeEyes(closed, closed), now=t):
                clicks.append(t)
            t += 1.0 / FPS
    return clicks


class TestBlinkDetector:
    def test_natural_blink_does_not_click(self):
        """An involuntary ~200 ms blink must not click."""
        clicks = run_sequence(BlinkDetector(), [(False, 0.5), (True, 0.2), (False, 0.5)])
        assert clicks == []

    def test_deliberate_blink_clicks_once_on_reopen(self):
        clicks = run_sequence(BlinkDetector(), [(False, 0.5), (True, 0.6), (False, 0.5)])
        assert len(clicks) == 1
        # Fires on the first open frame after the closure
        assert clicks[0] == pytest.approx(100.0 + 0.5 + 0.6, abs=1.5 / FPS)

    def test_very_long_closure_is_ignored(self):
        """Resting the eyes for several seconds is not a click."""
        clicks = run_sequence(BlinkDetector(), [(False, 0.5), (True, 3.0), (False, 0.5)])
        assert clicks == []

    def test_one_eye_closed_is_not_a_blink(self):
        detector = BlinkDetector()
        t, clicks = 100.0, []
        for left_closed in [False] * 10 + [True] * 20 + [False] * 10:
            if detector.detect(FakeEyes(left_closed=left_closed), now=t):
                clicks.append(t)
            t += 1 / FPS
        # The averaged EAR (0.30 + 0.05) / 2 = 0.175 would be below 0.21;
        # a blink requires both eyes closed
        assert clicks == []

    def test_debounce_between_clicks(self):
        detector = BlinkDetector(debounce_time=1.0)
        pattern = [(True, 0.5), (False, 0.1), (True, 0.5), (False, 0.5)]
        assert len(run_sequence(detector, pattern)) == 1

    def test_threshold_is_clipped(self):
        detector = BlinkDetector()
        detector.set_threshold(0.9)
        assert detector.ear_threshold == 0.4
