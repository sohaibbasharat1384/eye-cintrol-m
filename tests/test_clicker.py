"""Tests for click detection."""

import numpy as np
import pytest

from eyemouse.clicker import BlinkDetector, ClickManager, ClickMode, DwellDetector

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


class TestDwellDetector:
    @staticmethod
    def run(detector, positions_per_frame, start=100.0):
        """Feed one position per frame at 30 FPS; return click frame indices."""
        clicks = []
        for i, pos in enumerate(positions_per_frame):
            clicked, _ = detector.update(pos, now=start + i / FPS)
            if clicked:
                clicks.append(i)
        return clicks

    def test_dwell_clicks_after_dwell_time(self):
        detector = DwellDetector(dwell_time=0.6, dwell_radius=30)
        clicks = self.run(detector, [(500, 500)] * 30)
        # 0.6 s at 30 FPS (frame 18, or 19 due to floating point rounding)
        assert len(clicks) == 1 and clicks[0] in (18, 19)

    def test_staying_on_the_spot_does_not_click_again(self):
        """Previously this clicked every dwell_time + debounce (~1.1 s)."""
        detector = DwellDetector(dwell_time=0.6, dwell_radius=30)
        clicks = self.run(detector, [(500, 500)] * int(5 * FPS))
        assert len(clicks) == 1

    def test_rearms_after_leaving_the_spot(self):
        detector = DwellDetector(dwell_time=0.6, dwell_radius=30)
        frames = [(500, 500)] * 30 + [(700, 500)] * 30 + [(500, 500)] * 30
        assert len(self.run(detector, frames)) == 3

    def test_small_jitter_inside_radius_still_dwells(self):
        rng = np.random.default_rng(0)
        detector = DwellDetector(dwell_time=0.6, dwell_radius=30)
        frames = [tuple((np.array([500, 500]) + rng.normal(0, 5, 2)).astype(int)) for _ in range(30)]
        assert len(self.run(detector, frames)) == 1

    def test_reset_rearms(self):
        detector = DwellDetector(dwell_time=0.6, dwell_radius=30)
        self.run(detector, [(500, 500)] * 30)
        detector.reset()
        assert len(self.run(detector, [(500, 500)] * 30, start=200.0)) == 1


class TestClickManagerWink:
    class Mouse:
        def __init__(self):
            self.clicks = []

        def click(self, button='left'):
            self.clicks.append(button)

    # WinkDetector needs the open eye above ear_threshold + ear_diff_threshold
    # (0.21 + 0.1), so use a wide-open eye (EAR 0.40) for the open side
    WIDE_EYE = np.array([[0, 5], [3, 3], [6, 3], [10, 5], [6, 7], [3, 7]], dtype=float)

    def wink(self, left_closed):
        eyes = FakeEyes()
        eyes.left = CLOSED_EYE if left_closed else self.WIDE_EYE
        eyes.right = self.WIDE_EYE if left_closed else CLOSED_EYE
        return eyes

    @pytest.mark.parametrize("left_closed, expected", [(True, 'left'), (False, 'right')])
    def test_wink_clicks_matching_button(self, left_closed, expected):
        manager = ClickManager(self.Mouse())
        manager.set_mode(ClickMode.WINK)
        frames = [FakeEyes()] * 3 + [self.wink(left_closed)] * 5 + [FakeEyes()] * 3
        for frame in frames:
            manager.update(frame, (500, 500))
        assert manager.mouse.clicks == [expected]
