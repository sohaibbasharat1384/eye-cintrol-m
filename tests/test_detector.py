"""Tests for the gaze feature computed from face landmarks."""

from types import SimpleNamespace

import numpy as np
import pytest

from eyemouse.detector import FaceDetectionResult, FaceDetector

FRAME_W, FRAME_H = 640, 480


def make_result(iris_offset=(0.0, 0.0), angle_deg=0.0, shift=(0.0, 0.0), scale=1.0):
    """
    Build a FaceDetectionResult from synthetic landmarks (pixel layout of a
    face looking straight ahead), optionally moving both irises within their
    eyes and transforming the whole face (roll, shift, scale).
    """
    points = {
        # left eye: corners 33 (outer) and 133 (inner), iris 468
        33: (250.0, 200.0), 133: (290.0, 200.0), 468: (270.0, 200.0),
        # right eye: corners 362 (inner) and 263 (outer), iris 473
        362: (350.0, 200.0), 263: (390.0, 200.0), 473: (370.0, 200.0),
    }
    for idx in (468, 473):
        x, y = points[idx]
        points[idx] = (x + iris_offset[0], y + iris_offset[1])

    center = np.array([320.0, 240.0])
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for idx, p in points.items():
        q = center + scale * rot @ (np.array(p) - center) + np.array(shift)
        landmarks[idx] = SimpleNamespace(x=q[0] / FRAME_W, y=q[1] / FRAME_H)
    return FaceDetectionResult(landmarks, FRAME_W, FRAME_H)


class TestGazeFeature:
    def test_centered_iris(self):
        """An iris midway between the corners is (0.5, 0)."""
        np.testing.assert_allclose(make_result().get_gaze_feature(), [0.5, 0.0], atol=1e-6)

    def test_iris_offset_direction_and_scale(self):
        """Feature moves by offset / eye width, positive right and down."""
        feature = make_result(iris_offset=(4.0, 2.0)).get_gaze_feature()
        np.testing.assert_allclose(feature, [0.5 + 4 / 40, 2 / 40], atol=1e-6)

    @pytest.mark.parametrize("transform", [
        {"angle_deg": 15.0},
        {"angle_deg": -25.0},
        {"shift": (60.0, -30.0)},
        {"scale": 1.4},
        {"angle_deg": 10.0, "shift": (-20.0, 15.0), "scale": 0.8},
    ])
    def test_invariant_to_head_roll_shift_and_scale(self, transform):
        """Moving or tilting the whole face does not change the feature."""
        reference = make_result(iris_offset=(3.0, -1.5)).get_gaze_feature()
        moved = make_result(iris_offset=(3.0, -1.5), **transform).get_gaze_feature()
        np.testing.assert_allclose(moved, reference, atol=1e-5)

    def test_degenerate_eyes_return_neutral_value(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        result = FaceDetectionResult(landmarks, FRAME_W, FRAME_H)
        np.testing.assert_allclose(result.get_gaze_feature(), [0.5, 0.0])

    def test_corner_constants_match_eye_contours(self):
        """The corner pairs are the outermost points of each eye contour."""
        assert set(FaceDetector.LEFT_EYE_CORNERS) <= set(FaceDetector.LEFT_EYE_INDICES)
        assert set(FaceDetector.RIGHT_EYE_CORNERS) <= set(FaceDetector.RIGHT_EYE_INDICES)
