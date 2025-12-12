"""Tests for utility functions."""

import numpy as np
import pytest
from src.eyemouse.utils import (
    PerformanceMetrics,
    EWMAFilter,
    SimpleKalmanFilter,
    compute_eye_aspect_ratio,
    normalize_point,
    denormalize_point,
    StabilityBuffer
)


class TestPerformanceMetrics:
    """Test performance metrics tracking."""

    def test_fps_calculation(self):
        """Test FPS calculation."""
        metrics = PerformanceMetrics(window_size=10)

        # Add frames at 30 FPS (0.033s interval)
        for i in range(10):
            metrics.add_frame(i * 0.033, 0.01)

        fps = metrics.get_fps()
        assert 25 < fps < 35  # Should be around 30 FPS

    def test_latency_calculation(self):
        """Test latency calculation."""
        metrics = PerformanceMetrics(window_size=10)

        for i in range(10):
            metrics.add_frame(i * 0.033, 0.05)  # 50ms latency

        avg_latency = metrics.get_avg_latency()
        assert 45 < avg_latency < 55  # Should be around 50ms


class TestEWMAFilter:
    """Test EWMA filter."""

    def test_initialization(self):
        """Test filter initialization."""
        filter = EWMAFilter(alpha=0.3)
        assert filter.value is None

    def test_first_update(self):
        """Test first update sets value."""
        filter = EWMAFilter(alpha=0.3)
        value = np.array([1.0, 2.0])
        result = filter.update(value)

        np.testing.assert_array_equal(result, value)

    def test_smoothing(self):
        """Test smoothing behavior."""
        filter = EWMAFilter(alpha=0.5)

        # First value
        filter.update(np.array([0.0, 0.0]))

        # Second value - should be smoothed
        result = filter.update(np.array([10.0, 10.0]))

        # With alpha=0.5: 0.5*10 + 0.5*0 = 5
        assert 4.0 < result[0] < 6.0
        assert 4.0 < result[1] < 6.0

    def test_reset(self):
        """Test filter reset."""
        filter = EWMAFilter(alpha=0.3)
        filter.update(np.array([1.0, 2.0]))
        filter.reset()

        assert filter.value is None


class TestSimpleKalmanFilter:
    """Test Kalman filter."""

    def test_initialization(self):
        """Test filter initialization."""
        filter = SimpleKalmanFilter()
        assert filter.state is None

    def test_first_update(self):
        """Test first update initializes state."""
        filter = SimpleKalmanFilter()
        measurement = np.array([100.0, 200.0])
        result = filter.update(measurement)

        np.testing.assert_array_almost_equal(result, measurement)

    def test_smoothing(self):
        """Test smoothing reduces noise."""
        filter = SimpleKalmanFilter(
            process_variance=1e-3,
            measurement_variance=1e-1
        )

        # Initialize with stable point
        for _ in range(10):
            filter.update(np.array([100.0, 100.0]))

        # Add noisy measurement
        result = filter.update(np.array([150.0, 150.0]))

        # Should be smoothed, not jump to 150
        assert result[0] < 140.0
        assert result[1] < 140.0

    def test_reset(self):
        """Test filter reset."""
        filter = SimpleKalmanFilter()
        filter.update(np.array([1.0, 2.0]))
        filter.reset()

        assert filter.state is None


class TestEyeAspectRatio:
    """Test eye aspect ratio calculation."""

    def test_normal_eye(self):
        """Test EAR for normal open eye."""
        # Create synthetic eye landmarks (open eye)
        eye_landmarks = np.array([
            [0, 5],    # Left corner
            [3, 3],    # Top
            [6, 2],    # Top
            [10, 5],   # Right corner
            [6, 8],    # Bottom
            [3, 7]     # Bottom
        ], dtype=float)

        ear = compute_eye_aspect_ratio(eye_landmarks)

        # Open eye should have EAR around 0.25-0.35
        assert 0.2 < ear < 0.4

    def test_closed_eye(self):
        """Test EAR for closed eye."""
        # Create synthetic eye landmarks (closed eye - vertical distances small)
        eye_landmarks = np.array([
            [0, 5],    # Left corner
            [3, 5],    # Top (same height)
            [6, 5],    # Top
            [10, 5],   # Right corner
            [6, 5],    # Bottom (same height)
            [3, 5]     # Bottom
        ], dtype=float)

        ear = compute_eye_aspect_ratio(eye_landmarks)

        # Closed eye should have very small EAR
        assert ear < 0.15


class TestPointNormalization:
    """Test point normalization functions."""

    def test_normalize(self):
        """Test point normalization."""
        point = (320, 240)  # Center of 640x480
        normalized = normalize_point(point, 640, 480)

        assert normalized == pytest.approx((0.5, 0.5))

    def test_denormalize(self):
        """Test point denormalization."""
        point = (0.5, 0.5)
        denormalized = denormalize_point(point, 640, 480)

        assert denormalized == (320, 240)

    def test_round_trip(self):
        """Test normalize -> denormalize round trip."""
        original = (123, 456)
        normalized = normalize_point(original, 640, 480)
        denormalized = denormalize_point(normalized, 640, 480)

        assert denormalized == original


class TestStabilityBuffer:
    """Test stability buffer."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = StabilityBuffer(buffer_size=10)
        assert buffer.get_stable_value() is None

    def test_add_values(self):
        """Test adding values to buffer."""
        buffer = StabilityBuffer(buffer_size=5)

        for i in range(5):
            buffer.add(np.array([float(i), float(i)]))

        stable = buffer.get_stable_value()
        assert stable is not None
        assert stable[0] == pytest.approx(2.0)  # Mean of 0,1,2,3,4

    def test_is_stable(self):
        """Test stability detection."""
        buffer = StabilityBuffer(buffer_size=5)

        # Add stable values
        for _ in range(5):
            buffer.add(np.array([10.0, 10.0]))

        assert buffer.is_stable(threshold=0.1)

        # Add unstable value
        buffer.add(np.array([20.0, 20.0]))

        assert not buffer.is_stable(threshold=0.1)

    def test_clear(self):
        """Test buffer clear."""
        buffer = StabilityBuffer(buffer_size=5)
        buffer.add(np.array([1.0, 2.0]))
        buffer.clear()

        assert buffer.get_stable_value() is None
