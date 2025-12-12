"""Utility functions and math helpers."""

import time
from typing import Tuple, List, Optional
import numpy as np
from scipy.spatial.distance import euclidean


class PerformanceMetrics:
    """Track FPS and latency metrics."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: List[float] = []
        self.latencies: List[float] = []

    def add_frame(self, timestamp: float, latency: float):
        """Add a frame measurement."""
        self.frame_times.append(timestamp)
        self.latencies.append(latency)

        # Keep only recent measurements
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            self.latencies.pop(0)

    def get_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.frame_times) < 2:
            return 0.0

        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span == 0:
            return 0.0

        return (len(self.frame_times) - 1) / time_span

    def get_avg_latency(self) -> float:
        """Get average latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies) * 1000  # Convert to ms

    def get_latency_std(self) -> float:
        """Get latency standard deviation in milliseconds."""
        if len(self.latencies) < 2:
            return 0.0
        return np.std(self.latencies) * 1000


class EWMAFilter:
    """Exponentially Weighted Moving Average filter for smoothing."""

    def __init__(self, alpha: float = 0.3):
        """
        Initialize EWMA filter.

        Args:
            alpha: Smoothing factor (0-1). Lower = more smoothing, higher = more responsive.
        """
        self.alpha = np.clip(alpha, 0.01, 1.0)
        self.value: Optional[np.ndarray] = None

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement."""
        if self.value is None:
            self.value = measurement.copy()
        else:
            self.value = self.alpha * measurement + (1 - self.alpha) * self.value
        return self.value.copy()

    def reset(self):
        """Reset filter state."""
        self.value = None


class SimpleKalmanFilter:
    """Simple 2D Kalman filter for gaze point smoothing."""

    def __init__(
        self,
        process_variance: float = 1e-3,
        measurement_variance: float = 1e-1
    ):
        """
        Initialize Kalman filter.

        Args:
            process_variance: Process noise (lower = expect less movement)
            measurement_variance: Measurement noise (lower = trust measurements more)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # State: [x, y, vx, vy]
        self.state: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.dt = 1.0 / 30.0  # Assume 30 FPS

        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise
        q = self.process_variance
        self.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q, 0],
            [0, 0, 0, q]
        ])

        # Measurement noise
        r = self.measurement_variance
        self.R = np.array([
            [r, 0],
            [0, r]
        ])

    def update(self, measurement: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Update filter with new measurement.

        Args:
            measurement: [x, y] position
            dt: Time delta since last update (optional)

        Returns:
            Filtered [x, y] position
        """
        if dt is not None and dt > 0:
            self.dt = dt
            self.F[0, 2] = dt
            self.F[1, 3] = dt

        measurement = np.asarray(measurement, dtype=float)

        if self.state is None:
            # Initialize state
            self.state = np.array([measurement[0], measurement[1], 0, 0])
            self.covariance = np.eye(4) * 1000
            return measurement

        # Predict
        predicted_state = self.F @ self.state
        predicted_covariance = self.F @ self.covariance @ self.F.T + self.Q

        # Update
        innovation = measurement - (self.H @ predicted_state)
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)

        self.state = predicted_state + kalman_gain @ innovation
        self.covariance = (np.eye(4) - kalman_gain @ self.H) @ predicted_covariance

        return self.state[:2].copy()

    def reset(self):
        """Reset filter state."""
        self.state = None
        self.covariance = None


def compute_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.

    Args:
        eye_landmarks: Array of shape (N, 2) with eye contour points

    Returns:
        EAR value (lower values indicate closed eye)
    """
    if len(eye_landmarks) < 6:
        return 1.0

    # Vertical distances
    v1 = euclidean(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean(eye_landmarks[2], eye_landmarks[4])

    # Horizontal distance
    h = euclidean(eye_landmarks[0], eye_landmarks[3])

    if h == 0:
        return 1.0

    ear = (v1 + v2) / (2.0 * h)
    return ear


def normalize_point(point: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    """Normalize point to [0, 1] range."""
    return (point[0] / width, point[1] / height)


def denormalize_point(
    point: Tuple[float, float],
    width: int,
    height: int
) -> Tuple[int, int]:
    """Denormalize point from [0, 1] to pixel coordinates."""
    return (int(point[0] * width), int(point[1] * height))


def get_iris_center(iris_landmarks: np.ndarray) -> np.ndarray:
    """
    Get iris center from MediaPipe iris landmarks.

    Args:
        iris_landmarks: Array of shape (5, 2) with iris points

    Returns:
        Center point [x, y]
    """
    return np.mean(iris_landmarks, axis=0)


def compute_head_pose_compensation(
    face_bbox: Tuple[int, int, int, int],
    reference_bbox: Optional[Tuple[int, int, int, int]]
) -> Tuple[float, float]:
    """
    Compute offset compensation for head movement.

    Args:
        face_bbox: Current face bounding box (x, y, w, h)
        reference_bbox: Reference face bounding box from calibration

    Returns:
        Offset (dx, dy) to compensate for head movement
    """
    if reference_bbox is None:
        return (0.0, 0.0)

    # Compute center displacement
    curr_center_x = face_bbox[0] + face_bbox[2] / 2
    curr_center_y = face_bbox[1] + face_bbox[3] / 2

    ref_center_x = reference_bbox[0] + reference_bbox[2] / 2
    ref_center_y = reference_bbox[1] + reference_bbox[3] / 2

    # Normalize by face size
    dx = (curr_center_x - ref_center_x) / reference_bbox[2]
    dy = (curr_center_y - ref_center_y) / reference_bbox[3]

    return (dx, dy)


class StabilityBuffer:
    """Buffer to stabilize measurements by averaging over time."""

    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.buffer: List[np.ndarray] = []

    def add(self, value: np.ndarray):
        """Add value to buffer."""
        self.buffer.append(value.copy())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get_stable_value(self) -> Optional[np.ndarray]:
        """Get averaged stable value."""
        if not self.buffer:
            return None
        return np.mean(self.buffer, axis=0)

    def is_stable(self, threshold: float = 0.02) -> bool:
        """Check if buffer values are stable (low variance)."""
        if len(self.buffer) < self.buffer_size:
            return False

        values = np.array(self.buffer)
        std = np.std(values, axis=0)
        return np.all(std < threshold)

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
