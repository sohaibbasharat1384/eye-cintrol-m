"""Gaze tracking with smoothing and screen mapping."""

from typing import Optional, Tuple
import numpy as np
from .utils import SimpleKalmanFilter, EWMAFilter
from .detector import FaceDetectionResult


class GazeTracker:
    """Track gaze and map to screen coordinates."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        smoothing_method: str = "kalman",
        smoothing_factor: float = 0.3
    ):
        """
        Initialize gaze tracker.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            smoothing_method: "kalman", "ewma", or "none"
            smoothing_factor: Smoothing strength (0-1, lower = more smoothing)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.smoothing_method = smoothing_method

        # Initialize filters
        if smoothing_method == "kalman":
            self.filter = SimpleKalmanFilter(
                process_variance=1e-3,
                measurement_variance=0.1
            )
        elif smoothing_method == "ewma":
            self.filter = EWMAFilter(alpha=smoothing_factor)
        else:
            self.filter = None

        self.mapper: Optional['GazeMapper'] = None
        self.reference_bbox: Optional[Tuple[int, int, int, int]] = None

        self._last_raw_gaze: Optional[np.ndarray] = None
        self._last_mapped_gaze: Optional[np.ndarray] = None

    def set_smoothing_factor(self, factor: float):
        """Update smoothing factor (0-1)."""
        factor = np.clip(factor, 0.01, 1.0)

        if self.smoothing_method == "kalman" and isinstance(self.filter, SimpleKalmanFilter):
            # Adjust measurement variance (lower = trust measurements more = less smoothing)
            self.filter.measurement_variance = (1.0 - factor) * 0.5
            self.filter.R = np.eye(2) * self.filter.measurement_variance

        elif self.smoothing_method == "ewma" and isinstance(self.filter, EWMAFilter):
            self.filter.alpha = factor

    def set_mapper(self, mapper: 'GazeMapper'):
        """Set the gaze-to-screen mapper."""
        self.mapper = mapper

    def process(
        self,
        detection_result: FaceDetectionResult,
        dt: Optional[float] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Process detection result and return screen coordinates.

        Args:
            detection_result: Face detection result
            dt: Time delta since last update

        Returns:
            (screen_x, screen_y) tuple or None
        """
        # Get raw gaze point in image coordinates
        raw_gaze = detection_result.get_gaze_point()
        self._last_raw_gaze = raw_gaze.copy()

        # Get face bbox for head pose compensation
        face_bbox = detection_result.get_face_bbox()

        if self.reference_bbox is None:
            self.reference_bbox = face_bbox

        # Normalize gaze relative to face (helps with head movement)
        normalized_gaze = detection_result.get_normalized_gaze_point()

        # Map to screen coordinates
        if self.mapper is None or not self.mapper.is_calibrated():
            # Fallback: simple direct mapping
            screen_x = int(normalized_gaze[0] * self.screen_width)
            screen_y = int(normalized_gaze[1] * self.screen_height)
            mapped_point = np.array([screen_x, screen_y], dtype=float)
        else:
            # Use calibrated mapping on normalized gaze
            mapped_point = self.mapper.map_to_screen(normalized_gaze)


            if mapped_point is None:
                return None

        # Apply smoothing filter
        if self.filter is not None:
            filtered_point = self.filter.update(mapped_point, dt=dt)
        else:
            filtered_point = mapped_point

        self._last_mapped_gaze = filtered_point.copy()

        # Clamp to screen bounds
        screen_x = int(np.clip(filtered_point[0], 0, self.screen_width - 1))
        screen_y = int(np.clip(filtered_point[1], 0, self.screen_height - 1))

        return (screen_x, screen_y)

    def reset_filter(self):
        """Reset smoothing filter state."""
        if self.filter is not None:
            self.filter.reset()

    def get_last_raw_gaze(self) -> Optional[np.ndarray]:
        """Get last raw gaze point in image coordinates."""
        return self._last_raw_gaze

    def get_last_mapped_gaze(self) -> Optional[np.ndarray]:
        """Get last mapped gaze point in screen coordinates."""
        return self._last_mapped_gaze


class GazeMapper:
    """Map gaze points from image coordinates to screen coordinates."""

    def __init__(self, method: str = "polynomial"):
        """
        Initialize gaze mapper.

        Args:
            method: Mapping method - "polynomial", "rbf", or "affine"
        """
        self.method = method
        self.calibration_points: list = []  # List of (image_point, screen_point) tuples
        self.model = None
        self._is_calibrated = False

    def add_calibration_point(
        self,
        image_point: np.ndarray,
        screen_point: np.ndarray
    ):
        """
        Add a calibration point.

        Args:
            image_point: [x, y] in image coordinates
            screen_point: [x, y] in screen coordinates
        """
        self.calibration_points.append((
            np.asarray(image_point, dtype=float),
            np.asarray(screen_point, dtype=float)
        ))

    def clear_calibration(self):
        """Clear calibration data."""
        self.calibration_points.clear()
        self.model = None
        self._is_calibrated = False

    def fit(self) -> bool:
        """
        Fit mapping model from calibration points.

        Returns:
            True if successful
        """
        if len(self.calibration_points) < 4:
            return False

        # Extract points
        image_points = np.array([p[0] for p in self.calibration_points])
        screen_points = np.array([p[1] for p in self.calibration_points])

        try:
            if self.method == "affine":
                self.model = self._fit_affine(image_points, screen_points)
            elif self.method == "polynomial":
                self.model = self._fit_polynomial(image_points, screen_points, degree=2)
            elif self.method == "rbf":
                self.model = self._fit_rbf(image_points, screen_points)
            else:
                return False

            self._is_calibrated = True
            return True

        except Exception as e:
            print(f"Calibration fitting failed: {e}")
            return False

    def _fit_affine(
        self,
        image_points: np.ndarray,
        screen_points: np.ndarray
    ) -> dict:
        """Fit affine transformation."""
        # Solve: screen = A @ image + b
        # Use least squares for overdetermined system

        n = len(image_points)
        X = np.hstack([image_points, np.ones((n, 1))])  # Add bias column
        Y = screen_points

        # Solve separately for x and y
        params_x = np.linalg.lstsq(X, Y[:, 0], rcond=None)[0]
        params_y = np.linalg.lstsq(X, Y[:, 1], rcond=None)[0]

        return {
            'type': 'affine',
            'params_x': params_x,
            'params_y': params_y
        }

    def _fit_polynomial(
        self,
        image_points: np.ndarray,
        screen_points: np.ndarray,
        degree: int = 2
    ) -> dict:
        """Fit polynomial transformation."""
        from itertools import combinations_with_replacement

        # Generate polynomial features
        def polynomial_features(points, deg):
            n_samples = points.shape[0]
            features = [np.ones(n_samples)]

            for d in range(1, deg + 1):
                for combination in combinations_with_replacement([0, 1], d):
                    feature = np.ones(n_samples)
                    for idx in combination:
                        feature *= points[:, idx]
                    features.append(feature)

            return np.column_stack(features)

        X = polynomial_features(image_points, degree)
        Y = screen_points

        # Fit separately for x and y
        params_x = np.linalg.lstsq(X, Y[:, 0], rcond=None)[0]
        params_y = np.linalg.lstsq(X, Y[:, 1], rcond=None)[0]

        return {
            'type': 'polynomial',
            'degree': degree,
            'params_x': params_x,
            'params_y': params_y
        }

    def _fit_rbf(
        self,
        image_points: np.ndarray,
        screen_points: np.ndarray
    ) -> dict:
        """Fit RBF (Radial Basis Function) interpolation."""
        from scipy.interpolate import RBFInterpolator

        # Fit RBF for x and y separately
        rbf_x = RBFInterpolator(
            image_points,
            screen_points[:, 0],
            kernel='thin_plate_spline',
            smoothing=0.1
        )
        rbf_y = RBFInterpolator(
            image_points,
            screen_points[:, 1],
            kernel='thin_plate_spline',
            smoothing=0.1
        )

        return {
            'type': 'rbf',
            'rbf_x': rbf_x,
            'rbf_y': rbf_y
        }

    def map_to_screen(self, image_point: np.ndarray) -> Optional[np.ndarray]:
        """
        Map image point to screen coordinates.

        Args:
            image_point: [x, y] in image coordinates

        Returns:
            [x, y] in screen coordinates or None
        """
        if not self._is_calibrated or self.model is None:
            return None

        try:
            image_point = np.asarray(image_point, dtype=float).reshape(1, -1)

            if self.model['type'] == 'affine':
                X = np.hstack([image_point, np.ones((1, 1))])
                screen_x = float(X @ self.model['params_x'])
                screen_y = float(X @ self.model['params_y'])

            elif self.model['type'] == 'polynomial':
                from itertools import combinations_with_replacement

                degree = self.model['degree']
                features = [1.0]

                pt = image_point[0]
                for d in range(1, degree + 1):
                    for combination in combinations_with_replacement([0, 1], d):
                        feature = 1.0
                        for idx in combination:
                            feature *= pt[idx]
                        features.append(feature)

                X = np.array(features).reshape(1, -1)
                screen_x = float(X @ self.model['params_x'])
                screen_y = float(X @ self.model['params_y'])

            elif self.model['type'] == 'rbf':
                screen_x = float(self.model['rbf_x'](image_point)[0])
                screen_y = float(self.model['rbf_y'](image_point)[0])

            else:
                return None

            return np.array([screen_x, screen_y], dtype=float)

        except Exception as e:
            print(f"Mapping failed: {e}")
            return None

    def is_calibrated(self) -> bool:
        """Check if mapper is calibrated."""
        return self._is_calibrated

    def get_calibration_error(self) -> Optional[float]:
        """
        Compute mean calibration error in pixels.

        Returns:
            Mean error or None if not calibrated
        """
        if not self._is_calibrated or not self.calibration_points:
            return None

        errors = []
        for image_point, true_screen_point in self.calibration_points:
            predicted = self.map_to_screen(image_point)
            if predicted is not None:
                error = np.linalg.norm(predicted - true_screen_point)
                errors.append(error)

        return float(np.mean(errors)) if errors else None
