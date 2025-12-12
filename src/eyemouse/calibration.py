"""Calibration system for gaze mapping."""

import json
from typing import List, Tuple, Optional, Callable
import numpy as np
from .tracker import GazeMapper
from .utils import StabilityBuffer


class CalibrationManager:
    """Manage calibration process."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        num_points: int = 9,
        mapping_method="affine"

    ):
        """
        Initialize calibration manager.

        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            num_points: Number of calibration points (9, 12, or 16 recommended)
            mapping_method: Mapping method ("polynomial", "rbf", or "affine")
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_points = num_points
        self.mapping_method = mapping_method

        self.calibration_points = self._generate_calibration_grid()
        self.current_point_idx = 0

        self.gaze_buffer = StabilityBuffer(buffer_size=15)  # ~500ms at 30fps
        self.mapper = GazeMapper(method=mapping_method)

        self._is_calibrating = False
        self._calibration_complete = False

    def _generate_calibration_grid(self) -> List[Tuple[int, int]]:
        """Generate calibration point grid."""
        points = []

        if self.num_points == 9:
            # 3x3 grid
            rows, cols = 3, 3
        elif self.num_points == 12:
            # 3x4 grid
            rows, cols = 3, 4
        elif self.num_points == 16:
            # 4x4 grid
            rows, cols = 4, 4
        else:
            # Default to 3x3
            rows, cols = 3, 3

        # Add margins
        margin_x = self.screen_width * 0.1
        margin_y = self.screen_height * 0.1

        for row in range(rows):
            for col in range(cols):
                x = margin_x + (self.screen_width - 2 * margin_x) * col / (cols - 1)
                y = margin_y + (self.screen_height - 2 * margin_y) * row / (rows - 1)
                points.append((int(x), int(y)))

        return points

    def start_calibration(self):
        """Start calibration process."""
        self._is_calibrating = True
        self._calibration_complete = False
        self.current_point_idx = 0
        self.gaze_buffer.clear()
        self.mapper.clear_calibration()

    def add_gaze_sample(self, gaze_point: np.ndarray):
        """
        Add gaze sample for current calibration point.

        Args:
            gaze_point: [x, y] normalized gaze point (0–1) relative to face
        """

        if not self._is_calibrating:
            return

        self.gaze_buffer.add(gaze_point)

    def is_current_point_stable(self) -> bool:
        """Check if enough stable samples collected for current point."""
        return self.gaze_buffer.is_stable(threshold=5.0)  # 5 pixels std

    def confirm_current_point(self) -> bool:
        """
        Confirm current calibration point and move to next.

        Returns:
            True if more points remain, False if calibration complete
        """
        if not self._is_calibrating or self.current_point_idx >= len(self.calibration_points):
            return False

        # Get stable gaze point
        stable_gaze = self.gaze_buffer.get_stable_value()

        if stable_gaze is None:
            return False

        # Add to mapper
        screen_point = np.array(self.calibration_points[self.current_point_idx])
        self.mapper.add_calibration_point(stable_gaze, screen_point)

        # Debug output
        print(f"Point {self.current_point_idx + 1}: gaze={stable_gaze}, screen={screen_point}")

        # Move to next point
        self.current_point_idx += 1
        self.gaze_buffer.clear()

        # Check if calibration complete
        if self.current_point_idx >= len(self.calibration_points):
            return self._finish_calibration()

        return True

    def _finish_calibration(self) -> bool:
        """Finish calibration and fit mapper."""
        # Print gaze range statistics
        if self.mapper.calibration_points:
            gaze_points = np.array([p[0] for p in self.mapper.calibration_points])
            print(f"\nGaze range: X=[{gaze_points[:, 0].min():.3f}, {gaze_points[:, 0].max():.3f}], "
                  f"Y=[{gaze_points[:, 1].min():.3f}, {gaze_points[:, 1].max():.3f}]")
            print(f"Gaze variation: X={gaze_points[:, 0].std():.3f}, Y={gaze_points[:, 1].std():.3f}\n")

        success = self.mapper.fit()

        if success:
            self._calibration_complete = True
            self._is_calibrating = False
            print(f"Calibration complete. Error: {self.mapper.get_calibration_error():.1f} px")
        else:
            print("Calibration failed - not enough points or fitting error")

        return success

    def get_current_target_point(self) -> Optional[Tuple[int, int]]:
        """Get current calibration target point."""
        if not self._is_calibrating or self.current_point_idx >= len(self.calibration_points):
            return None

        return self.calibration_points[self.current_point_idx]

    def get_progress(self) -> Tuple[int, int]:
        """
        Get calibration progress.

        Returns:
            (current_point, total_points) tuple
        """
        return (self.current_point_idx, len(self.calibration_points))

    def is_calibrating(self) -> bool:
        """Check if calibration is in progress."""
        return self._is_calibrating

    def is_complete(self) -> bool:
        """Check if calibration is complete."""
        return self._calibration_complete

    def get_mapper(self) -> GazeMapper:
        """Get calibrated mapper."""
        return self.mapper

    def save_calibration(self, filepath: str) -> bool:
        """
        Save calibration data to file.

        Args:
            filepath: Path to save file

        Returns:
            True if successful
        """
        if not self._calibration_complete:
            return False

        try:
            data = {
                'screen_width': self.screen_width,
                'screen_height': self.screen_height,
                'mapping_method': self.mapping_method,
                'calibration_points': [
                    {
                        'image': img_pt.tolist(),
                        'screen': scr_pt.tolist()
                    }
                    for img_pt, scr_pt in self.mapper.calibration_points
                ],
                'calibration_error': self.mapper.get_calibration_error()
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Failed to save calibration: {e}")
            return False

    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration data from file.

        Args:
            filepath: Path to calibration file

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Verify screen dimensions match
            if (data['screen_width'] != self.screen_width or
                data['screen_height'] != self.screen_height):
                print("Warning: Screen dimensions don't match calibration data")
                return False

            # Reconstruct mapper
            self.mapper = GazeMapper(method=data['mapping_method'])

            for point_data in data['calibration_points']:
                img_pt = np.array(point_data['image'])
                scr_pt = np.array(point_data['screen'])
                self.mapper.add_calibration_point(img_pt, scr_pt)

            # Fit mapper
            if self.mapper.fit():
                self._calibration_complete = True
                self._is_calibrating = False
                print(f"Calibration loaded. Error: {self.mapper.get_calibration_error():.1f} px")
                return True
            else:
                return False

        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False

    def reset(self):
        """Reset calibration state."""
        self._is_calibrating = False
        self._calibration_complete = False
        self.current_point_idx = 0
        self.gaze_buffer.clear()
        self.mapper.clear_calibration()


class CalibrationEvaluator:
    """Evaluate calibration accuracy."""

    def __init__(self, mapper: GazeMapper):
        """
        Initialize evaluator.

        Args:
            mapper: Calibrated gaze mapper
        """
        self.mapper = mapper
        self.test_samples: List[Tuple[np.ndarray, np.ndarray]] = []

    def add_test_sample(self, image_point: np.ndarray, true_screen_point: np.ndarray):
        """Add test sample for evaluation."""
        self.test_samples.append((
            np.asarray(image_point),
            np.asarray(true_screen_point)
        ))

    def evaluate(self) -> dict:
        """
        Evaluate calibration accuracy.

        Returns:
            Dictionary with accuracy metrics
        """
        if not self.test_samples:
            return {
                'mean_error': None,
                'std_error': None,
                'max_error': None,
                'num_samples': 0
            }

        errors = []
        for image_pt, true_screen_pt in self.test_samples:
            predicted = self.mapper.map_to_screen(image_pt)
            if predicted is not None:
                error = np.linalg.norm(predicted - true_screen_pt)
                errors.append(error)

        if not errors:
            return {
                'mean_error': None,
                'std_error': None,
                'max_error': None,
                'num_samples': 0
            }

        return {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'num_samples': len(errors)
        }

    def clear(self):
        """Clear test samples."""
        self.test_samples.clear()
