"""Tests for gaze tracking and mapping."""

import numpy as np
import pytest
from src.eyemouse.tracker import GazeMapper


class TestGazeMapper:
    """Test gaze mapping functionality."""

    def create_calibration_data(self):
        """Create synthetic calibration data (3x3 grid)."""
        # Image points (simulated gaze positions)
        image_points = [
            [100, 100], [320, 100], [540, 100],
            [100, 240], [320, 240], [540, 240],
            [100, 380], [320, 380], [540, 380]
        ]

        # Screen points (target positions on 1920x1080 screen)
        screen_points = [
            [200, 200], [960, 200], [1720, 200],
            [200, 540], [960, 540], [1720, 540],
            [200, 880], [960, 880], [1720, 880]
        ]

        return image_points, screen_points

    def test_affine_mapping(self):
        """Test affine transformation mapping."""
        mapper = GazeMapper(method="affine")
        image_points, screen_points = self.create_calibration_data()

        for img_pt, scr_pt in zip(image_points, screen_points):
            mapper.add_calibration_point(
                np.array(img_pt),
                np.array(scr_pt)
            )

        success = mapper.fit()
        assert success
        assert mapper.is_calibrated()

        # Test mapping accuracy on calibration points
        for img_pt, scr_pt in zip(image_points, screen_points):
            mapped = mapper.map_to_screen(np.array(img_pt))
            assert mapped is not None

            # Should be reasonably close
            error = np.linalg.norm(mapped - np.array(scr_pt))
            assert error < 100  # Within 100 pixels

    def test_polynomial_mapping(self):
        """Test polynomial mapping."""
        mapper = GazeMapper(method="polynomial")
        image_points, screen_points = self.create_calibration_data()

        for img_pt, scr_pt in zip(image_points, screen_points):
            mapper.add_calibration_point(
                np.array(img_pt),
                np.array(scr_pt)
            )

        success = mapper.fit()
        assert success
        assert mapper.is_calibrated()

        # Test mapping
        test_point = np.array([320, 240])  # Center
        mapped = mapper.map_to_screen(test_point)

        assert mapped is not None
        # Should map near screen center
        assert 800 < mapped[0] < 1100
        assert 400 < mapped[1] < 700

    def test_rbf_mapping(self):
        """Test RBF interpolation mapping."""
        mapper = GazeMapper(method="rbf")
        image_points, screen_points = self.create_calibration_data()

        for img_pt, scr_pt in zip(image_points, screen_points):
            mapper.add_calibration_point(
                np.array(img_pt),
                np.array(scr_pt)
            )

        success = mapper.fit()
        assert success
        assert mapper.is_calibrated()

        # RBF should interpolate calibration points very accurately
        for img_pt, scr_pt in zip(image_points, screen_points):
            mapped = mapper.map_to_screen(np.array(img_pt))
            assert mapped is not None

            error = np.linalg.norm(mapped - np.array(scr_pt))
            assert error < 50  # Should be very accurate

    def test_insufficient_points(self):
        """Test that fitting fails with insufficient points."""
        mapper = GazeMapper(method="affine")

        # Add only 2 points (need at least 4)
        mapper.add_calibration_point(np.array([100, 100]), np.array([200, 200]))
        mapper.add_calibration_point(np.array([200, 200]), np.array([400, 400]))

        success = mapper.fit()
        assert not success
        assert not mapper.is_calibrated()

    def test_calibration_error(self):
        """Test calibration error computation."""
        mapper = GazeMapper(method="affine")
        image_points, screen_points = self.create_calibration_data()

        for img_pt, scr_pt in zip(image_points, screen_points):
            mapper.add_calibration_point(
                np.array(img_pt),
                np.array(scr_pt)
            )

        mapper.fit()

        error = mapper.get_calibration_error()
        assert error is not None
        assert error >= 0
        # Affine should have some error due to model limitations
        assert error < 200  # Reasonable error threshold

    def test_clear_calibration(self):
        """Test clearing calibration data."""
        mapper = GazeMapper(method="affine")
        mapper.add_calibration_point(np.array([100, 100]), np.array([200, 200]))

        mapper.clear_calibration()

        assert len(mapper.calibration_points) == 0
        assert not mapper.is_calibrated()

    def test_mapping_before_calibration(self):
        """Test that mapping returns None before calibration."""
        mapper = GazeMapper(method="affine")

        result = mapper.map_to_screen(np.array([100, 100]))
        assert result is None


class TestGazeMapperIntegration:
    """Integration tests for gaze mapping."""

    def test_realistic_calibration_scenario(self):
        """Test a realistic calibration and mapping scenario."""
        mapper = GazeMapper(method="polynomial")

        # Simulate 9-point calibration on 1920x1080 screen
        # with 640x480 camera feed

        calibration_data = [
            # (image_x, image_y, screen_x, screen_y)
            (160, 120, 192, 108),    # Top-left
            (320, 120, 960, 108),    # Top-center
            (480, 120, 1728, 108),   # Top-right
            (160, 240, 192, 540),    # Middle-left
            (320, 240, 960, 540),    # Center
            (480, 240, 1728, 540),   # Middle-right
            (160, 360, 192, 972),    # Bottom-left
            (320, 360, 960, 972),    # Bottom-center
            (480, 360, 1728, 972),   # Bottom-right
        ]

        for img_x, img_y, scr_x, scr_y in calibration_data:
            # Add some noise to simulate real measurements
            noisy_img_x = img_x + np.random.normal(0, 5)
            noisy_img_y = img_y + np.random.normal(0, 5)

            mapper.add_calibration_point(
                np.array([noisy_img_x, noisy_img_y]),
                np.array([scr_x, scr_y])
            )

        success = mapper.fit()
        assert success

        # Test mapping accuracy on a test point
        test_image_point = np.array([320, 240])  # Center of camera
        mapped = mapper.map_to_screen(test_image_point)

        assert mapped is not None

        # Should map near screen center (960, 540)
        error = np.linalg.norm(mapped - np.array([960, 540]))
        assert error < 150  # Reasonable accuracy
