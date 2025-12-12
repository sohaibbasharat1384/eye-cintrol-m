"""Tests for calibration system."""

import numpy as np
import pytest
from src.eyemouse.calibration import CalibrationManager, CalibrationEvaluator
from src.eyemouse.tracker import GazeMapper


class TestCalibrationManager:
    """Test calibration manager."""

    def test_initialization(self):
        """Test calibration manager initialization."""
        manager = CalibrationManager(
            screen_width=1920,
            screen_height=1080,
            num_points=9
        )

        assert len(manager.calibration_points) == 9
        assert not manager.is_calibrating()
        assert not manager.is_complete()

    def test_calibration_grid_generation(self):
        """Test calibration point grid generation."""
        manager = CalibrationManager(
            screen_width=1920,
            screen_height=1080,
            num_points=9
        )

        points = manager.calibration_points
        assert len(points) == 9

        # Check that points span the screen with margins
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Should have points near edges but not at exact 0,0
        assert min(xs) > 100  # Has margin
        assert max(xs) < 1820  # Has margin
        assert min(ys) > 50
        assert max(ys) < 1030

    def test_start_calibration(self):
        """Test starting calibration."""
        manager = CalibrationManager(
            screen_width=1920,
            screen_height=1080,
            num_points=9
        )

        manager.start_calibration()

        assert manager.is_calibrating()
        assert not manager.is_complete()

        progress = manager.get_progress()
        assert progress == (0, 9)

    def test_calibration_flow(self):
        """Test complete calibration flow."""
        manager = CalibrationManager(
            screen_width=1920,
            screen_height=1080,
            num_points=9
        )

        manager.start_calibration()

        # Simulate collecting samples for each point
        for i in range(9):
            target = manager.get_current_target_point()
            assert target is not None

            # Add stable gaze samples
            for _ in range(20):
                # Simulate looking at target with small noise
                gaze = np.array([
                    300 + i * 50 + np.random.normal(0, 2),
                    200 + np.random.normal(0, 2)
                ])
                manager.add_gaze_sample(gaze)

            # Should be stable
            assert manager.is_current_point_stable()

            # Confirm point
            has_more = manager.confirm_current_point()

            if i < 8:
                assert has_more
            else:
                # Last point - calibration should complete
                assert not has_more
                assert manager.is_complete()
                assert not manager.is_calibrating()

    def test_insufficient_samples(self):
        """Test that unstable samples are not confirmed."""
        manager = CalibrationManager(
            screen_width=1920,
            screen_height=1080,
            num_points=9
        )

        manager.start_calibration()

        # Add only a few samples
        for _ in range(3):
            manager.add_gaze_sample(np.array([100.0, 100.0]))

        # Should not be stable yet
        assert not manager.is_current_point_stable()

    def test_reset(self):
        """Test calibration reset."""
        manager = CalibrationManager(
            screen_width=1920,
            screen_height=1080,
            num_points=9
        )

        manager.start_calibration()
        manager.reset()

        assert not manager.is_calibrating()
        assert not manager.is_complete()
        assert manager.get_current_target_point() is None


class TestCalibrationEvaluator:
    """Test calibration evaluator."""

    def test_evaluate_accuracy(self):
        """Test accuracy evaluation."""
        # Create a simple mapper
        mapper = GazeMapper(method="affine")

        # Perfect mapping: screen = image * 2
        for i in range(5):
            img_pt = np.array([i * 100.0, i * 100.0])
            scr_pt = img_pt * 2
            mapper.add_calibration_point(img_pt, scr_pt)

        mapper.fit()

        # Create evaluator
        evaluator = CalibrationEvaluator(mapper)

        # Add test samples
        for i in range(5):
            img_pt = np.array([i * 100.0, i * 100.0])
            true_scr_pt = img_pt * 2
            evaluator.add_test_sample(img_pt, true_scr_pt)

        # Evaluate
        metrics = evaluator.evaluate()

        assert metrics['num_samples'] == 5
        assert metrics['mean_error'] is not None
        assert metrics['std_error'] is not None
        assert metrics['max_error'] is not None

        # Should have low error on training data
        assert metrics['mean_error'] < 10

    def test_evaluate_with_errors(self):
        """Test evaluation with mapping errors."""
        mapper = GazeMapper(method="affine")

        # Create calibration with some mapping complexity
        for i in range(5):
            img_pt = np.array([i * 100.0, i * 50.0])
            scr_pt = np.array([i * 200.0, i * 150.0])
            mapper.add_calibration_point(img_pt, scr_pt)

        mapper.fit()

        evaluator = CalibrationEvaluator(mapper)

        # Add test samples at different locations
        test_data = [
            (np.array([50, 25]), np.array([100, 75])),
            (np.array([150, 75]), np.array([300, 225])),
            (np.array([250, 125]), np.array([500, 375])),
        ]

        for img_pt, true_scr_pt in test_data:
            evaluator.add_test_sample(img_pt, true_scr_pt)

        metrics = evaluator.evaluate()

        assert metrics['num_samples'] == 3
        # Will have some error due to affine limitations
        assert 0 <= metrics['mean_error'] < 200

    def test_clear_samples(self):
        """Test clearing test samples."""
        mapper = GazeMapper(method="affine")
        evaluator = CalibrationEvaluator(mapper)

        evaluator.add_test_sample(np.array([1, 1]), np.array([2, 2]))
        evaluator.clear()

        metrics = evaluator.evaluate()
        assert metrics['num_samples'] == 0
