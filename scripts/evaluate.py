"""Evaluation script for eye tracking accuracy."""

import time
import argparse
import numpy as np
import cv2
from src.eyemouse.capture import CameraCapture
from src.eyemouse.detector import FaceDetector
from src.eyemouse.calibration import CalibrationManager, CalibrationEvaluator


def run_evaluation(num_test_points: int = 9):
    """
    Run accuracy evaluation of the eye tracking system.

    Args:
        num_test_points: Number of test points to evaluate
    """
    print("=" * 60)
    print("Eye Control Mouse - Accuracy Evaluation")
    print("=" * 60)

    # Initialize components
    print("\n[1/5] Initializing camera...")
    camera = CameraCapture(camera_id=0, width=640, height=480, fps=30)

    if not camera.start():
        print("ERROR: Failed to open camera")
        return

    print("[2/5] Initializing face detector...")
    detector = FaceDetector(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Assume 1920x1080 screen (adjust as needed)
    screen_width, screen_height = 1920, 1080

    print(f"[3/5] Starting calibration (screen: {screen_width}x{screen_height})...")
    calibration_manager = CalibrationManager(
        screen_width=screen_width,
        screen_height=screen_height,
        num_points=9,
        mapping_method="polynomial"
    )

    calibration_manager.start_calibration()

    # Calibration loop
    for i in range(9):
        target = calibration_manager.get_current_target_point()
        print(f"\nCalibration point {i+1}/9: Look at ({target[0]}, {target[1]})")
        print("Press SPACE when ready, or 'q' to quit")

        samples_collected = 0
        while True:
            result = camera.read(timeout=0.1)
            if result is None:
                continue

            frame, _ = result
            detection_result = detector.process(frame)

            if detection_result is not None:
                # Draw landmarks
                frame = detection_result.draw_landmarks(frame)

                # Collect samples
                gaze_point = detection_result.get_gaze_point()
                calibration_manager.add_gaze_sample(gaze_point)
                samples_collected += 1

                # Show progress
                cv2.putText(
                    frame,
                    f"Samples: {samples_collected}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                if calibration_manager.is_current_point_stable():
                    cv2.putText(
                        frame,
                        "STABLE - Press SPACE",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and calibration_manager.is_current_point_stable():
                calibration_manager.confirm_current_point()
                break
            elif key == ord('q'):
                camera.stop()
                detector.close()
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    if not calibration_manager.is_complete():
        print("\nERROR: Calibration failed")
        camera.stop()
        detector.close()
        return

    mapper = calibration_manager.get_mapper()
    cal_error = mapper.get_calibration_error()
    print(f"\n[4/5] Calibration complete! Average error: {cal_error:.1f} pixels")

    # Evaluation phase
    print(f"\n[5/5] Starting evaluation with {num_test_points} test points...")
    evaluator = CalibrationEvaluator(mapper)

    # Generate test grid
    test_points = []
    margin = 0.15
    for row in range(3):
        for col in range(3):
            if len(test_points) >= num_test_points:
                break
            x = (margin + (1 - 2 * margin) * col / 2) * screen_width
            y = (margin + (1 - 2 * margin) * row / 2) * screen_height
            test_points.append((int(x), int(y)))

    # Collect test samples
    for i, test_point in enumerate(test_points):
        print(f"\nTest point {i+1}/{len(test_points)}: Look at ({test_point[0]}, {test_point[1]})")
        print("Press SPACE when ready")

        samples = []
        while True:
            result = camera.read(timeout=0.1)
            if result is None:
                continue

            frame, _ = result
            detection_result = detector.process(frame)

            if detection_result is not None:
                frame = detection_result.draw_landmarks(frame)
                gaze_point = detection_result.get_gaze_point()
                samples.append(gaze_point)

                if len(samples) > 30:  # Keep last 30 samples
                    samples.pop(0)

                cv2.putText(
                    frame,
                    f"Samples: {len(samples)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Evaluation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and len(samples) >= 20:
                # Use average of samples
                avg_gaze = np.mean(samples, axis=0)
                evaluator.add_test_sample(avg_gaze, np.array(test_point))
                break
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()
    camera.stop()
    detector.close()

    # Compute results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    metrics = evaluator.evaluate()

    print(f"\nNumber of test samples: {metrics['num_samples']}")
    print(f"Mean error: {metrics['mean_error']:.1f} pixels")
    print(f"Std deviation: {metrics['std_error']:.1f} pixels")
    print(f"Max error: {metrics['max_error']:.1f} pixels")

    # Convert to visual angle (approximate)
    # Assume 24" monitor at 60cm viewing distance
    pixels_per_cm = screen_width / 53.0  # 24" ≈ 53cm width
    viewing_distance_cm = 60

    def pixels_to_degrees(pixels):
        cm = pixels / pixels_per_cm
        radians = np.arctan(cm / viewing_distance_cm)
        return np.degrees(radians)

    mean_degrees = pixels_to_degrees(metrics['mean_error'])

    print(f"\nMean error in visual angle: {mean_degrees:.2f}°")

    # Quality assessment
    print("\n" + "-" * 60)
    if metrics['mean_error'] < 50:
        print("Quality: EXCELLENT - Suitable for precise tasks")
    elif metrics['mean_error'] < 100:
        print("Quality: GOOD - Suitable for general mouse control")
    elif metrics['mean_error'] < 150:
        print("Quality: ACCEPTABLE - May need recalibration")
    else:
        print("Quality: POOR - Recalibration recommended")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate eye tracking accuracy")
    parser.add_argument(
        "--test-points",
        type=int,
        default=9,
        help="Number of test points (default: 9)"
    )

    args = parser.parse_args()

    try:
        run_evaluation(args.test_points)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
