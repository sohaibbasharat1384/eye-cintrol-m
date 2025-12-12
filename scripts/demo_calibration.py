"""Demo showing calibration and mapping process."""

import cv2
import numpy as np
from src.eyemouse.capture import CameraCapture
from src.eyemouse.detector import FaceDetector
from src.eyemouse.calibration import CalibrationManager
from src.eyemouse.tracker import GazeTracker


def main():
    """Run calibration demo."""
    print("=" * 60)
    print("Eye Control Mouse - Calibration Demo")
    print("=" * 60)
    print("\nThis demo shows:")
    print("- 9-point calibration process")
    print("- Gaze-to-screen mapping")
    print("- Real-time cursor prediction")
    print("\nInstructions:")
    print("1. Look at each calibration point")
    print("2. Press SPACE when ready to confirm")
    print("3. After calibration, your gaze will control a virtual cursor")
    print("\nPress 'q' to quit")
    print("=" * 60)

    # Initialize components
    print("\nInitializing...")
    camera = CameraCapture(camera_id=0, width=640, height=480, fps=30)

    if not camera.start():
        print("ERROR: Failed to open camera")
        return

    detector = FaceDetector(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Use camera resolution for virtual screen
    cam_width, cam_height = camera.get_resolution()
    screen_width, screen_height = cam_width * 2, cam_height * 2

    calibration_manager = CalibrationManager(
        screen_width=screen_width,
        screen_height=screen_height,
        num_points=9,
        mapping_method="polynomial"
    )

    tracker = GazeTracker(
        screen_width=screen_width,
        screen_height=screen_height,
        smoothing_method="kalman",
        smoothing_factor=0.3
    )

    print("\nStarting calibration...\n")
    calibration_manager.start_calibration()

    # Calibration phase
    while calibration_manager.is_calibrating():
        result = camera.read(timeout=0.1)
        if result is None:
            continue

        frame, _ = result
        detection_result = detector.process(frame)

        if detection_result is not None:
            frame = detection_result.draw_landmarks(frame)

            # Get current target
            target = calibration_manager.get_current_target_point()
            current, total = calibration_manager.get_progress()

            # Draw target on frame (scaled to camera coordinates)
            target_x = int(target[0] * cam_width / screen_width)
            target_y = int(target[1] * cam_height / screen_height)

            cv2.circle(frame, (target_x, target_y), 20, (0, 0, 255), 2)
            cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)

            # Add gaze samples
            gaze_point = detection_result.get_gaze_point()
            calibration_manager.add_gaze_sample(gaze_point)

            # Show status
            cv2.putText(
                frame,
                f"Point {current + 1}/{total}",
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
            else:
                cv2.putText(
                    frame,
                    "Keep looking at target...",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

        cv2.imshow("Calibration Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if calibration_manager.is_current_point_stable():
                calibration_manager.confirm_current_point()
        elif key == ord('q'):
            camera.stop()
            detector.close()
            cv2.destroyAllWindows()
            return

    # Calibration complete
    if calibration_manager.is_complete():
        mapper = calibration_manager.get_mapper()
        error = mapper.get_calibration_error()
        print(f"\nCalibration complete! Average error: {error:.1f} pixels\n")

        tracker.set_mapper(mapper)

        # Tracking phase
        print("Now tracking your gaze. Press 'q' to quit.\n")

        # Create virtual screen visualization
        virtual_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        while True:
            result = camera.read(timeout=0.1)
            if result is None:
                continue

            frame, _ = result
            detection_result = detector.process(frame)

            if detection_result is not None:
                frame = detection_result.draw_landmarks(frame)

                # Get cursor position
                cursor_pos = tracker.process(detection_result)

                if cursor_pos is not None:
                    # Draw on virtual screen
                    virtual_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

                    # Draw cursor
                    cv2.circle(
                        virtual_screen,
                        cursor_pos,
                        15,
                        (0, 255, 0),
                        -1
                    )

                    # Draw trail (fade)
                    cv2.circle(
                        virtual_screen,
                        cursor_pos,
                        30,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        virtual_screen,
                        f"Cursor: ({cursor_pos[0]}, {cursor_pos[1]})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

            # Show both windows
            cv2.imshow("Calibration Demo - Camera", frame)
            cv2.imshow("Calibration Demo - Virtual Screen", virtual_screen)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    else:
        print("\nCalibration failed!")

    # Cleanup
    camera.stop()
    detector.close()
    cv2.destroyAllWindows()
    print("\nDemo complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
