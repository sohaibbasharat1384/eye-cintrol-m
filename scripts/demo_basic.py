"""Basic demo showing camera, landmarks, and gaze tracking."""

import cv2
import numpy as np
from src.eyemouse.capture import CameraCapture
from src.eyemouse.detector import FaceDetector


def main():
    """Run basic demo."""
    print("=" * 60)
    print("Eye Control Mouse - Basic Demo")
    print("=" * 60)
    print("\nThis demo shows:")
    print("- Camera feed")
    print("- Face and eye landmark detection")
    print("- Iris center tracking")
    print("\nPress 'q' to quit")
    print("=" * 60)

    # Initialize camera
    print("\nInitializing camera...")
    camera = CameraCapture(camera_id=0, width=640, height=480, fps=30)

    if not camera.start():
        print("ERROR: Failed to open camera")
        return

    # Initialize detector
    print("Initializing face detector...")
    detector = FaceDetector(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("\nDemo running... Press 'q' to quit\n")

    # Main loop
    frame_count = 0
    while True:
        result = camera.read(timeout=0.1)

        if result is None:
            continue

        frame, timestamp = result
        frame_count += 1

        # Process frame
        detection_result = detector.process(frame)

        if detection_result is not None:
            # Draw landmarks
            frame = detection_result.draw_landmarks(
                frame,
                draw_iris=True,
                draw_eyes=True,
                draw_face=False
            )

            # Get gaze point
            gaze_point = detection_result.get_gaze_point()

            # Draw gaze point
            cv2.circle(
                frame,
                tuple(gaze_point.astype(int)),
                8,
                (255, 0, 255),
                3
            )

            # Get face bbox
            bbox = detection_result.get_face_bbox()
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                (0, 255, 0),
                2
            )

            # Display info
            confidence = detection_result.get_confidence()
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Gaze: ({gaze_point[0]:.0f}, {gaze_point[1]:.0f})",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

        else:
            # No face detected
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Display frame
        cv2.imshow("Eye Control Mouse - Basic Demo", frame)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    print("\nCleaning up...")
    camera.stop()
    detector.close()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")
    print("Demo complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
