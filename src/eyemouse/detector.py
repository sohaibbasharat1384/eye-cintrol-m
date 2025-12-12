"""Eye and iris detection using MediaPipe."""

from typing import Optional, Tuple, List
import numpy as np
import cv2
import mediapipe as mp


class FaceDetector:
    """Face and iris detection using MediaPipe FaceMesh."""

    # MediaPipe FaceMesh landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]  # Center is 468
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]  # Center is 473

    # For head pose estimation
    FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe FaceMesh detector.

        Args:
            static_image_mode: Whether to treat images as static
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.frame_width = 0
        self.frame_height = 0

    def process(self, frame: np.ndarray) -> Optional['FaceDetectionResult']:
        """
        Process frame and detect face landmarks.

        Args:
            frame: RGB or BGR image

        Returns:
            FaceDetectionResult or None if no face detected
        """
        self.frame_height, self.frame_width = frame.shape[:2]

        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # Use first face
        face_landmarks = results.multi_face_landmarks[0]

        return FaceDetectionResult(
            face_landmarks=face_landmarks,
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )

    def close(self):
        """Release resources."""
        self.face_mesh.close()


class FaceDetectionResult:
    """Container for face detection results."""

    def __init__(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int
    ):
        """
        Initialize detection result.

        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.face_landmarks = face_landmarks
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _get_landmarks(self, indices: List[int]) -> np.ndarray:
        """Get landmark coordinates for given indices."""
        points = []
        for idx in indices:
            landmark = self.face_landmarks.landmark[idx]
            x = landmark.x * self.frame_width
            y = landmark.y * self.frame_height
            points.append([x, y])
        return np.array(points, dtype=np.float32)

    def get_left_eye_landmarks(self) -> np.ndarray:
        """Get left eye contour landmarks."""
        return self._get_landmarks(FaceDetector.LEFT_EYE_INDICES)

    def get_right_eye_landmarks(self) -> np.ndarray:
        """Get right eye contour landmarks."""
        return self._get_landmarks(FaceDetector.RIGHT_EYE_INDICES)

    def get_left_iris_landmarks(self) -> np.ndarray:
        """Get left iris landmarks."""
        return self._get_landmarks(FaceDetector.LEFT_IRIS_INDICES)

    def get_right_iris_landmarks(self) -> np.ndarray:
        """Get right iris landmarks."""
        return self._get_landmarks(FaceDetector.RIGHT_IRIS_INDICES)

    def get_left_iris_center(self) -> np.ndarray:
        """Get left iris center (first point is center in MediaPipe)."""
        landmark = self.face_landmarks.landmark[FaceDetector.LEFT_IRIS_INDICES[0]]
        return np.array([
            landmark.x * self.frame_width,
            landmark.y * self.frame_height
        ], dtype=np.float32)

    def get_right_iris_center(self) -> np.ndarray:
        """Get right iris center (first point is center in MediaPipe)."""
        landmark = self.face_landmarks.landmark[FaceDetector.RIGHT_IRIS_INDICES[0]]
        return np.array([
            landmark.x * self.frame_width,
            landmark.y * self.frame_height
        ], dtype=np.float32)

    def get_gaze_point(self) -> np.ndarray:
        """Get average gaze point from both irises."""
        left_center = self.get_left_iris_center()
        right_center = self.get_right_iris_center()
        return (left_center + right_center) / 2.0

    def get_face_bbox(self) -> Tuple[int, int, int, int]:
        """
        Get face bounding box.

        Returns:
            (x, y, width, height) tuple
        """
        face_points = self._get_landmarks(FaceDetector.FACE_OVAL_INDICES)

        x_min = np.min(face_points[:, 0])
        y_min = np.min(face_points[:, 1])
        x_max = np.max(face_points[:, 0])
        y_max = np.max(face_points[:, 1])

        width = x_max - x_min
        height = y_max - y_min

        return (int(x_min), int(y_min), int(width), int(height))

    def get_normalized_gaze_point(self) -> np.ndarray:
        """
        Get gaze point normalized relative to face bbox.

        Returns:
            [x, y] in range [0, 1] relative to face
        """
        gaze = self.get_gaze_point()
        bbox = self.get_face_bbox()

        # Normalize relative to face bbox
        norm_x = (gaze[0] - bbox[0]) / bbox[2] if bbox[2] > 0 else 0.5
        norm_y = (gaze[1] - bbox[1]) / bbox[3] if bbox[3] > 0 else 0.5

        return np.array([norm_x, norm_y], dtype=np.float32)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        draw_iris: bool = True,
        draw_eyes: bool = True,
        draw_face: bool = False
    ) -> np.ndarray:
        """
        Draw detected landmarks on frame.

        Args:
            frame: Frame to draw on (will be modified)
            draw_iris: Draw iris landmarks
            draw_eyes: Draw eye contours
            draw_face: Draw face oval

        Returns:
            Frame with drawings
        """
        if draw_face:
            face_points = self._get_landmarks(FaceDetector.FACE_OVAL_INDICES)
            for point in face_points:
                cv2.circle(frame, tuple(point.astype(int)), 1, (128, 128, 128), -1)

        if draw_eyes:
            # Left eye
            left_eye = self.get_left_eye_landmarks()
            for i in range(len(left_eye)):
                start = tuple(left_eye[i].astype(int))
                end = tuple(left_eye[(i + 1) % len(left_eye)].astype(int))
                cv2.line(frame, start, end, (0, 255, 0), 1)

            # Right eye
            right_eye = self.get_right_eye_landmarks()
            for i in range(len(right_eye)):
                start = tuple(right_eye[i].astype(int))
                end = tuple(right_eye[(i + 1) % len(right_eye)].astype(int))
                cv2.line(frame, start, end, (0, 255, 0), 1)

        if draw_iris:
            # Left iris
            left_iris = self.get_left_iris_landmarks()
            left_center = left_iris[0]  # First point is center
            cv2.circle(frame, tuple(left_center.astype(int)), 3, (0, 255, 255), -1)

            # Right iris
            right_iris = self.get_right_iris_landmarks()
            right_center = right_iris[0]
            cv2.circle(frame, tuple(right_center.astype(int)), 3, (0, 255, 255), -1)

            # Gaze point
            gaze = self.get_gaze_point()
            cv2.circle(frame, tuple(gaze.astype(int)), 5, (255, 0, 255), 2)

        return frame

    def get_confidence(self) -> float:
        """
        Estimate detection confidence (0-1).

        Note: MediaPipe doesn't provide per-landmark confidence,
        so we use presence/visibility as proxy.
        """
        # Check if key landmarks have good visibility
        key_indices = [1, 33, 133, 362, 263]  # Nose, eye corners
        visibilities = []

        for idx in key_indices:
            landmark = self.face_landmarks.landmark[idx]
            if hasattr(landmark, 'visibility'):
                visibilities.append(landmark.visibility)

        if visibilities:
            return np.mean(visibilities)
        return 1.0  # Assume good if no visibility info
