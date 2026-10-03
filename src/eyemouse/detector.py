"""Eye and iris detection using the MediaPipe Face Landmarker task."""

import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

logger = logging.getLogger(__name__)

MODEL_FILENAME = "face_landmarker.task"
# Pinned model version (float16, v1) from Google's MediaPipe model storage
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
# Environment variable that can point to a local copy of the model
MODEL_PATH_ENV = "EYEMOUSE_FACE_MODEL"


def _user_cache_dir() -> Path:
    """Per-user cache directory for downloaded model files."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "eyemouse"


def get_model_path() -> str:
    """
    Locate the Face Landmarker model, downloading it on first use.

    Lookup order:
        1. The file named by the EYEMOUSE_FACE_MODEL environment variable
        2. A copy shipped inside the package (eyemouse/models/)
        3. The per-user cache directory, downloading the model there if
           it is not present yet

    Returns:
        Path to the .task model file
    """
    env_path = os.environ.get(MODEL_PATH_ENV)
    if env_path:
        if not Path(env_path).is_file():
            raise FileNotFoundError(f"{MODEL_PATH_ENV} points to a missing file: {env_path}")
        return env_path

    packaged = Path(__file__).resolve().parent / "models" / MODEL_FILENAME
    if packaged.is_file():
        return str(packaged)

    cached = _user_cache_dir() / MODEL_FILENAME
    if not cached.is_file():
        _download_model(cached)
    return str(cached)


def _download_model(destination: Path):
    """Download the model to destination atomically (temp file + rename)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".download")
    logger.info("Downloading face landmark model to %s", destination)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, open(tmp_path, "wb") as f:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if tmp_path.stat().st_size == 0:
            raise IOError("downloaded model file is empty")
        tmp_path.replace(destination)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the face landmark model from {MODEL_URL}: {e}. "
            f"Download it manually and set {MODEL_PATH_ENV} to its path, "
            f"or place it at {destination}."
        ) from e


class FaceDetector:
    """Face and iris detection using the MediaPipe Face Landmarker task."""

    # Face mesh landmark indices (the 478-point model includes the irises)
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
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None
    ):
        """
        Initialize the MediaPipe Face Landmarker.

        Args:
            static_image_mode: Treat every frame as an unrelated image
                (no tracking between frames)
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Kept for backward compatibility. The Face
                Landmarker model always outputs the refined 478 landmarks,
                including the irises, so this has no effect.
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_path: Path to face_landmarker.task (default: see
                get_model_path())
        """
        self._running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path or get_model_path()),
            running_mode=self._running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # VIDEO mode requires strictly increasing timestamps
        self._last_timestamp_ms = -1

        self.frame_width = 0
        self.frame_height = 0

    def process(self, frame: np.ndarray) -> Optional['FaceDetectionResult']:
        """
        Process frame and detect face landmarks.

        Args:
            frame: BGR image as delivered by OpenCV (or grayscale)

        Returns:
            FaceDetectionResult or None if no face detected
        """
        self.frame_height, self.frame_width = frame.shape[:2]

        # The model expects RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))

        if self._running_mode == RunningMode.VIDEO:
            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
            results = self.landmarker.detect_for_video(image, timestamp_ms)
        else:
            results = self.landmarker.detect(image)

        if not results.face_landmarks:
            return None

        # Use first face
        face_landmarks = results.face_landmarks[0]

        return FaceDetectionResult(
            face_landmarks=face_landmarks,
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )

    def close(self):
        """Release resources."""
        self.landmarker.close()


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
            face_landmarks: Sequence of normalized landmarks for one face
                (objects with x and y in the 0-1 range)
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
            landmark = self.face_landmarks[idx]
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
        landmark = self.face_landmarks[FaceDetector.LEFT_IRIS_INDICES[0]]
        return np.array([
            landmark.x * self.frame_width,
            landmark.y * self.frame_height
        ], dtype=np.float32)

    def get_right_iris_center(self) -> np.ndarray:
        """Get right iris center (first point is center in MediaPipe)."""
        landmark = self.face_landmarks[FaceDetector.RIGHT_IRIS_INDICES[0]]
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

    def get_confidence(self) -> Optional[float]:
        """
        Detection confidence (0-1), if the model provides one.

        The MediaPipe face models do not output a per-face score, and their
        per-landmark presence/visibility fields are left unset (None, or 0.0
        with the legacy FaceMesh). Reporting a made-up number would be
        misleading, so None is returned in that case; callers should then
        rely on whether a face was detected at all.

        Returns:
            Mean presence/visibility of key landmarks, or None if unavailable
        """
        key_indices = [1, 33, 133, 362, 263]  # Nose, eye corners
        scores = []

        for idx in key_indices:
            landmark = self.face_landmarks[idx]
            for field in ("presence", "visibility"):
                value = getattr(landmark, field, None)
                if value:  # None and 0.0 both mean "not populated"
                    scores.append(float(value))
                    break

        if scores:
            return float(np.mean(scores))
        return None
