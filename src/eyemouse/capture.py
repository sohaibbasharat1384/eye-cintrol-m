"""Camera capture module with threading."""

import logging
import sys
import time
import threading
import queue
from typing import List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def candidate_backends() -> List[int]:
    """
    OpenCV capture backends to try, in order of preference.

    On Windows OpenCV defaults to Media Foundation (MSMF), which with many
    webcams delivers only a few frames per second and takes seconds to
    start; DirectShow typically gives the camera's full rate. Measured on
    the development machine (scripts/benchmark_camera.py, 640x480):
    MSMF 0.9 FPS / 4 s startup, DirectShow 30.6 FPS / 1.1 s. DirectShow is
    tried first, with OpenCV's default as fallback.
    """
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


class CameraCapture:
    """Threaded camera capture for low-latency frame acquisition."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 2,
        backend: Optional[int] = None
    ):
        """
        Initialize camera capture.

        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Target FPS
            buffer_size: Frame queue size (small for low latency)
            backend: OpenCV backend (cv2.CAP_*) to use; by default the
                ones from candidate_backends() are tried in order
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.backend = backend

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_frame: Optional[Tuple[np.ndarray, float]] = None

    def start(self) -> bool:
        """
        Start camera capture thread.

        Returns:
            True if successful, False otherwise
        """
        if self.running:
            return True

        backends = [self.backend] if self.backend is not None else candidate_backends()
        for backend in backends:
            self.cap = self._open(backend)
            if self.cap is not None:
                break
        else:
            return False

        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Camera %d initialized: %dx%d via %s",
            self.camera_id, actual_width, actual_height, self.cap.getBackendName()
        )

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        return True

    def _open(self, backend: int) -> Optional[cv2.VideoCapture]:
        """Open and configure the camera with one backend.

        Returns the capture only if it opened *and* delivered a frame, so a
        backend that opens but cannot stream falls through to the next one.
        """
        cap = cv2.VideoCapture(self.camera_id, backend)
        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

        ok, _ = cap.read()
        if not ok:
            logger.debug("Camera %d opened via %s but delivered no frame",
                         self.camera_id, cap.getBackendName())
            cap.release()
            return None
        return cap

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            timestamp = time.time()

            # Flip horizontally for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)

            # Update latest frame
            with self._lock:
                self._latest_frame = (frame, timestamp)

            # Try to put in queue (non-blocking)
            try:
                self.frame_queue.put_nowait((frame, timestamp))
            except queue.Full:
                # Discard oldest frame
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame, timestamp))
                except queue.Empty:
                    pass

    def read(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """
        Read latest frame from queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            (frame, timestamp) tuple or None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_latest(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Read the most recent frame (non-blocking).

        Returns:
            (frame, timestamp) tuple or None
        """
        with self._lock:
            return self._latest_frame

    def stop(self):
        """Stop camera capture and release resources."""
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def get_resolution(self) -> Tuple[int, int]:
        """Get actual camera resolution."""
        if self.cap is None or not self.cap.isOpened():
            return (self.width, self.height)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def list_cameras(max_cameras: int = 5) -> list:
    """
    List available camera devices.

    Args:
        max_cameras: Maximum number of cameras to check

    Returns:
        List of available camera IDs
    """
    available = []
    backend = candidate_backends()[0]

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            available.append(i)
        cap.release()

    return available
