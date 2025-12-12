"""Camera capture module with threading."""

import time
import threading
import queue
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraCapture:
    """Threaded camera capture for low-latency frame acquisition."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 2
    ):
        """
        Initialize camera capture.

        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Target FPS
            buffer_size: Frame queue size (small for low latency)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

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

        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            return False

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera initialized: {actual_width}x{actual_height}")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        return True

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

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()

    return available
