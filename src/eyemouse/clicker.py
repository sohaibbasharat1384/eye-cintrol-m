"""Click detection and mouse control."""

import time
from typing import Optional, Tuple
from enum import Enum
import numpy as np
import pyautogui
from .utils import compute_eye_aspect_ratio
from .detector import FaceDetectionResult


# Disable PyAutoGUI fail-safe for production use
# (Users can still use keyboard shortcuts to exit)
pyautogui.FAILSAFE = False


class ClickMode(Enum):
    """Click detection modes."""
    NONE = "none"
    BLINK = "blink"
    DWELL = "dwell"
    WINK_LEFT = "wink_left"
    WINK_RIGHT = "wink_right"


class MouseController:
    """Control mouse cursor and clicks."""

    def __init__(self, enabled: bool = False):
        """
        Initialize mouse controller.

        Args:
            enabled: Whether mouse control is enabled
        """
        self.enabled = enabled
        self._last_position: Optional[Tuple[int, int]] = None

    def set_enabled(self, enabled: bool):
        """Enable or disable mouse control."""
        self.enabled = enabled

    def move_to(self, x: int, y: int, smooth: bool = False):
        """
        Move mouse cursor to position.

        Args:
            x: Screen x coordinate
            y: Screen y coordinate
            smooth: Use smooth movement (slower)
        """
        if not self.enabled:
            return

        try:
            if smooth and self._last_position is not None:
                # Smooth movement
                duration = 0.05  # 50ms
                pyautogui.moveTo(x, y, duration=duration)
            else:
                # Instant movement
                pyautogui.moveTo(x, y)

            self._last_position = (x, y)

        except Exception as e:
            print(f"Mouse move failed: {e}")

    def click(self, button: str = 'left'):
        """
        Perform mouse click.

        Args:
            button: 'left', 'right', or 'middle'
        """
        if not self.enabled:
            return

        try:
            pyautogui.click(button=button)
        except Exception as e:
            print(f"Mouse click failed: {e}")

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size."""
        return pyautogui.size()


class BlinkDetector:
    """Detect eye blinks for click activation."""

    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 2,
        debounce_time: float = 0.5
    ):
        """
        Initialize blink detector.

        Args:
            ear_threshold: EAR threshold below which eye is considered closed
            consecutive_frames: Number of consecutive frames needed to confirm blink
            debounce_time: Minimum time between blinks in seconds
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.debounce_time = debounce_time

        self._blink_counter = 0
        self._last_blink_time = 0.0

    def detect(self, detection_result: FaceDetectionResult) -> bool:
        """
        Detect blink from face detection result.

        Args:
            detection_result: Face detection result

        Returns:
            True if blink detected
        """
        current_time = time.time()

        # Check debounce
        if current_time - self._last_blink_time < self.debounce_time:
            return False

        # Compute EAR for both eyes
        left_eye = detection_result.get_left_eye_landmarks()
        right_eye = detection_result.get_right_eye_landmarks()

        left_ear = compute_eye_aspect_ratio(left_eye)
        right_ear = compute_eye_aspect_ratio(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0

        # Check if eyes are closed
        if avg_ear < self.ear_threshold:
            self._blink_counter += 1
        else:
            if self._blink_counter >= self.consecutive_frames:
                # Blink detected!
                self._last_blink_time = current_time
                self._blink_counter = 0
                return True

            self._blink_counter = 0

        return False

    def set_threshold(self, threshold: float):
        """Set EAR threshold."""
        self.ear_threshold = np.clip(threshold, 0.1, 0.4)

    def reset(self):
        """Reset detector state."""
        self._blink_counter = 0
        self._last_blink_time = 0.0


class WinkDetector:
    """Detect left/right winks for click activation."""

    def __init__(
        self,
        ear_threshold: float = 0.21,
        ear_diff_threshold: float = 0.1,
        consecutive_frames: int = 3,
        debounce_time: float = 0.6
    ):
        """
        Initialize wink detector.

        Args:
            ear_threshold: EAR threshold below which eye is considered closed
            ear_diff_threshold: Minimum difference between left/right EAR for wink
            consecutive_frames: Number of consecutive frames needed
            debounce_time: Minimum time between winks in seconds
        """
        self.ear_threshold = ear_threshold
        self.ear_diff_threshold = ear_diff_threshold
        self.consecutive_frames = consecutive_frames
        self.debounce_time = debounce_time

        self._left_wink_counter = 0
        self._right_wink_counter = 0
        self._last_wink_time = 0.0

    def detect(self, detection_result: FaceDetectionResult) -> Optional[str]:
        """
        Detect wink from face detection result.

        Args:
            detection_result: Face detection result

        Returns:
            'left' or 'right' if wink detected, None otherwise
        """
        current_time = time.time()

        # Check debounce
        if current_time - self._last_wink_time < self.debounce_time:
            return None

        # Compute EAR for both eyes
        left_eye = detection_result.get_left_eye_landmarks()
        right_eye = detection_result.get_right_eye_landmarks()

        left_ear = compute_eye_aspect_ratio(left_eye)
        right_ear = compute_eye_aspect_ratio(right_eye)

        ear_diff = abs(left_ear - right_ear)

        # Check for left wink (left eye closed, right eye open)
        if left_ear < self.ear_threshold and right_ear > self.ear_threshold + self.ear_diff_threshold:
            self._left_wink_counter += 1
            self._right_wink_counter = 0
        # Check for right wink (right eye closed, left eye open)
        elif right_ear < self.ear_threshold and left_ear > self.ear_threshold + self.ear_diff_threshold:
            self._right_wink_counter += 1
            self._left_wink_counter = 0
        else:
            # Check if wink was completed
            if self._left_wink_counter >= self.consecutive_frames:
                self._last_wink_time = current_time
                self._left_wink_counter = 0
                return 'left'
            elif self._right_wink_counter >= self.consecutive_frames:
                self._last_wink_time = current_time
                self._right_wink_counter = 0
                return 'right'

            self._left_wink_counter = 0
            self._right_wink_counter = 0

        return None

    def reset(self):
        """Reset detector state."""
        self._left_wink_counter = 0
        self._right_wink_counter = 0
        self._last_wink_time = 0.0


class DwellDetector:
    """Detect dwell (cursor staying in place) for click activation."""

    def __init__(
        self,
        dwell_time: float = 0.6,
        dwell_radius: int = 30
    ):
        """
        Initialize dwell detector.

        Args:
            dwell_time: Time in seconds cursor must stay in place
            dwell_radius: Maximum movement radius in pixels
        """
        self.dwell_time = dwell_time
        self.dwell_radius = dwell_radius

        self._dwell_start_time: Optional[float] = None
        self._dwell_position: Optional[Tuple[int, int]] = None
        self._last_click_time = 0.0
        self._debounce_time = 0.5

    def update(self, cursor_position: Tuple[int, int]) -> Tuple[bool, float]:
        """
        Update dwell detector with cursor position.

        Args:
            cursor_position: (x, y) screen coordinates

        Returns:
            (click_triggered, progress) tuple where progress is 0-1
        """
        current_time = time.time()

        # Check debounce
        if current_time - self._last_click_time < self._debounce_time:
            return (False, 0.0)

        if self._dwell_position is None:
            # Start new dwell
            self._dwell_position = cursor_position
            self._dwell_start_time = current_time
            return (False, 0.0)

        # Check if cursor moved too far
        distance = np.sqrt(
            (cursor_position[0] - self._dwell_position[0]) ** 2 +
            (cursor_position[1] - self._dwell_position[1]) ** 2
        )

        if distance > self.dwell_radius:
            # Reset dwell
            self._dwell_position = cursor_position
            self._dwell_start_time = current_time
            return (False, 0.0)

        # Check dwell time
        elapsed = current_time - self._dwell_start_time
        progress = min(elapsed / self.dwell_time, 1.0)

        if elapsed >= self.dwell_time:
            # Dwell complete - trigger click
            self._last_click_time = current_time
            self._dwell_position = None
            self._dwell_start_time = None
            return (True, 1.0)

        return (False, progress)

    def reset(self):
        """Reset detector state."""
        self._dwell_start_time = None
        self._dwell_position = None

    def get_dwell_info(self) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Get current dwell information.

        Returns:
            ((x, y), progress) tuple or None
        """
        if self._dwell_position is None or self._dwell_start_time is None:
            return None

        elapsed = time.time() - self._dwell_start_time
        progress = min(elapsed / self.dwell_time, 1.0)

        return (self._dwell_position, progress)


class ClickManager:
    """Manage click detection and execution."""

    def __init__(self, mouse_controller: MouseController):
        """
        Initialize click manager.

        Args:
            mouse_controller: Mouse controller instance
        """
        self.mouse = mouse_controller
        self.mode = ClickMode.NONE

        self.blink_detector = BlinkDetector()
        self.wink_detector = WinkDetector()
        self.dwell_detector = DwellDetector()

    def set_mode(self, mode: ClickMode):
        """Set click detection mode."""
        self.mode = mode

        # Reset all detectors
        self.blink_detector.reset()
        self.wink_detector.reset()
        self.dwell_detector.reset()

    def update(
        self,
        detection_result: Optional[FaceDetectionResult],
        cursor_position: Optional[Tuple[int, int]]
    ):
        """
        Update click detection and perform click if triggered.

        Args:
            detection_result: Face detection result (for blink/wink)
            cursor_position: Current cursor position (for dwell)
        """
        if self.mode == ClickMode.NONE:
            return

        try:
            if self.mode == ClickMode.BLINK and detection_result is not None:
                if self.blink_detector.detect(detection_result):
                    self.mouse.click('left')

            elif self.mode == ClickMode.WINK_LEFT and detection_result is not None:
                wink = self.wink_detector.detect(detection_result)
                if wink == 'left':
                    self.mouse.click('left')
                elif wink == 'right':
                    self.mouse.click('right')

            elif self.mode == ClickMode.WINK_RIGHT and detection_result is not None:
                wink = self.wink_detector.detect(detection_result)
                if wink == 'right':
                    self.mouse.click('right')
                elif wink == 'left':
                    self.mouse.click('left')

            elif self.mode == ClickMode.DWELL and cursor_position is not None:
                triggered, _ = self.dwell_detector.update(cursor_position)
                if triggered:
                    self.mouse.click('left')

        except Exception as e:
            print(f"Click detection error: {e}")

    def get_dwell_progress(self) -> Optional[Tuple[Tuple[int, int], float]]:
        """Get current dwell progress for visualization."""
        if self.mode == ClickMode.DWELL:
            return self.dwell_detector.get_dwell_info()
        return None

    def configure_blink(self, threshold: float):
        """Configure blink detection threshold."""
        self.blink_detector.set_threshold(threshold)

    def configure_dwell(self, dwell_time: float, radius: int):
        """Configure dwell detection parameters."""
        self.dwell_detector.dwell_time = max(0.3, dwell_time)
        self.dwell_detector.dwell_radius = max(10, radius)
