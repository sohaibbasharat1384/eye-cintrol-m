"""Click detection and mouse control."""

import logging
import threading
import time
from typing import Optional, Tuple
from enum import Enum
import numpy as np
import pyautogui
from .utils import compute_eye_aspect_ratio, locked
from .detector import FaceDetectionResult

logger = logging.getLogger(__name__)


# Disable PyAutoGUI fail-safe for production use
# (Users can still use keyboard shortcuts to exit)
pyautogui.FAILSAFE = False

# PyAutoGUI sleeps for PAUSE seconds (default 0.1) after every call. The
# processing loop moves the cursor on every frame, so the default pause would
# cap the pipeline below 10 FPS.
pyautogui.PAUSE = 0


class ClickMode(Enum):
    """Click detection modes."""
    NONE = "none"
    BLINK = "blink"
    DWELL = "dwell"
    WINK = "wink"  # left wink = left click, right wink = right click


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
            logger.warning("Mouse move failed: %s", e)

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
            logger.warning("Mouse click failed: %s", e)

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size."""
        return pyautogui.size()


class BlinkDetector:
    """Detect deliberate (long) blinks for click activation.

    Natural, involuntary blinks last about 100-300 ms and happen many times
    a minute, so a click requires both eyes to stay closed for at least
    `min_duration`. Closures longer than `max_duration` (resting the eyes,
    looking down at the keyboard) are ignored. The click fires when the
    eyes reopen.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        min_duration: float = 0.4,
        max_duration: float = 2.0,
        debounce_time: float = 0.5
    ):
        """
        Initialize blink detector.

        Args:
            ear_threshold: EAR threshold below which eye is considered closed
            min_duration: Minimum closure in seconds to count as a click
            max_duration: Longer closures are ignored (seconds)
            debounce_time: Minimum time between clicks in seconds
        """
        self.ear_threshold = ear_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.debounce_time = debounce_time

        self._closed_since: Optional[float] = None
        self._last_blink_time = 0.0

    def detect(self, detection_result: FaceDetectionResult, now: Optional[float] = None) -> bool:
        """
        Detect a deliberate blink from a face detection result.

        Args:
            detection_result: Face detection result
            now: Current time in seconds (default: time.time(); for tests)

        Returns:
            True on the frame the eyes reopen after a long enough closure
        """
        current_time = time.time() if now is None else now

        left_ear = compute_eye_aspect_ratio(detection_result.get_left_eye_landmarks())
        right_ear = compute_eye_aspect_ratio(detection_result.get_right_eye_landmarks())

        # Both eyes must be closed; with an averaged EAR a single closed
        # eye (a wink, or squinting one eye) also counted as a blink
        if left_ear < self.ear_threshold and right_ear < self.ear_threshold:
            # Eyes closed: remember when the closure started
            if self._closed_since is None:
                self._closed_since = current_time
            return False

        if self._closed_since is None:
            return False

        # Eyes reopened: decide based on how long they were closed
        duration = current_time - self._closed_since
        self._closed_since = None

        if current_time - self._last_blink_time < self.debounce_time:
            return False

        if self.min_duration <= duration <= self.max_duration:
            self._last_blink_time = current_time
            return True

        return False

    def set_threshold(self, threshold: float):
        """Set EAR threshold."""
        self.ear_threshold = float(np.clip(threshold, 0.1, 0.4))

    def reset(self):
        """Reset detector state."""
        self._closed_since = None
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
    """Detect dwell (cursor staying in place) for click activation.

    After a click the detector is disarmed until the cursor leaves the
    clicked spot (moves farther than dwell_radius from it); otherwise
    simply keeping the gaze on one spot, e.g. while reading, would click
    again and again.
    """

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
        # Where the last dwell click happened; set until the cursor leaves it
        self._last_click_position: Optional[Tuple[int, int]] = None

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def update(
        self,
        cursor_position: Tuple[int, int],
        now: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Update dwell detector with cursor position.

        Args:
            cursor_position: (x, y) screen coordinates
            now: Current time in seconds (default: time.time(); for tests)

        Returns:
            (click_triggered, progress) tuple where progress is 0-1
        """
        current_time = time.time() if now is None else now

        # Check debounce
        if current_time - self._last_click_time < self._debounce_time:
            return (False, 0.0)

        # Re-arm only after the cursor has left the spot clicked last
        if self._last_click_position is not None:
            if self._distance(cursor_position, self._last_click_position) <= self.dwell_radius:
                return (False, 0.0)
            self._last_click_position = None

        if self._dwell_position is None:
            # Start new dwell
            self._dwell_position = cursor_position
            self._dwell_start_time = current_time
            return (False, 0.0)

        # Check if cursor moved too far
        if self._distance(cursor_position, self._dwell_position) > self.dwell_radius:
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
            self._last_click_position = self._dwell_position
            self._dwell_position = None
            self._dwell_start_time = None
            return (True, 1.0)

        return (False, progress)

    def reset(self):
        """Reset detector state."""
        self._dwell_start_time = None
        self._dwell_position = None
        self._last_click_position = None

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

        # update() runs in the processing thread; mode and settings are
        # changed from the GUI thread
        self._lock = threading.RLock()

    @locked
    def set_mode(self, mode: ClickMode):
        """Set click detection mode."""
        self.mode = mode

        # Reset all detectors
        self.blink_detector.reset()
        self.wink_detector.reset()
        self.dwell_detector.reset()

    @locked
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

            elif self.mode == ClickMode.WINK and detection_result is not None:
                wink = self.wink_detector.detect(detection_result)
                if wink in ('left', 'right'):
                    self.mouse.click(wink)

            elif self.mode == ClickMode.DWELL and cursor_position is not None:
                triggered, _ = self.dwell_detector.update(cursor_position)
                if triggered:
                    self.mouse.click('left')

        except Exception as e:
            logger.warning("Click detection error: %s", e)

    @locked
    def any_eye_closed(self, detection_result: FaceDetectionResult) -> bool:
        """Whether at least one eye is below the blink EAR threshold."""
        threshold = self.blink_detector.ear_threshold
        return (
            compute_eye_aspect_ratio(detection_result.get_left_eye_landmarks()) < threshold
            or compute_eye_aspect_ratio(detection_result.get_right_eye_landmarks()) < threshold
        )

    @locked
    def get_dwell_progress(self) -> Optional[Tuple[Tuple[int, int], float]]:
        """Get current dwell progress for visualization."""
        if self.mode == ClickMode.DWELL:
            return self.dwell_detector.get_dwell_info()
        return None

    @locked
    def configure_blink(self, threshold: float):
        """Configure blink detection threshold."""
        self.blink_detector.set_threshold(threshold)

    @locked
    def configure_dwell(self, dwell_time: float, radius: int):
        """Configure dwell detection parameters."""
        self.dwell_detector.dwell_time = max(0.3, dwell_time)
        self.dwell_detector.dwell_radius = max(10, radius)
