"""PyQt6 GUI for eye control mouse application."""

import logging
import time
import sys
from typing import Optional, Tuple
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QGroupBox, QMessageBox,
    QProgressBar, QTabWidget, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QPoint
from PyQt6.QtGui import (
    QImage, QPixmap, QKeySequence, QShortcut, QPainter, QColor, QPen, QFont, QGuiApplication
)

from .capture import CameraCapture
from .detector import FaceDetector
from .tracker import GazeTracker
from .calibration import CalibrationManager
from .clicker import MouseController, ClickManager, ClickMode
from .utils import PerformanceMetrics, format_error_px

logger = logging.getLogger(__name__)


class ProcessingThread(QThread):
    """Background thread for video processing."""

    frame_processed = pyqtSignal(np.ndarray, dict)  # frame, info
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        camera: CameraCapture,
        detector: FaceDetector,
        tracker: GazeTracker,
        click_manager: ClickManager
    ):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.click_manager = click_manager

        self.running = False
        self.tracking_enabled = False
        self.calibration_manager: Optional[CalibrationManager] = None

        self.metrics = PerformanceMetrics()

    def run(self):
        """Main processing loop."""
        self.running = True
        last_time = time.time()

        while self.running:
            try:
                # Read frame
                result = self.camera.read(timeout=0.1)
                if result is None:
                    continue

                frame, timestamp = result
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                # Process frame
                detection_result = self.detector.process(frame)

                info = {
                    'face_detected': detection_result is not None,
                    'cursor_position': None,
                    'confidence': None,
                    'calibrating': False,
                    'calibration_target': None,
                    'calibration_progress': (0, 0),
                    'dwell_info': None
                }

                calibrating = (
                    self.calibration_manager is not None
                    and self.calibration_manager.is_calibrating()
                )
                if calibrating:
                    # Reported even without a face so the overlay keeps
                    # showing the target and can tell the user what is wrong
                    info['calibrating'] = True
                    info['calibration_target'] = self.calibration_manager.get_current_target_point()
                    info['calibration_progress'] = self.calibration_manager.get_progress()

                if detection_result is not None:
                    info['confidence'] = detection_result.get_confidence()

                    # Handle calibration
                    if calibrating:
                        # Add gaze samples
                        gaze_point = detection_result.get_normalized_gaze_point()
                        self.calibration_manager.add_gaze_sample(gaze_point)

                        # توجه: تأیید نقطه (confirm_current_point)
                        # فقط باید با دکمه‌ی Space / Confirm در GUI انجام شود،
                        # نه خودکار داخل thread.

                        # Draw on frame
                        frame = detection_result.draw_landmarks(frame)


                    # Handle normal tracking
                    elif self.tracking_enabled:
                        cursor_pos = self.tracker.process(detection_result, dt=dt)

                        if cursor_pos is not None:
                            info['cursor_position'] = cursor_pos

                            # Move the system cursor before click detection so
                            # a triggered click lands at the current gaze point
                            self.click_manager.mouse.move_to(*cursor_pos)

                            # Update click detection
                            self.click_manager.update(detection_result, cursor_pos)

                            # Get dwell info for visualization
                            info['dwell_info'] = self.click_manager.get_dwell_progress()

                        # Draw on frame
                        frame = detection_result.draw_landmarks(frame)

                # Update metrics
                latency = time.time() - timestamp
                self.metrics.add_frame(current_time, latency)

                info['fps'] = self.metrics.get_fps()
                info['latency'] = self.metrics.get_avg_latency()

                # Emit processed frame
                self.frame_processed.emit(frame, info)

            except Exception as e:
                self.error_occurred.emit(str(e))
                time.sleep(0.1)

    def stop(self):
        """Stop processing thread."""
        self.running = False


class CalibrationOverlay(QWidget):
    """Fullscreen overlay for calibration targets.

    The overlay is a separate top-level window, so the main window's
    shortcuts do not reach it; it handles Space/Escape itself and reports
    them through signals.
    """

    confirm_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, screen_size: Tuple[int, int], parent=None):
        """
        Args:
            screen_size: (width, height) of the coordinate space the
                calibration targets are expressed in, i.e. the space used
                by PyAutoGUI for cursor moves
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.screen_size = screen_size
        self.target_point = None
        self.progress_text = ""
        self.status_text = ""
        self.is_stable = False
        self.auto_confirm = False
        self._closing_programmatically = False

        # Frameless, always on top, covering the primary screen
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def show_on_primary_screen(self):
        """Show fullscreen on the primary screen and take keyboard focus."""
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def set_target(self, x: int, y: int, progress_text: str, is_stable: bool = False):
        """Set the current calibration target point."""
        self.target_point = (x, y)
        self.progress_text = progress_text
        self.is_stable = is_stable
        self.update()

    def set_status(self, text: str):
        """Show a short status/warning line under the instructions."""
        self.status_text = text
        self.update()

    def close_overlay(self):
        """Close without emitting cancel_requested."""
        self._closing_programmatically = True
        self.close()

    def _to_widget_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert a target from screen (PyAutoGUI) space to widget pixels.

        On Windows with display scaling PyAutoGUI reports physical pixels
        while Qt paints in logical (device independent) pixels; on macOS
        both are logical. Scaling by the ratio of the two sizes handles
        every case without platform checks.
        """
        screen_w, screen_h = self.screen_size
        if screen_w <= 0 or screen_h <= 0:
            return x, y
        return (
            int(round(x * self.width() / screen_w)),
            int(round(y * self.height() / screen_h)),
        )

    def keyPressEvent(self, event):
        """Handle Space (confirm) and Escape (cancel)."""
        if event.key() == Qt.Key.Key_Space:
            self.confirm_requested.emit()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_requested.emit()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Treat closing the overlay by the user (e.g. Alt+F4) as cancel."""
        if not self._closing_programmatically:
            self.cancel_requested.emit()
        super().closeEvent(event)

    def paintEvent(self, event):
        """Draw the calibration target."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent dark background. Style sheets are not painted on
        # translucent plain QWidgets, so the fill has to be done here.
        painter.fillRect(self.rect(), QColor(0, 0, 0, 200))

        if self.target_point:
            x, y = self._to_widget_coords(*self.target_point)

            # Outer ring turns green once the gaze samples are stable
            ring_color = QColor(0, 200, 0) if self.is_stable else QColor(255, 0, 0)

            # Draw target circles (bullseye pattern)
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(ring_color)
            painter.drawEllipse(QPoint(x, y), 25, 25)

            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(QPoint(x, y), 15, 15)

            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(QPoint(x, y), 5, 5)

            # Draw crosshair
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(x - 35, y, x - 28, y)
            painter.drawLine(x + 28, y, x + 35, y)
            painter.drawLine(x, y - 35, x, y - 28)
            painter.drawLine(x, y + 28, x, y + 35)

        # Draw instruction text at bottom
        if self.progress_text:
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Arial", 16, QFont.Weight.Bold)
            painter.setFont(font)
            text_rect = self.rect()
            text_rect.setTop(self.height() - 160)
            if self.auto_confirm:
                action = "Keep looking - confirmed automatically when the ring turns green"
            else:
                action = "Press SPACE when the ring turns green"
            text = (
                f"{self.progress_text}\n\n"
                "Look at the center of the target\n"
                f"{action}  -  ESC to cancel"
            )
            if self.status_text:
                text += f"\n{self.status_text}"
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)


class DwellIndicator(QWidget):
    """Small ring drawn on screen around the dwell point, filling up as the
    dwell click approaches.

    The window is transparent for mouse input, so the click triggered by
    the dwell reaches the application underneath, and it never takes focus.
    """

    SIZE = 64          # widget size in logical pixels
    RADIUS = 24        # ring radius
    MIN_PROGRESS = 0.15  # hide very short dwells caused by normal jitter

    def __init__(self, screen_size: Tuple[int, int]):
        """
        Args:
            screen_size: (width, height) of the PyAutoGUI coordinate space
                the dwell positions are expressed in
        """
        super().__init__(None)
        self.screen_size = screen_size
        self.progress = 0.0

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(self.SIZE, self.SIZE)

    def _to_logical(self, x: int, y: int) -> Tuple[int, int]:
        """Convert PyAutoGUI screen coordinates to Qt logical coordinates
        on the primary screen (see CalibrationOverlay._to_widget_coords)."""
        screen = QGuiApplication.primaryScreen()
        screen_w, screen_h = self.screen_size
        if screen is None or screen_w <= 0 or screen_h <= 0:
            return x, y
        geometry = screen.geometry()
        return (
            geometry.x() + int(round(x * geometry.width() / screen_w)),
            geometry.y() + int(round(y * geometry.height() / screen_h)),
        )

    def update_dwell(self, dwell_info):
        """Show/move/hide the ring for ClickManager.get_dwell_progress()."""
        if dwell_info is None or dwell_info[1] < self.MIN_PROGRESS:
            self.hide()
            return

        (x, y), progress = dwell_info
        cx, cy = self._to_logical(x, y)
        self.move(cx - self.SIZE // 2, cy - self.SIZE // 2)
        self.progress = progress
        if not self.isVisible():
            self.show()
        self.update()

    def paintEvent(self, event):
        """Draw a faint full ring plus the progress arc."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPoint(self.SIZE // 2, self.SIZE // 2)
        r = self.RADIUS

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 90), 7))
        painter.drawEllipse(center, r, r)

        # Arc from 12 o'clock, clockwise (Qt angles are in 1/16 degree)
        painter.setPen(QPen(QColor(0, 200, 0), 5))
        painter.drawArc(
            center.x() - r, center.y() - r, 2 * r, 2 * r,
            90 * 16, -int(360 * 16 * self.progress)
        )


class EyeMouseGUI(QMainWindow):
    """Main application window."""

    # With auto-confirm, samples collected during this time after a new
    # target appears are discarded, so the eyes have moved to the target
    # before stability is evaluated (saccade reaction time is ~200-300 ms)
    AUTO_CONFIRM_SETTLE_TIME = 0.8  # seconds

    def __init__(self, camera_id: int = 0):
        super().__init__()

        self.camera_id = camera_id

        self.setWindowTitle("Eye Control Mouse")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize components
        self.camera: Optional[CameraCapture] = None
        self.detector: Optional[FaceDetector] = None
        self.tracker: Optional[GazeTracker] = None
        self.mouse_controller: Optional[MouseController] = None
        self.click_manager: Optional[ClickManager] = None
        self.calibration_manager: Optional[CalibrationManager] = None
        self.processing_thread: Optional[ProcessingThread] = None
        self.calibration_overlay: Optional[CalibrationOverlay] = None
        self.dwell_indicator: Optional[DwellIndicator] = None

        self.is_initialized = False
        self.is_tracking = False

        # Auto-confirm state for the current calibration point
        self._point_started_at = 0.0
        self._point_settled = False

        # Setup UI
        self._setup_ui()
        self._setup_shortcuts()

        # Initialize system
        QTimer.singleShot(100, self._initialize_system)

    def _setup_ui(self):
        """Setup user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left side: Video preview
        left_panel = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(self.video_label)

        # Status bar
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        left_panel.addWidget(self.status_label)

        # Metrics
        metrics_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.latency_label = QLabel("Latency: -- ms")
        self.confidence_label = QLabel("Face: --")
        metrics_layout.addWidget(self.fps_label)
        metrics_layout.addWidget(self.latency_label)
        metrics_layout.addWidget(self.confidence_label)
        metrics_layout.addStretch()
        left_panel.addLayout(metrics_layout)

        main_layout.addLayout(left_panel, stretch=2)

        # Right side: Controls
        right_panel = QVBoxLayout()

        # Tracking controls
        tracking_group = self._create_tracking_group()
        right_panel.addWidget(tracking_group)

        # Calibration controls
        calibration_group = self._create_calibration_group()
        right_panel.addWidget(calibration_group)

        # Settings
        settings_group = self._create_settings_group()
        right_panel.addWidget(settings_group)

        right_panel.addStretch()

        main_layout.addLayout(right_panel, stretch=1)

    def _create_tracking_group(self) -> QGroupBox:
        """Create tracking controls group."""
        group = QGroupBox("Tracking Control")
        layout = QVBoxLayout()

        # Camera selection
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        num_cameras = max(4, self.camera_id + 1)
        self.camera_combo.addItems([f"Camera {i}" for i in range(num_cameras)])
        self.camera_combo.setCurrentIndex(self.camera_id)
        # Connected after selecting the initial camera so that does not
        # trigger a camera switch
        self.camera_combo.currentIndexChanged.connect(self._change_camera)
        camera_layout.addWidget(self.camera_combo)
        layout.addLayout(camera_layout)

        self.tracking_button = QPushButton("Enable Tracking")
        self.tracking_button.setCheckable(True)
        self.tracking_button.clicked.connect(self._toggle_tracking)
        self.tracking_button.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """)
        layout.addWidget(self.tracking_button)

        # Click mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Click Mode:"))
        self.click_mode_combo = QComboBox()
        self.click_mode_combo.addItems(["None", "Blink", "Dwell", "Wink (L/R)"])
        self.click_mode_combo.currentIndexChanged.connect(self._change_click_mode)
        mode_layout.addWidget(self.click_mode_combo)
        layout.addLayout(mode_layout)

        group.setLayout(layout)
        return group

    def _create_calibration_group(self) -> QGroupBox:
        """Create calibration controls group."""
        group = QGroupBox("Calibration")
        layout = QVBoxLayout()

        self.calibrate_button = QPushButton("Start Calibration")
        self.calibrate_button.clicked.connect(self._start_calibration)
        layout.addWidget(self.calibrate_button)

        # Optional: confirm points automatically instead of pressing Space
        self.auto_confirm_checkbox = QCheckBox("Auto-confirm when gaze is stable")
        self.auto_confirm_checkbox.setToolTip(
            "Confirm each calibration point automatically once your gaze has been\n"
            "stable on it, instead of pressing Space. Off by default."
        )
        layout.addWidget(self.auto_confirm_checkbox)

        self.calibration_progress = QProgressBar()
        self.calibration_progress.setVisible(False)
        layout.addWidget(self.calibration_progress)

        self.calibration_label = QLabel("")
        self.calibration_label.setWordWrap(True)
        layout.addWidget(self.calibration_label)

        # Confirm button (hidden initially)
        self.confirm_button = QPushButton("Confirm Point (Space)")
        self.confirm_button.clicked.connect(self._confirm_calibration_point)
        self.confirm_button.setVisible(False)
        layout.addWidget(self.confirm_button)

        # Load/Save buttons
        buttons_layout = QHBoxLayout()
        self.save_cal_button = QPushButton("Save")
        self.save_cal_button.clicked.connect(self._save_calibration)
        self.load_cal_button = QPushButton("Load")
        self.load_cal_button.clicked.connect(self._load_calibration)
        buttons_layout.addWidget(self.save_cal_button)
        buttons_layout.addWidget(self.load_cal_button)
        layout.addLayout(buttons_layout)

        group.setLayout(layout)
        return group

    def _create_settings_group(self) -> QGroupBox:
        """Create settings controls group."""
        group = QGroupBox("Settings")
        layout = QVBoxLayout()

        # Smoothing
        smoothing_layout = QVBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing (higher = smoother, slower):"))
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setToolTip(
            "Higher values make the cursor steadier but slower to follow your gaze;\n"
            "lower values make it faster but more jittery."
        )
        self.smoothing_slider.setMinimum(1)
        self.smoothing_slider.setMaximum(100)
        self.smoothing_slider.setValue(75)
        self.smoothing_slider.valueChanged.connect(self._update_smoothing)
        self.smoothing_value_label = QLabel("75")
        smoothing_controls = QHBoxLayout()
        smoothing_controls.addWidget(self.smoothing_slider)
        smoothing_controls.addWidget(self.smoothing_value_label)
        smoothing_layout.addLayout(smoothing_controls)
        layout.addLayout(smoothing_layout)

        # Blink threshold
        blink_layout = QVBoxLayout()
        blink_layout.addWidget(QLabel("Blink Threshold:"))
        self.blink_slider = QSlider(Qt.Orientation.Horizontal)
        self.blink_slider.setMinimum(10)
        self.blink_slider.setMaximum(40)
        self.blink_slider.setValue(21)
        self.blink_slider.valueChanged.connect(self._update_blink_threshold)
        self.blink_value_label = QLabel("0.21")
        blink_controls = QHBoxLayout()
        blink_controls.addWidget(self.blink_slider)
        blink_controls.addWidget(self.blink_value_label)
        blink_layout.addLayout(blink_controls)
        layout.addLayout(blink_layout)

        # Dwell time
        dwell_layout = QVBoxLayout()
        dwell_layout.addWidget(QLabel("Dwell Time (ms):"))
        self.dwell_slider = QSlider(Qt.Orientation.Horizontal)
        self.dwell_slider.setMinimum(300)
        self.dwell_slider.setMaximum(1500)
        self.dwell_slider.setValue(600)
        self.dwell_slider.valueChanged.connect(self._update_dwell_time)
        self.dwell_value_label = QLabel("600")
        dwell_controls = QHBoxLayout()
        dwell_controls.addWidget(self.dwell_slider)
        dwell_controls.addWidget(self.dwell_value_label)
        dwell_layout.addLayout(dwell_controls)
        layout.addLayout(dwell_layout)

        group.setLayout(layout)
        return group

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Space: Pause/Resume or confirm calibration point
        space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_shortcut.activated.connect(self._handle_space_key)

        # C: Calibrate
        cal_shortcut = QShortcut(QKeySequence("C"), self)
        cal_shortcut.activated.connect(self._start_calibration)

        # Q: Quit
        quit_shortcut = QShortcut(QKeySequence("Q"), self)
        quit_shortcut.activated.connect(self.close)

    def _initialize_system(self):
        """Initialize camera and tracking system."""
        try:
            # Initialize camera
            self.camera = CameraCapture(camera_id=self.camera_id, width=640, height=480, fps=30)

            if not self.camera.start():
                raise Exception(f"Failed to open camera {self.camera_id}")

            # Initialize detector
            self.detector = FaceDetector(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Initialize mouse controller
            self.mouse_controller = MouseController(enabled=False)
            screen_width, screen_height = self.mouse_controller.get_screen_size()

            # Initialize tracker
            self.tracker = GazeTracker(
                screen_width=screen_width,
                screen_height=screen_height,
                smoothing_method="kalman",
                smoothing_factor=0.3
            )
            # Apply the slider's initial value so filter and UI agree from
            # the first frame, not only after the slider is moved
            self._update_smoothing(self.smoothing_slider.value())

            # Initialize click manager
            self.click_manager = ClickManager(self.mouse_controller)

            # Initialize calibration manager
            self.calibration_manager = CalibrationManager(
                screen_width=screen_width,
                screen_height=screen_height,
                num_points=9,
                mapping_method="affine"
            )

            # On-screen dwell progress ring (shown only in Dwell click mode)
            self.dwell_indicator = DwellIndicator(screen_size=(screen_width, screen_height))

            # Start processing thread
            self._start_processing_thread()

            self.is_initialized = True
            self.status_label.setText("Ready - Click 'Enable Tracking' to start")

        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize: {e}")
            self.status_label.setText(f"Error: {e}")

    @pyqtSlot(np.ndarray, dict)
    def _on_frame_processed(self, frame: np.ndarray, info: dict):
        """Handle processed frame from processing thread."""
        # Update video display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

        # Update metrics
        self.fps_label.setText(f"FPS: {info['fps']:.1f}")
        self.latency_label.setText(f"Latency: {info['latency']:.1f} ms")
        if not info['face_detected']:
            self.confidence_label.setText("Face: Not detected")
        elif info['confidence'] is not None:
            self.confidence_label.setText(f"Face: Detected ({info['confidence']:.2f})")
        else:
            self.confidence_label.setText("Face: Detected")

        # Dwell progress ring (dwell_info is only set while tracking in
        # Dwell mode)
        if self.dwell_indicator is not None:
            self.dwell_indicator.update_dwell(info['dwell_info'] if self.is_tracking else None)

        # Update status
        if info['calibrating']:
            current, total = info['calibration_progress']
            self.calibration_progress.setValue(current)
            target = info['calibration_target']
            if target:
                auto_confirm = self.auto_confirm_checkbox.isChecked()
                self.calibration_label.setText(
                    f"Look at the target point ({current + 1}/{total})\n"
                    + ("Hold your gaze until it is confirmed" if auto_confirm
                       else "Press Space when stable")
                )
                is_stable = self.calibration_manager.is_current_point_stable()
                if auto_confirm and not self._point_settled:
                    # Samples are still from before the eyes reached the target
                    is_stable = False

                # Show calibration overlay with target point
                if self.calibration_overlay:
                    progress_text = f"Calibration Point {current + 1} of {total}"
                    self.calibration_overlay.set_target(
                        target[0], target[1], progress_text, is_stable=is_stable
                    )
                    if not info['face_detected']:
                        self.calibration_overlay.set_status(
                            "No face detected - adjust position/lighting"
                        )
                    elif self.calibration_overlay.status_text.startswith("No face"):
                        self.calibration_overlay.set_status("")

                if auto_confirm and info['face_detected']:
                    self._auto_confirm_step()
        elif not info['face_detected']:
            self.status_label.setText("No face detected - adjust position/lighting")
        elif self.is_tracking:
            self.status_label.setText("Tracking active")
        else:
            self.status_label.setText("Ready")

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Handle processing errors."""
        logger.warning("Processing error: %s", error_msg)

    def _toggle_tracking(self):
        """Toggle tracking on/off."""
        if not self.is_initialized:
            return

        self.is_tracking = self.tracking_button.isChecked()

        if self.processing_thread:
            self.processing_thread.tracking_enabled = self.is_tracking

        if self.mouse_controller:
            self.mouse_controller.set_enabled(self.is_tracking)

        if self.is_tracking:
            if not self.calibration_manager.is_complete():
                reply = QMessageBox.question(
                    self,
                    "Calibration Required",
                    "System is not calibrated. Start calibration now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.tracking_button.setChecked(False)
                    self.is_tracking = False
                    self._start_calibration()
                    return

            self.tracking_button.setText("Disable Tracking")
            self.tracker.set_mapper(self.calibration_manager.get_mapper())
        else:
            self.tracking_button.setText("Enable Tracking")
            if self.dwell_indicator is not None:
                self.dwell_indicator.hide()

    def _change_click_mode(self, index: int):
        """Change click detection mode."""
        if not self.click_manager:
            return

        modes = [ClickMode.NONE, ClickMode.BLINK, ClickMode.DWELL, ClickMode.WINK_LEFT]
        self.click_manager.set_mode(modes[index])
        if self.dwell_indicator is not None:
            self.dwell_indicator.hide()

    def _start_calibration(self):
        """Start calibration process."""
        if not self.is_initialized or self.calibration_manager.is_calibrating():
            return

        # Disable tracking
        if self.is_tracking:
            self.tracking_button.setChecked(False)
            self._toggle_tracking()

        # Start calibration
        self.calibration_manager.start_calibration()
        self.calibration_progress.setVisible(True)
        self.calibration_progress.setValue(0)
        self.confirm_button.setVisible(True)
        self.calibrate_button.setEnabled(False)

        _, total = self.calibration_manager.get_progress()
        self.calibration_progress.setMaximum(total)

        # Fullscreen target overlay; it forwards Space/Escape back to us
        self.calibration_overlay = CalibrationOverlay(
            screen_size=(self.calibration_manager.screen_width,
                         self.calibration_manager.screen_height)
        )
        self.calibration_overlay.confirm_requested.connect(self._confirm_calibration_point)
        self.calibration_overlay.cancel_requested.connect(self._cancel_calibration)
        self.calibration_overlay.auto_confirm = self.auto_confirm_checkbox.isChecked()
        self.calibration_overlay.show_on_primary_screen()

        self._begin_calibration_point()

        self.status_label.setText("Calibration started")

    def _begin_calibration_point(self):
        """Reset the auto-confirm timer when a new target is shown."""
        self._point_started_at = time.monotonic()
        self._point_settled = False

    def _auto_confirm_step(self):
        """Confirm the current point automatically once the gaze is stable.

        Called for every processed frame while calibrating with
        auto-confirm enabled. Confirmation runs here in the GUI thread,
        through the same path as pressing Space.
        """
        if time.monotonic() - self._point_started_at < self.AUTO_CONFIRM_SETTLE_TIME:
            return

        if not self._point_settled:
            # Drop samples recorded while the eyes were still moving to the
            # target; stability is judged on fresh samples only
            self.calibration_manager.discard_samples()
            self._point_settled = True
            return

        if self.calibration_manager.is_current_point_stable():
            self._confirm_calibration_point()

    def _close_calibration_overlay(self):
        """Close the calibration overlay if it is open."""
        if self.calibration_overlay is not None:
            self.calibration_overlay.close_overlay()
            self.calibration_overlay.deleteLater()
            self.calibration_overlay = None

    def _end_calibration_ui(self):
        """Restore the calibration controls after calibration ends."""
        self._close_calibration_overlay()
        self.calibration_progress.setVisible(False)
        self.confirm_button.setVisible(False)
        self.calibrate_button.setEnabled(True)
        self.activateWindow()

    def _cancel_calibration(self):
        """Abort calibration and restore the previous calibration, if any."""
        if not self.calibration_manager or not self.calibration_manager.is_calibrating():
            return

        self.calibration_manager.cancel_calibration()
        self._end_calibration_ui()

        if self.calibration_manager.is_complete():
            self.calibration_label.setText("Calibration cancelled - previous calibration kept")
        else:
            self.calibration_label.setText("Calibration cancelled")
        self.status_label.setText("Calibration cancelled")

    def _confirm_calibration_point(self):
        """Confirm current calibration point."""
        if not self.calibration_manager or not self.calibration_manager.is_calibrating():
            return

        if not self.calibration_manager.is_current_point_stable():
            # A message box would open behind the always-on-top overlay and
            # block input, so the warning is shown on the overlay instead
            if self.calibration_overlay is not None:
                self.calibration_overlay.set_status("Not stable yet - keep looking at the target")
            else:
                QMessageBox.warning(self, "Not Stable", "Please keep looking at the target point")
            return

        if self.calibration_overlay is not None:
            self.calibration_overlay.set_status("")

        has_more = self.calibration_manager.confirm_current_point()

        if has_more:
            self._begin_calibration_point()
        else:
            # Calibration finished (successfully or not)
            self._end_calibration_ui()

            if self.calibration_manager.is_complete():
                error = self.calibration_manager.get_mapper().get_calibration_error()
                QMessageBox.information(
                    self,
                    "Calibration Complete",
                    f"Calibration successful!\nAverage error: {format_error_px(error)}"
                )
                self.calibration_label.setText(f"Calibrated (error: {format_error_px(error)})")
            else:
                QMessageBox.warning(self, "Calibration Failed", "Calibration failed. Please try again.")
                self.calibration_label.setText("Calibration failed")

    def _handle_space_key(self):
        """Handle space key press."""
        if self.calibration_manager and self.calibration_manager.is_calibrating():
            self._confirm_calibration_point()
        else:
            self.tracking_button.setChecked(not self.tracking_button.isChecked())
            self._toggle_tracking()

    def _update_smoothing(self, value: int):
        """Update smoothing from the slider (higher value = smoother cursor)."""
        self.smoothing_value_label.setText(str(value))

        # GazeTracker's factor is a responsiveness (higher = follows new
        # measurements more closely = less smoothing), so it is inverted
        responsiveness = 1.0 - value / 100.0

        if self.tracker:
            self.tracker.set_smoothing_factor(responsiveness)

    def _update_blink_threshold(self, value: int):
        """Update blink detection threshold."""
        threshold = value / 100.0
        self.blink_value_label.setText(f"{threshold:.2f}")

        if self.click_manager:
            self.click_manager.configure_blink(threshold)

    def _update_dwell_time(self, value: int):
        """Update dwell time."""
        self.dwell_value_label.setText(str(value))

        if self.click_manager:
            self.click_manager.configure_dwell(value / 1000.0, 30)

    def _save_calibration(self):
        """Save calibration to file."""
        if not self.calibration_manager or not self.calibration_manager.is_complete():
            QMessageBox.warning(self, "No Calibration", "No calibration to save")
            return

        if self.calibration_manager.save_calibration("calibration_data.json"):
            QMessageBox.information(self, "Saved", "Calibration saved to calibration_data.json")
        else:
            QMessageBox.warning(self, "Save Failed", "Failed to save calibration")

    def _load_calibration(self):
        """Load calibration from file."""
        if not self.calibration_manager:
            return

        if self.calibration_manager.load_calibration("calibration_data.json"):
            error = self.calibration_manager.get_mapper().get_calibration_error()
            self.calibration_label.setText(f"Calibrated (error: {format_error_px(error)})")
            QMessageBox.information(self, "Loaded", f"Calibration loaded (error: {format_error_px(error)})")
        else:
            QMessageBox.warning(self, "Load Failed", "Failed to load calibration")

    def _change_camera(self, index: int):
        """Change camera source."""
        if not self.is_initialized or index == self.camera_id:
            return

        # Disable tracking if enabled
        was_tracking = self.is_tracking
        if self.is_tracking:
            self.tracking_button.setChecked(False)
            self._toggle_tracking()

        # Stop current camera and processing
        self._stop_processing()

        # Start new camera
        new_camera = CameraCapture(camera_id=index, width=640, height=480, fps=30)

        if new_camera.start():
            self.camera = new_camera
            self.camera_id = index
            self.status_label.setText(f"Switched to Camera {index}")
        else:
            QMessageBox.critical(self, "Camera Error", f"Failed to open Camera {index}")

            # Go back to the previous camera. Signals are blocked so resetting
            # the combo box does not re-enter this slot.
            self.camera_combo.blockSignals(True)
            self.camera_combo.setCurrentIndex(self.camera_id)
            self.camera_combo.blockSignals(False)

            self.camera = CameraCapture(camera_id=self.camera_id, width=640, height=480, fps=30)
            if not self.camera.start():
                self.status_label.setText(f"Error: failed to reopen Camera {self.camera_id}")
                return
            self.status_label.setText(f"Camera {index} unavailable - back to Camera {self.camera_id}")

        # Restart processing with whichever camera is now open
        self._start_processing_thread()

        # Re-enable tracking if it was enabled
        if was_tracking:
            self.tracking_button.setChecked(True)
            self._toggle_tracking()

    def _start_processing_thread(self):
        """Start a processing thread for the current camera."""
        self.processing_thread = ProcessingThread(
            self.camera,
            self.detector,
            self.tracker,
            self.click_manager
        )
        self.processing_thread.calibration_manager = self.calibration_manager
        self.processing_thread.frame_processed.connect(self._on_frame_processed)
        self.processing_thread.error_occurred.connect(self._on_error)
        self.processing_thread.start()

    def _stop_processing(self):
        """Stop the processing thread, then release the camera."""
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.processing_thread = None

        if self.camera:
            self.camera.stop()

    def closeEvent(self, event):
        """Handle window close."""
        self._close_calibration_overlay()
        if self.dwell_indicator is not None:
            self.dwell_indicator.close()
        self._stop_processing()

        if self.detector:
            self.detector.close()

        event.accept()


def run_gui(camera_id: int = 0):
    """Run the GUI application.

    Args:
        camera_id: Camera device ID to open at startup
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = EyeMouseGUI(camera_id=camera_id)
    window.show()

    sys.exit(app.exec())
