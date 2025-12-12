"""PyQt6 GUI for eye control mouse application."""

import time
import sys
from typing import Optional
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QGroupBox, QMessageBox,
    QProgressBar, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QPoint
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut, QPainter, QColor, QPen, QFont

from .capture import CameraCapture
from .detector import FaceDetector
from .tracker import GazeTracker
from .calibration import CalibrationManager
from .clicker import MouseController, ClickManager, ClickMode
from .utils import PerformanceMetrics


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
                    'confidence': 0.0,
                    'calibrating': False,
                    'calibration_target': None,
                    'calibration_progress': (0, 0),
                    'dwell_info': None
                }

                if detection_result is not None:
                    info['confidence'] = detection_result.get_confidence()

                    # Handle calibration
                    if self.calibration_manager and self.calibration_manager.is_calibrating():
                        info['calibrating'] = True
                        info['calibration_target'] = self.calibration_manager.get_current_target_point()
                        info['calibration_progress'] = self.calibration_manager.get_progress()

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
    """Fullscreen overlay for calibration targets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target_point = None
        self.progress_text = ""

        # Make window frameless and fullscreen
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()

        # Set black semi-transparent background
        self.setStyleSheet("background-color: rgba(0, 0, 0, 180);")

    def set_target(self, x: int, y: int, progress_text: str):
        """Set the current calibration target point."""
        self.target_point = (x, y)
        self.progress_text = progress_text
        self.update()

    def paintEvent(self, event):
        """Draw the calibration target."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.target_point:
            x, y = self.target_point

            # Draw target circles (bullseye pattern)
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.setBrush(QColor(255, 0, 0))
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
            text_rect.setTop(self.height() - 100)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter,
                           f"{self.progress_text}\n\nLook at the red target\nPress SPACE when ready")


class EyeMouseGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

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

        self.is_initialized = False
        self.is_tracking = False

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
        self.confidence_label = QLabel("Confidence: --")
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
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2", "Camera 3"])
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
        smoothing_layout.addWidget(QLabel("Smoothing:"))
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setMinimum(1)
        self.smoothing_slider.setMaximum(100)
        self.smoothing_slider.setValue(30)
        self.smoothing_slider.valueChanged.connect(self._update_smoothing)
        self.smoothing_value_label = QLabel("30")
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
            self.camera = CameraCapture(camera_id=0, width=640, height=480, fps=30)

            if not self.camera.start():
                raise Exception("Failed to open camera")

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

            # Initialize click manager
            self.click_manager = ClickManager(self.mouse_controller)

            # Initialize calibration manager
            self.calibration_manager = CalibrationManager(
                screen_width=screen_width,
                screen_height=screen_height,
                num_points=9,
                mapping_method="affine"
            )

            # Start processing thread
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
        self.confidence_label.setText(f"Confidence: {info['confidence']:.2f}")

        # Update status
        if info['calibrating']:
            current, total = info['calibration_progress']
            self.calibration_progress.setValue(int(100 * current / total))
            target = info['calibration_target']
            if target:
                self.calibration_label.setText(
                    f"Look at the target point ({current + 1}/{total})\n"
                    f"Press Space when stable"
                )
                # Show calibration overlay with target point
                if self.calibration_overlay:
                    progress_text = f"Calibration Point {current + 1} of {total}"
                    self.calibration_overlay.set_target(target[0], target[1], progress_text)
        elif not info['face_detected']:
            self.status_label.setText("No face detected - adjust position/lighting")
        elif self.is_tracking:
            self.status_label.setText("Tracking active")
        else:
            self.status_label.setText("Ready")

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Handle processing errors."""
        print(f"Processing error: {error_msg}")

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

    def _change_click_mode(self, index: int):
        """Change click detection mode."""
        if not self.click_manager:
            return

        modes = [ClickMode.NONE, ClickMode.BLINK, ClickMode.DWELL, ClickMode.WINK_LEFT]
        self.click_manager.set_mode(modes[index])

    def _start_calibration(self):
        """Start calibration process."""
        if not self.is_initialized:
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

        self.status_label.setText("Calibration started")

    def _confirm_calibration_point(self):
        """Confirm current calibration point."""
        if not self.calibration_manager or not self.calibration_manager.is_calibrating():
            return

        if not self.calibration_manager.is_current_point_stable():
            QMessageBox.warning(self, "Not Stable", "Please keep looking at the target point")
            return

        has_more = self.calibration_manager.confirm_current_point()

        if not has_more:
            # Calibration complete
            self.calibration_progress.setVisible(False)
            self.confirm_button.setVisible(False)
            self.calibrate_button.setEnabled(True)

            if self.calibration_manager.is_complete():
                error = self.calibration_manager.get_mapper().get_calibration_error()
                QMessageBox.information(
                    self,
                    "Calibration Complete",
                    f"Calibration successful!\nAverage error: {error:.1f} pixels"
                )
                self.calibration_label.setText(f"Calibrated (error: {error:.1f}px)")
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
        """Update smoothing factor."""
        factor = value / 100.0
        self.smoothing_value_label.setText(str(value))

        if self.tracker:
            self.tracker.set_smoothing_factor(factor)

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
            self.calibration_label.setText(f"Calibrated (error: {error:.1f}px)")
            QMessageBox.information(self, "Loaded", f"Calibration loaded (error: {error:.1f}px)")
        else:
            QMessageBox.warning(self, "Load Failed", "Failed to load calibration")

    def _change_camera(self, index: int):
        """Change camera source."""
        if not self.is_initialized:
            return

        # Disable tracking if enabled
        was_tracking = self.is_tracking
        if self.is_tracking:
            self.tracking_button.setChecked(False)
            self._toggle_tracking()

        # Stop current camera and processing
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread.wait()

        if self.camera:
            self.camera.stop()

        # Start new camera
        self.camera = CameraCapture(camera_id=index, width=640, height=480, fps=30)

        if not self.camera.start():
            QMessageBox.critical(self, "Camera Error", f"Failed to open Camera {index}")
            # Try to go back to camera 0
            self.camera_combo.setCurrentIndex(0)
            self.camera = CameraCapture(camera_id=0, width=640, height=480, fps=30)
            self.camera.start()
            return

        # Restart processing thread with new camera
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

        self.status_label.setText(f"Switched to Camera {index}")

        # Re-enable tracking if it was enabled
        if was_tracking:
            self.tracking_button.setChecked(True)
            self._toggle_tracking()

    def closeEvent(self, event):
        """Handle window close."""
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread.wait()

        if self.camera:
            self.camera.stop()

        if self.detector:
            self.detector.close()

        event.accept()


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = EyeMouseGUI()
    window.show()

    sys.exit(app.exec())
