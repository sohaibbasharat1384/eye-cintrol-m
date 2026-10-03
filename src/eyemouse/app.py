"""Main application entry point."""

import sys
import argparse
import logging
import os

# Add src directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eyemouse.gui import run_gui
from eyemouse.hotkey import DEFAULT_STOP_HOTKEY

logger = logging.getLogger("eyemouse")


def main():
    """Main entry point for eye control mouse application."""
    parser = argparse.ArgumentParser(
        description="Eye Control Mouse - Hands-free mouse control using webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eyemouse              # Run with GUI
  eyemouse --debug      # Run with debug logging
  eyemouse --camera 1   # Use another camera
  eyemouse --stop-hotkey "<ctrl>+<shift>+s"   # Change the emergency stop key

Keyboard Shortcuts (in GUI):
  Space    - Pause/Resume tracking or confirm calibration point
  C        - Start calibration
  Esc      - Cancel calibration
  Q        - Quit application

System-wide (works in any application):
  Ctrl+Alt+Space - Emergency stop: disable tracking / cancel calibration

For help and documentation, visit: https://github.com/sohaibbasharat1384/eye-control-mouse
        """
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID (default: the last used camera, or 0)"
    )

    parser.add_argument(
        "--stop-hotkey",
        default=DEFAULT_STOP_HOTKEY,
        help=(
            "System-wide emergency stop hotkey in pynput syntax "
            f"(default: {DEFAULT_STOP_HOTKEY})"
        )
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.debug("Debug logging enabled")

    # Run GUI
    try:
        run_gui(camera_id=args.camera, stop_hotkey=args.stop_hotkey)
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
