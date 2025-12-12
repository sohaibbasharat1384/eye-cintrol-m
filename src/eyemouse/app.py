"""Main application entry point."""

import sys
import argparse
import os

# Add src directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eyemouse.gui import run_gui


def main():
    """Main entry point for eye control mouse application."""
    parser = argparse.ArgumentParser(
        description="Eye Control Mouse - Hands-free mouse control using webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eyemouse              # Run with GUI
  eyemouse --debug      # Run with debug logging

Keyboard Shortcuts (in GUI):
  Space    - Pause/Resume tracking or confirm calibration point
  C        - Start calibration
  Q        - Quit application

For help and documentation, visit: https://github.com/your-repo
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
        default=0,
        help="Camera device ID (default: 0)"
    )

    args = parser.parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("Debug logging enabled")

    # Run GUI
    try:
        run_gui()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
