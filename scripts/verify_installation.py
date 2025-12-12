"""Verify that all dependencies are installed correctly."""

import sys
import importlib


def check_import(module_name, display_name=None):
    """Check if a module can be imported."""
    if display_name is None:
        display_name = module_name

    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK] {display_name:20} (version: {version})")
        return True
    except ImportError as e:
        print(f"[FAIL] {display_name:20} - NOT FOUND")
        print(f"  Error: {e}")
        return False


def check_camera():
    """Check if camera is accessible."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"[OK] {'Camera Access':20}")
                return True
            else:
                print(f"[FAIL] {'Camera Access':20} - Can't read frames")
                return False
        else:
            print(f"[FAIL] {'Camera Access':20} - Can't open camera")
            return False
    except Exception as e:
        print(f"[FAIL] {'Camera Access':20} - Error: {e}")
        return False


def main():
    """Run verification checks."""
    print("=" * 60)
    print("Eye Control Mouse - Installation Verification")
    print("=" * 60)

    print(f"\nPython Version: {sys.version}")
    print(f"Platform: {sys.platform}\n")

    print("Checking dependencies...\n")

    checks = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pyautogui", "PyAutoGUI"),
        ("pynput", "PyNput"),
        ("PyQt6", "PyQt6"),
        ("pykalman", "PyKalman"),
    ]

    results = []
    for module, display in checks:
        results.append(check_import(module, display))

    print("\nChecking hardware...\n")
    camera_ok = check_camera()

    print("\n" + "=" * 60)

    all_ok = all(results) and camera_ok

    if all_ok:
        print("[OK] All checks passed! Installation is complete.")
        print("\nNext steps:")
        print("1. Run: eyemouse")
        print("2. Or try demo: python scripts/demo_basic.py")
    else:
        print("[FAIL] Some checks failed. Please install missing dependencies.")
        print("\nTo install all dependencies:")
        print("  pip install -r requirements.txt")

        if not camera_ok:
            print("\nCamera issues:")
            print("- Check camera permissions")
            print("- Ensure camera is not in use by another application")
            print("- Try a different camera ID (--camera 1)")

    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
