"""Build script for creating standalone executables."""

import os
import sys
import subprocess
import platform


def build_windows():
    """Build Windows executable."""
    print("Building Windows executable...")

    cmd = [
        "pyinstaller",
        "--name=EyeMouse",
        "--onefile",
        "--windowed",
        "--icon=assets/icon.ico" if os.path.exists("assets/icon.ico") else "",
        "--add-data=src;src",
        "src/eyemouse/app.py",
    ]

    # Remove empty strings
    cmd = [c for c in cmd if c]

    subprocess.run(cmd, check=True)
    print("\nBuild complete! Executable: dist/EyeMouse.exe")


def build_macos():
    """Build macOS application."""
    print("Building macOS application...")

    cmd = [
        "pyinstaller",
        "--name=EyeMouse",
        "--onefile",
        "--windowed",
        "--icon=assets/icon.icns" if os.path.exists("assets/icon.icns") else "",
        "--add-data=src:src",
        "src/eyemouse/app.py",
    ]

    cmd = [c for c in cmd if c]

    subprocess.run(cmd, check=True)
    print("\nBuild complete! Application: dist/EyeMouse.app")


def build_linux():
    """Build Linux executable."""
    print("Building Linux executable...")

    cmd = [
        "pyinstaller",
        "--name=eyemouse",
        "--onefile",
        "--add-data=src:src",
        "src/eyemouse/app.py",
    ]

    subprocess.run(cmd, check=True)
    print("\nBuild complete! Executable: dist/eyemouse")


def main():
    """Main build function."""
    system = platform.system()

    print("=" * 60)
    print("Eye Control Mouse - Build Script")
    print("=" * 60)
    print(f"\nPlatform: {system}")
    print("\nThis will create a standalone executable for distribution.")
    print("\nRequirements:")
    print("- pip install pyinstaller")
    print("- All dependencies from requirements.txt")

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Build cancelled.")
        return

    try:
        if system == "Windows":
            build_windows()
        elif system == "Darwin":
            build_macos()
        elif system == "Linux":
            build_linux()
        else:
            print(f"Unsupported platform: {system}")
            return

        print("\n" + "=" * 60)
        print("Build successful!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test the executable in dist/")
        print("2. Create installer package (optional)")
        print("   - Windows: Use Inno Setup or NSIS")
        print("   - macOS: Use create-dmg")
        print("   - Linux: Use AppImage or dpkg")

    except subprocess.CalledProcessError as e:
        print(f"\n\nBuild failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
