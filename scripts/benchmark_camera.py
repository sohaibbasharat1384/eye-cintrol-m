"""Find out where the frame rate is lost: camera capture or processing.

Measures, for one camera:
  1. raw capture FPS for each OpenCV backend available on this platform
     (on Windows: MSMF, the OpenCV default, and DirectShow), with the
     camera's default pixel format and with MJPG requested
  2. face landmark detection time per frame, on live frames

Close Eye Control Mouse (or anything else using the camera) first.
Frames are processed in memory only; nothing is saved.

Usage:
    python scripts/benchmark_camera.py [--camera 0] [--seconds 5]
"""

import os
import sys

# Make the package importable when run as `python scripts/<name>.py` from a
# source checkout, without requiring `pip install -e .`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Silence OpenCV's per-frame warnings (a failing backend can print thousands)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import argparse
import time

import cv2
import numpy as np


def fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "?"
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() else "?"


def backends_for_platform():
    if sys.platform == "win32":
        return [("MSMF (default)", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
    if sys.platform == "darwin":
        return [("AVFoundation (default)", cv2.CAP_AVFOUNDATION)]
    return [("V4L2 (default)", cv2.CAP_V4L2)]


def measure_capture(camera_id, backend, mjpg, seconds, width=640, height=480, fps=30):
    """Return a result dict, or None if the camera could not be opened."""
    t_open = time.perf_counter()
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        return None

    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Startup: wait for the first frame (up to 5 s)
    first_ok = False
    while time.perf_counter() - t_open < 5:
        ok, frame = cap.read()
        if ok:
            first_ok = True
            break
    startup = time.perf_counter() - t_open

    frames, failures, brightness = 0, 0, []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        if ok:
            frames += 1
            if frames % 10 == 0:
                brightness.append(float(frame.mean()))
        else:
            failures += 1
            time.sleep(0.005)
    elapsed = time.perf_counter() - t0

    result = {
        "fps": frames / elapsed,
        "failures": failures,
        "startup": startup if first_ok else None,
        "size": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        "reported_fps": cap.get(cv2.CAP_PROP_FPS),
        "format": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        "brightness": float(np.mean(brightness)) if brightness else None,
    }
    cap.release()
    return result


def measure_detection(camera_id, backend, seconds):
    """Time FaceDetector.process() on live frames."""
    from eyemouse.detector import FaceDetector

    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = FaceDetector()
    times, faces = [], 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        t = time.perf_counter()
        result = detector.process(cv2.flip(frame, 1))
        times.append(time.perf_counter() - t)
        faces += result is not None
    cap.release()
    detector.close()
    if not times:
        return None
    return {"ms": 1000 * float(np.mean(times)), "frames": len(times), "faces": faces}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds per measurement")
    args = parser.parse_args()

    print(f"Camera {args.camera}, {args.seconds:.0f} s per test, requesting 640x480 @ 30 FPS\n")
    print(f"{'backend':24} {'format':7} {'FPS':>6} {'startup':>8} {'fails':>6}  size      fmt   light")

    best = None
    for name, backend in backends_for_platform():
        for mjpg in (False, True):
            r = measure_capture(args.camera, backend, mjpg, args.seconds)
            label = "MJPG" if mjpg else "default"
            if r is None:
                print(f"{name:24} {label:7}  could not open (camera busy or unsupported)")
                continue
            startup = f"{r['startup']:.1f}s" if r["startup"] is not None else "none"
            light = f"{r['brightness']:.0f}" if r["brightness"] is not None else "-"
            print(f"{name:24} {label:7} {r['fps']:6.1f} {startup:>8} {r['failures']:6d}  "
                  f"{r['size']:9} {r['format']:5} {light:>5}")
            if r["fps"] > 0 and (best is None or r["fps"] > best[2]):
                best = (name, backend, r["fps"], label)

    if best is None:
        print("\nNo frames could be captured. Close other applications using the camera")
        print("(including Eye Control Mouse) and retry.")
        return 1

    print(f"\nFastest capture: {best[0]} / {best[3]} at {best[2]:.1f} FPS")
    det = measure_detection(args.camera, best[1], args.seconds)
    if det:
        print(f"Face detection: {det['ms']:.1f} ms per frame "
              f"(face found in {det['faces']}/{det['frames']} frames)")
        processing_limit = 1000.0 / det["ms"]
        print(f"Processing alone could sustain about {processing_limit:.0f} FPS")
        if best[2] < 0.6 * processing_limit:
            print("\n=> The camera is the bottleneck, not the processing.")
        else:
            print("\n=> Processing is the bottleneck.")
    print("Average brightness (0-255) is shown as 'light'; many webcams lower their")
    print("frame rate in dim light (values below ~60) to lengthen exposure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
