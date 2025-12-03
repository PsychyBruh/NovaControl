from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from vision.gaze_tracker import GazeTracker


def scan_cameras(max_index: int = 5) -> list[int]:
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(idx)
            cap.release()
    return available


def open_camera(preferred: int | None) -> cv2.VideoCapture:
    if preferred is not None:
        cap = cv2.VideoCapture(preferred)
        if cap.isOpened():
            return cap

    available = scan_cameras()
    if not available:
        print("No cameras detected.", file=sys.stderr)
        sys.exit(1)

    chosen = available[0]
    try:
        choice = input(f"Select camera from {available} (default {chosen}): ").strip()
        if choice:
            chosen = int(choice)
    except Exception:
        pass

    cap = cv2.VideoCapture(chosen)
    if not cap.isOpened():
        print(f"Could not open camera {chosen}", file=sys.stderr)
        sys.exit(1)
    return cap


def calibrate(duration: float, camera_index: int | None, output: Path) -> None:
    tracker = GazeTracker(bus=None)
    cap = open_camera(camera_index)
    ratios = []
    start = time.time()

    try:
        while time.time() - start < duration:
            ok, frame = cap.read()
            if not ok:
                break
            result = tracker.process(frame)
            if result.ratio is not None:
                ratios.append(result.ratio)

            cv2.putText(
                frame,
                "Calibrating gaze center... look straight.",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Gaze Calibration", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    if not ratios:
        print("No gaze samples collected; calibration failed.", file=sys.stderr)
        sys.exit(1)

    center = sum(ratios) / len(ratios)
    output_data = {"center_ratio": center, "samples": len(ratios), "duration_sec": duration}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, indent=2))
    print(f"Saved gaze calibration to {output} (center_ratio={center:.3f}, samples={len(ratios)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate gaze center using FaceMesh iris landmarks.")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds to sample gaze.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (optional, will prompt if not set).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts") / "gaze_calibration.json",
        help="Path to save calibration JSON.",
    )
    args = parser.parse_args()
    calibrate(args.duration, args.camera, args.output)


if __name__ == "__main__":
    main()
