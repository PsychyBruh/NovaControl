from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.gaze_tracker import GazeTracker
from vision.hand_tracker import HandTracker


POINTS_ORDER = [
    ("center", "Look at CENTER"),
    ("top_left", "Look at TOP LEFT"),
    ("top_right", "Look at TOP RIGHT"),
    ("bottom_right", "Look at BOTTOM RIGHT"),
    ("bottom_left", "Look at BOTTOM LEFT"),
]


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


def capture_samples(
    cap,
    gaze: GazeTracker,
    hand: HandTracker,
    label: str,
    wait_timeout: float = 12.0,
    capture_duration: float = 1.0,
    stable_frames: int = 5,
) -> Tuple[float, float]:
    """
    Waits for an OPEN_PALM to start capture, then collects gaze samples for capture_duration.
    """
    samples_x = []
    samples_y = []
    wait_start = time.time()
    stable = 0

    while time.time() - wait_start < wait_timeout:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        g_res = gaze.process(frame)
        h_res = hand.process(frame)

        msg = f"{label} | Show OPEN PALM to start capture"
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Screen Gaze Calibration", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

        if h_res.gesture == "OPEN_PALM":
            stable += 1
        else:
            stable = 0

        if stable >= stable_frames:
            # Begin capture window
            capture_start = time.time()
            while time.time() - capture_start < capture_duration:
                ok, frame2 = cap.read()
                if not ok:
                    break
                frame2 = cv2.flip(frame2, 1)
                g2 = gaze.process(frame2)
                if g2.ratio is not None and g2.landmarks:
                    samples_x.append(g2.ratio)
                    v = vertical_from_landmarks(g2.landmarks)
                    if v is not None:
                        samples_y.append(v)
                cv2.putText(
                    frame2,
                    f"Capturing {label}...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Screen Gaze Calibration", frame2)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            break

    if not samples_x or not samples_y:
        return 0.5, 0.5
    return sum(samples_x) / len(samples_x), sum(samples_y) / len(samples_y)


def calibrate(camera_index: int | None, output: Path) -> None:
    gaze = GazeTracker(bus=None)
    hand = HandTracker(bus=None)
    cap = open_camera(camera_index)

    results: Dict[str, Dict[str, float]] = {}
    try:
        for key, label in POINTS_ORDER:
            print(f"Calibrating {key}... Look at target, show OPEN PALM to capture.")
            x, y = capture_samples(duration=1.2, cap=cap, gaze=gaze, hand=hand, label=label)
            results[key] = {"x": x, "y": y}
            img = frame_black(cap)
            img = cv2.putText(
                img,
                f"Captured {key}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Screen Gaze Calibration", img)
            cv2.waitKey(400)
    finally:
        cap.release()
        gaze.close()
        hand.close()
        cv2.destroyAllWindows()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"Saved screen gaze calibration to {output}")


def frame_black(cap) -> "cv2.Mat":
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception:
        w, h = 640, 360
    return np.zeros((h, w, 3), dtype=np.uint8)


def vertical_from_landmarks(landmarks) -> float | None:
    pts = landmarks.landmark
    left_top, left_bottom = pts[159], pts[145]
    right_top, right_bottom = pts[386], pts[374]
    left_iris = pts[468]
    right_iris = pts[473]

    def _v_ratio(top, bottom, iris):
        span = bottom.y - top.y
        if span == 0:
            return None
        return (iris.y - top.y) / span

    vals = [r for r in (_v_ratio(left_top, left_bottom, left_iris), _v_ratio(right_top, right_bottom, right_iris)) if r is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate screen-referenced gaze with 5-point capture.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (optional; will prompt if not set).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts") / "screen_gaze_calibration.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()
    calibrate(args.camera, args.output)


if __name__ == "__main__":
    main()
