from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import mediapipe as mp
import numpy as np

from core.config import OverlayConfig
from core.event_bus import EventBus
from vision.gaze_tracker import GazeResult, GazeTracker
from vision.hand_tracker import HandResult, HandTracker

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class OverlayState:
    gesture: str = "NONE"
    gesture_conf: float = 0.0
    gaze: str = "NONE"
    gaze_conf: float = 0.0
    mode: str = "SAFE"
    intent: str = "NONE"
    fps: float = 0.0
    vision_latency_ms: float = 0.0


class PreviewOverlay:
    """
    Simple PiP preview window that runs hand + gaze tracking on camera frames
    and draws landmarks plus status labels.
    """

    def __init__(
        self,
        config: OverlayConfig,
        hand_tracker: Optional[HandTracker] = None,
        gaze_tracker: Optional[GazeTracker] = None,
        bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config
        self.hand_tracker = hand_tracker or HandTracker()
        self.gaze_tracker = gaze_tracker or GazeTracker()
        self.state = OverlayState()
        self.bus = bus

    def set_mode(self, mode: str) -> None:
        self.state.mode = mode

    def set_intent(self, intent: str) -> None:
        self.state.intent = intent

    def run(self, camera_index: int | None = None, stop_event: Optional["threading.Event"] = None) -> None:
        cam_idx = self.config.camera_index if camera_index is None else camera_index
        cap = self._open_camera(cam_idx)
        if cap is None:
            raise RuntimeError("Could not open any camera. Please check connections or permissions.")

        window_flag = cv2.WINDOW_NORMAL if self.config.allow_resize else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(self.config.window_title, window_flag)
        cv2.resizeWindow(self.config.window_title, *self.config.box_size)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                ts = time.time()
                ok, frame = cap.read()
                if not ok:
                    break

                if self.config.mirror:
                    frame = cv2.flip(frame, 1)

                hand_res = self.hand_tracker.process(frame)
                gaze_res = self.gaze_tracker.process(frame)

                self._update_state(hand_res, gaze_res)
                self._draw(frame, hand_res, gaze_res, ts)

                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        finally:
            cap.release()
            self.hand_tracker.close()
            self.gaze_tracker.close()
            cv2.destroyAllWindows()

    def _open_camera(self, preferred_index: int) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(preferred_index)
        if self.config.capture_size:
            try:
                w, h = self.config.capture_size
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            except Exception:
                pass
        if cap.isOpened():
            return cap

        print(f"Preferred camera index {preferred_index} unavailable. Scanning for available cameras...")
        available = self._scan_cameras()
        if not available:
            print("No cameras detected.")
            return None

        chosen = available[0]
        try:
            choice = input(f"Select camera from {available} (default {chosen}): ").strip()
            if choice:
                chosen = int(choice)
        except Exception:
            pass

        cap = cv2.VideoCapture(chosen)
        if cap.isOpened():
            print(f"Using camera index {chosen}")
            return cap

        print(f"Could not open selected camera index {chosen}.")
        return None

    def _scan_cameras(self, max_index: int = 5) -> List[int]:
        found: List[int] = []
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(idx)
                cap.release()
        return found

    # Drawing helpers -----------------------------------------------------
    def _draw(
        self,
        frame: np.ndarray,
        hand_res: HandResult,
        gaze_res: GazeResult,
        start_ts: float,
    ) -> None:
        if hand_res.landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_res.landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        if gaze_res.landmarks:
            mp_drawing.draw_landmarks(
                frame,
                gaze_res.landmarks,
                mp_face_mesh_connections(),
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1),
            )

        resized = self._resize_for_display(frame)
        end_ts = time.time()
        self.state.vision_latency_ms = (end_ts - start_ts) * 1000.0
        self._draw_status_bar(resized)
        cv2.imshow(self.config.window_title, resized)

    def _draw_status_bar(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        bar_height = 70
        cv2.rectangle(frame, (0, 0), (w, bar_height), (0, 0, 0), -1)

        txt = (
            f"Gesture: {self.state.gesture} ({self.state.gesture_conf:.2f})  |  "
            f"Gaze: {self.state.gaze} ({self.state.gaze_conf:.2f})  |  "
            f"Mode: {self.state.mode}  |  Intent: {self.state.intent}  |  "
            f"FPS: {self.state.fps:.1f}  |  Latency: {self.state.vision_latency_ms:.1f}ms"
        )
        cv2.putText(
            frame,
            txt,
            (12, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


    def _update_state(self, hand_res: HandResult, gaze_res: GazeResult) -> None:
        now = time.time()
        if not hasattr(self, "_last_frame_ts"):
            self._last_frame_ts = now
        dt = now - getattr(self, "_last_frame_ts")
        self._last_frame_ts = now
        if dt > 0:
            # Low-pass FPS to reduce flicker.
            self.state.fps = self.state.fps * 0.9 + (1.0 / dt) * 0.1 if self.state.fps else 1.0 / dt

        self.state.gesture = hand_res.gesture
        self.state.gesture_conf = hand_res.confidence
        self.state.gaze = gaze_res.direction
        self.state.gaze_conf = gaze_res.confidence

        # Pull latest intent/mode from bus if available
        if self.bus:
            intent = self.bus.latest("intent")
            if intent:
                self.state.intent = intent.name
            mode = self.bus.latest("mode")
            if mode:
                self.state.mode = mode.name

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """
        Maintain aspect ratio and pad to the current window size if resizable;
        otherwise resize to the configured box_size.
        """
        target_w, target_h = self.config.box_size

        if self.config.allow_resize:
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(self.config.window_title)
                if win_w > 0 and win_h > 0:
                    target_w, target_h = win_w, win_h
            except Exception:
                pass

        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to target size to avoid stretching or cutoff
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        if pad_top < 0 or pad_left < 0:
            return resized  # fallback, shouldn't happen

        return cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )


def mp_face_mesh_connections():
    return mp.solutions.face_mesh.FACEMESH_TESSELATION
