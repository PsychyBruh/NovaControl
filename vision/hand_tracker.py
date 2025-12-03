from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from core.event_bus import EventBus, Event

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


@dataclass
class HandResult:
    gesture: str
    confidence: float
    landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList]
    cursor_norm: Optional[Tuple[float, float]]
    handedness: Optional[str]


class HandTracker:
    """
    MediaPipe Hands wrapper with lightweight gesture classification and cursor smoothing.
    Publishes gesture + point events to the EventBus (if provided).
    """

    def __init__(
        self,
        bus: Optional[EventBus] = None,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        pinch_threshold: float = 0.05,
        stable_window: int = 3,
        cursor_ema: float = 0.2,
        auto_stretch: bool = True,
        stretch_decay: float = 0.995,
        min_span: float = 0.1,
    ) -> None:
        self.bus = bus
        self.pinch_threshold = pinch_threshold
        self.stable_window = stable_window
        self.cursor_alpha = cursor_ema
        self.auto_stretch = auto_stretch
        self.stretch_decay = stretch_decay
        self.min_span = min_span
        self._hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_gesture = "NONE"
        self._stable_frames = 0
        self._cursor_state: Optional[Tuple[float, float]] = None
        # Track observed fingertip range for auto-stretch mapping to screen edges.
        self._range_min = [0.25, 0.25]
        self._range_max = [0.75, 0.75]

    def process(self, frame_bgr: np.ndarray) -> HandResult:
        start_ts = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            self._reset_stability()
            return HandResult("NONE", 0.0, None, self._cursor_state, None)

        hand_landmarks = result.multi_hand_landmarks[0]
        handedness = (
            result.multi_handedness[0].classification[0].label
            if result.multi_handedness
            else None
        )

        gesture, conf = self._classify(hand_landmarks)
        gesture = self._stabilize_gesture(gesture)

        cursor_norm = self._update_cursor(hand_landmarks)

        self._publish_events(gesture, conf, handedness, cursor_norm, start_ts)

        return HandResult(gesture, conf, hand_landmarks, cursor_norm, handedness)

    def close(self) -> None:
        self._hands.close()

    # Gesture helpers -----------------------------------------------------
    def _classify(
        self, landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList
    ) -> Tuple[str, float]:
        points = landmarks.landmark

        def _finger_extended(tip: int, pip: int) -> bool:
            return points[tip].y < points[pip].y

        index_up = _finger_extended(8, 6)
        middle_up = _finger_extended(12, 10)
        ring_up = _finger_extended(16, 14)
        pinky_up = _finger_extended(20, 18)

        thumb_tip = points[4]
        index_tip = points[8]
        pinch_dist = np.linalg.norm(
            np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
        )

        if pinch_dist < self.pinch_threshold:
            return "PINCH", max(0.5, 1.0 - pinch_dist * 10)

        extended_count = sum([index_up, middle_up, ring_up, pinky_up])
        if extended_count >= 4:
            return "OPEN_PALM", 0.9
        if index_up and not (middle_up or ring_up or pinky_up):
            return "POINT", 0.85
        if extended_count == 0:
            return "FIST", 0.85

        return "UNKNOWN", 0.3

    def _stabilize_gesture(self, gesture: str) -> str:
        if gesture == self._last_gesture:
            self._stable_frames += 1
        else:
            self._stable_frames = 1
            self._last_gesture = gesture

        if self._stable_frames >= self.stable_window:
            return gesture
        return "NONE"

    def _reset_stability(self) -> None:
        self._stable_frames = 0
        self._last_gesture = "NONE"

    # Cursor smoothing ----------------------------------------------------
    def _update_cursor(
        self, landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList
    ) -> Optional[Tuple[float, float]]:
        pt = landmarks.landmark[8]  # index fingertip
        x = pt.x
        y = pt.y

        if self.auto_stretch:
            x, y = self._stretch_to_edges(x, y)

        current = (x, y)
        if self._cursor_state is None:
            self._cursor_state = current
            return current

        alpha = self.cursor_alpha
        prev_x, prev_y = self._cursor_state
        smoothed = (prev_x * (1 - alpha) + current[0] * alpha, prev_y * (1 - alpha) + current[1] * alpha)
        self._cursor_state = smoothed
        return smoothed

    # Publishing ----------------------------------------------------------
    def _publish_events(
        self,
        gesture: str,
        confidence: float,
        handedness: Optional[str],
        cursor_norm: Optional[Tuple[float, float]],
        ts: float,
    ) -> None:
        if self.bus is None:
            return

        meta = {"handedness": handedness} if handedness else {}
        event = Event(ts=ts, type="gesture", name=gesture, confidence=confidence, meta=meta)
        self._publish(event)

        if cursor_norm:
            cursor_event = Event(
                ts=ts,
                type="point",
                name="CURSOR",
                confidence=confidence,
                meta={"x_norm": cursor_norm[0], "y_norm": cursor_norm[1]},
            )
            self._publish(cursor_event)

    def _publish(self, event: Event) -> None:
        try:
            self.bus.publish_threadsafe(event)
        except Exception:
            # If the bus is not bound yet, fail quietly.
            pass

    def _stretch_to_edges(self, x: float, y: float) -> Tuple[float, float]:
        """
        Track observed fingertip min/max and map to full [0,1] range so camera borders map to screen borders.
        """
        # Exponential decay to avoid range growing forever.
        decay = self.stretch_decay
        self._range_min[0] = min(self._range_min[0], x)
        self._range_min[1] = min(self._range_min[1], y)
        self._range_max[0] = max(self._range_max[0], x)
        self._range_max[1] = max(self._range_max[1], y)

        self._range_min[0] = self._range_min[0] * decay + x * (1 - decay)
        self._range_min[1] = self._range_min[1] * decay + y * (1 - decay)
        self._range_max[0] = self._range_max[0] * decay + x * (1 - decay)
        self._range_max[1] = self._range_max[1] * decay + y * (1 - decay)

        span_x = max(self.min_span, self._range_max[0] - self._range_min[0])
        span_y = max(self.min_span, self._range_max[1] - self._range_min[1])

        norm_x = (x - self._range_min[0]) / span_x
        norm_y = (y - self._range_min[1]) / span_y

        norm_x = float(np.clip(norm_x, 0.0, 1.0))
        norm_y = float(np.clip(norm_y, 0.0, 1.0))
        return norm_x, norm_y
