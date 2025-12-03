from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from core.event_bus import Event, EventBus

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


@dataclass
class GazeResult:
    direction: str
    confidence: float
    ratio: Optional[float]
    landmarks: Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList]


class GazeTracker:
    """
    MediaPipe FaceMesh wrapper to estimate gaze direction (L/C/R) with smoothing.
    Publishes gaze events and SAFE mode when face is lost.
    """

    def __init__(
        self,
        bus: Optional[EventBus] = None,
        smoothing: float = 0.25,
        lost_face_threshold: int = 10,
    ) -> None:
        self.bus = bus
        self.alpha = smoothing
        self.lost_face_threshold = lost_face_threshold
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._ratio_ema: Optional[float] = None
        self._y_ema: Optional[float] = None
        self._no_face_frames = 0
        self._auto_stretch = True
        self._stretch_decay = 0.995
        self._min_span = 0.1
        self._range_min = [0.25, 0.25]
        self._range_max = [0.75, 0.75]

    def process(self, frame_bgr) -> GazeResult:
        start_ts = time.time()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        landmarks = result.multi_face_landmarks[0] if result.multi_face_landmarks else None

        if landmarks is None:
            self._no_face_frames += 1
            if self._no_face_frames >= self.lost_face_threshold:
                self._publish(Event(ts=start_ts, type="mode", name="SAFE", confidence=1.0))
            return GazeResult("NONE", 0.0, None, None)

        self._no_face_frames = 0
        ratio = self._compute_iris_ratio(landmarks)
        vertical = self._compute_vertical_ratio(landmarks)
        if ratio is None:
            return GazeResult("NONE", 0.0, None, landmarks)

        ratio, vertical = self._stretch(ratio, vertical)

        smoothed_ratio = self._smooth_ratio(ratio)
        smoothed_y = self._smooth_y(vertical if vertical is not None else 0.5)
        direction, confidence = self._direction_from_ratio(smoothed_ratio)
        self._publish(
            Event(
                ts=start_ts,
                type="gaze",
                name=direction,
                confidence=confidence,
                meta={"ratio": smoothed_ratio, "y_ratio": smoothed_y},
            )
        )
        return GazeResult(direction, confidence, smoothed_ratio, landmarks)

    def close(self) -> None:
        self._face_mesh.close()

    # Helpers -------------------------------------------------------------
    def _compute_iris_ratio(
        self, landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList
    ) -> Optional[float]:
        pts = landmarks.landmark
        left_outer, left_inner = pts[33], pts[133]
        right_outer, right_inner = pts[362], pts[263]
        left_iris = pts[468]
        right_iris = pts[473]

        def _eye_ratio(outer, inner, iris):
            span = inner.x - outer.x
            if span == 0:
                return None
            return (iris.x - outer.x) / span

        left_ratio = _eye_ratio(left_outer, left_inner, left_iris)
        right_ratio = _eye_ratio(right_outer, right_inner, right_iris)
        ratios = [r for r in (left_ratio, right_ratio) if r is not None]
        if not ratios:
            return None
        return float(np.clip(np.mean(ratios), 0.0, 1.0))

    def _compute_vertical_ratio(
        self, landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList
    ) -> Optional[float]:
        pts = landmarks.landmark
        # Eye top/bottom landmarks (MediaPipe indices).
        left_top, left_bottom = pts[159], pts[145]
        right_top, right_bottom = pts[386], pts[374]
        left_iris = pts[468]
        right_iris = pts[473]

        def _v_ratio(top, bottom, iris):
            span = bottom.y - top.y
            if span == 0:
                return None
            return (iris.y - top.y) / span

        left_ratio = _v_ratio(left_top, left_bottom, left_iris)
        right_ratio = _v_ratio(right_top, right_bottom, right_iris)
        ratios = [r for r in (left_ratio, right_ratio) if r is not None]
        if not ratios:
            return None
        return float(np.clip(np.mean(ratios), 0.0, 1.0))

    def _smooth_ratio(self, ratio: float) -> float:
        if self._ratio_ema is None:
            self._ratio_ema = ratio
        else:
            self._ratio_ema = self._ratio_ema * (1 - self.alpha) + ratio * self.alpha
        return self._ratio_ema

    def _smooth_y(self, ratio: float) -> float:
        if self._y_ema is None:
            self._y_ema = ratio
        else:
            self._y_ema = self._y_ema * (1 - self.alpha) + ratio * self.alpha
        return self._y_ema

    def _direction_from_ratio(self, ratio: float) -> Tuple[str, float]:
        if ratio < 0.4:
            return "LEFT", 1.0 - ratio
        if ratio > 0.6:
            return "RIGHT", ratio
        return "CENTER", 1.0 - abs(ratio - 0.5) * 2

    def _stretch(self, x: float, y: Optional[float]) -> Tuple[float, Optional[float]]:
        if not self._auto_stretch:
            return x, y
        if y is None:
            y = 0.5
        decay = self._stretch_decay
        self._range_min[0] = min(self._range_min[0], x)
        self._range_min[1] = min(self._range_min[1], y)
        self._range_max[0] = max(self._range_max[0], x)
        self._range_max[1] = max(self._range_max[1], y)

        self._range_min[0] = self._range_min[0] * decay + x * (1 - decay)
        self._range_min[1] = self._range_min[1] * decay + y * (1 - decay)
        self._range_max[0] = self._range_max[0] * decay + x * (1 - decay)
        self._range_max[1] = self._range_max[1] * decay + y * (1 - decay)

        span_x = max(self._min_span, self._range_max[0] - self._range_min[0])
        span_y = max(self._min_span, self._range_max[1] - self._range_min[1])

        norm_x = float(np.clip((x - self._range_min[0]) / span_x, 0.0, 1.0))
        norm_y = float(np.clip((y - self._range_min[1]) / span_y, 0.0, 1.0))
        return norm_x, norm_y

    def _publish(self, event: Event) -> None:
        if self.bus is None:
            return
        try:
            self.bus.publish_threadsafe(event)
        except Exception:
            pass
