from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple


class ScreenGazeMapper:
    """
    Maps raw gaze ratios (x,y) to screen-normalized coordinates using a 5-point calibration
    (center + four corners) via bilinear interpolation.
    """

    def __init__(self, path: str = "scripts/screen_gaze_calibration.json") -> None:
        self.path = Path(path)
        self.data = self._load(path)

    def _load(self, path: str) -> Optional[dict]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def available(self) -> bool:
        return self.data is not None

    def map(self, x: float, y: float) -> Tuple[float, float]:
        """
        Bilinear interpolation using stored calibration points.
        Returns normalized screen coordinates (0..1, 0..1).
        """
        if not self.data:
            return x, y

        d = self.data
        try:
            tl = d["top_left"]
            tr = d["top_right"]
            bl = d["bottom_left"]
            br = d["bottom_right"]
            center = d.get("center")
        except Exception:
            return x, y

        if center:
            x -= (center.get("x", 0.5) - 0.5)
            y -= (center.get("y", 0.5) - 0.5)

        # First rough y using averaged top/bottom rows
        top_y_avg = (tl["y"] + tr["y"]) / 2
        bot_y_avg = (bl["y"] + br["y"]) / 2
        y_norm = clamp01((y - top_y_avg) / max(1e-6, bot_y_avg - top_y_avg))

        # Interpolate x bounds using y_norm
        left_x = lerp(tl["x"], bl["x"], y_norm)
        right_x = lerp(tr["x"], br["x"], y_norm)
        x_norm = clamp01((x - left_x) / max(1e-6, right_x - left_x))

        # Refine y using x_norm
        top_y = lerp(tl["y"], tr["y"], x_norm)
        bot_y = lerp(bl["y"], br["y"], x_norm)
        y_norm = clamp01((y - top_y) / max(1e-6, bot_y - top_y))

        return x_norm, y_norm


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp01(val: float) -> float:
    return max(0.0, min(1.0, val))
