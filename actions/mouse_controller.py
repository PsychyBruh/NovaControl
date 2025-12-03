from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    from pynput import mouse
except ImportError:
    mouse = None


class MouseController:
    """Thin wrapper over pynput mouse control with basic helpers."""

    def __init__(self, calibration: Optional[dict] = None, calibration_path: str = "scripts/workspace_calibration.json") -> None:
        self._mouse = mouse.Controller() if mouse else None
        self._dragging = False
        self._calibration = calibration or self._load_calibration(calibration_path)

    def move_absolute_norm(self, x_norm: float, y_norm: float) -> None:
        if not self._mouse:
            return

        x_norm, y_norm = self._apply_calibration(x_norm, y_norm)

        size = self._screen_size()
        if not size:
            return
        w, h = size
        x = max(0, min(int(x_norm * w), w - 1))
        y = max(0, min(int(y_norm * h), h - 1))
        self._mouse.position = (x, y)

    def move_relative(self, dx: float, dy: float) -> None:
        if not self._mouse:
            return
        x, y = self._mouse.position
        self._mouse.position = (x + dx, y + dy)

    def click(self, button: str = "left") -> None:
        if not self._mouse:
            return
        btn = self._button(button)
        if btn:
            self._mouse.click(btn, 1)

    def double_click(self, button: str = "left") -> None:
        if not self._mouse:
            return
        btn = self._button(button)
        if btn:
            self._mouse.click(btn, 2)

    def right_click(self) -> None:
        self.click("right")

    def scroll(self, dy: float) -> None:
        if not self._mouse:
            return
        self._mouse.scroll(0, dy)

    def drag_start(self, button: str = "left") -> None:
        if not self._mouse or self._dragging:
            return
        btn = self._button(button)
        if btn:
            self._mouse.press(btn)
            self._dragging = True

    def drag_stop(self, button: str = "left") -> None:
        if not self._mouse or not self._dragging:
            return
        btn = self._button(button)
        if btn:
            self._mouse.release(btn)
            self._dragging = False

    def _button(self, name: str):
        if not mouse:
            return None
        name = name.lower()
        if name == "left":
            return mouse.Button.left
        if name == "right":
            return mouse.Button.right
        if name == "middle":
            return mouse.Button.middle
        return None

    def _screen_size(self) -> Optional[Tuple[int, int]]:
        if sys.platform.startswith("win"):
            try:
                import ctypes

                user32 = ctypes.windll.user32
                return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            except Exception:
                return None
        # Fallback: try to get current position and assume 1920x1080
        return (1920, 1080)

    def _apply_calibration(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        x_norm = self._clamp01(x_norm)
        y_norm = self._clamp01(y_norm)
        if not self._calibration:
            return x_norm, y_norm
        try:
            min_x = float(self._calibration["min_x"])
            max_x = float(self._calibration["max_x"])
            min_y = float(self._calibration["min_y"])
            max_y = float(self._calibration["max_y"])
        except Exception:
            return x_norm, y_norm

        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        adj_x = self._clamp01((x_norm - min_x) / span_x)
        adj_y = self._clamp01((y_norm - min_y) / span_y)
        return adj_x, adj_y

    def _load_calibration(self, path: str) -> Optional[dict]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _clamp01(self, val: float) -> float:
        return max(0.0, min(1.0, val))
