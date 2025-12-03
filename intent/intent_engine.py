from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Optional

from actions.mouse_controller import MouseController
from actions.safety_guard import SafetyGuard
from core.config import AppConfig
from core.event_bus import Event, EventBus
from vision.screen_gaze_mapper import ScreenGazeMapper


class IntentEngine:
    """
    Consumes gesture/point/mode events, produces intent events, and executes
    approved actions via SafetyGuard + MouseController.
    """

    def __init__(
        self,
        bus: EventBus,
        safety_guard: SafetyGuard,
        mouse: Optional[MouseController] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.bus = bus
        self.safety_guard = safety_guard
        self.mouse = mouse or MouseController()
        self.config = config or AppConfig()
        self._last_point: Optional[tuple[float, float]] = None
        self._last_gesture: str = "NONE"
        self._pinch_start_ts: Optional[float] = None
        self._drag_active = False
        self._task: Optional[asyncio.Task] = None
        self._fist_start_ts: Optional[float] = None
        self.control_mode: str = "EYE"  # "EYE" or "HAND"
        self._last_gaze_y: float = 0.5
        self._gaze_mapper = ScreenGazeMapper()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(Exception):
                await self._task

    async def _run(self) -> None:
        async for event in self.bus.subscribe("*"):
            self._handle_event(event)

    def _handle_event(self, event: Event) -> None:
        if event.type == "mode":
            self.safety_guard.set_mode(event.name)
        elif event.type == "system" and event.name == "EMERGENCY_STOP":
            self.safety_guard.emergency_stop()
        elif event.type == "gesture":
            self._handle_gesture(event)
        elif event.type == "point":
            self._last_point = (event.meta.get("x_norm"), event.meta.get("y_norm"))
        elif event.type == "gaze":
            self._handle_gaze(event)

    def _handle_gesture(self, event: Event) -> None:
        gesture = event.name.upper()
        now = time.time()

        if gesture == "OPEN_PALM":
            # Toggle control mode and cancel active actions.
            self.control_mode = "HAND" if self.control_mode == "EYE" else "EYE"
            self._emit_intent("CANCEL", event.confidence, {"mode": self.control_mode})
            self._reset_pinch()
            self._reset_fist()
            self._drag_active = False
            self._last_gesture = gesture
            return

        # In eye mode, ignore hand gestures except OPEN_PALM.
        if self.control_mode == "EYE":
            self._last_gesture = gesture
            return

        if gesture == "FIST":
            if self._fist_start_ts is None:
                self._fist_start_ts = now
            hold_ms = (now - self._fist_start_ts) * 1000
            # Short hold: precise click
            if hold_ms >= 100:
                self._emit_intent("CLICK", event.confidence, {"mode": "PRECISE"})
                self._reset_fist()
            self._last_gesture = gesture
            return
        else:
            self._reset_fist()

        if gesture == "PINCH":
            if self._pinch_start_ts is None:
                self._pinch_start_ts = now
            hold_ms = (now - self._pinch_start_ts) * 1000
            if not self._drag_active and hold_ms >= self.config.safety.drag_hold_ms:
                if self._emit_intent("DRAG_START", event.confidence, {}):
                    self._drag_active = True
        else:
            if self._drag_active:
                if self._emit_intent("DRAG_STOP", event.confidence, {}):
                    self._drag_active = False
            elif self._pinch_start_ts is not None and self._last_gesture == "PINCH":
                self._emit_intent("CLICK", event.confidence, {})
            self._reset_pinch()

        if gesture == "POINT" and self._last_point:
            x, y = self._last_point
            self._emit_intent(
                "CURSOR_MOVE",
                event.confidence,
                {"x_norm": x, "y_norm": y},
            )

        self._last_gesture = gesture

    def _handle_gaze(self, event: Event) -> None:
        if self.control_mode != "EYE":
            return
        ratio = event.meta.get("ratio")
        y_ratio = event.meta.get("y_ratio")
        if ratio is None:
            return
        x = float(ratio)
        y = float(y_ratio) if y_ratio is not None else self._last_gaze_y
        if self._gaze_mapper and self._gaze_mapper.available():
            x, y = self._gaze_mapper.map(x, y)
        self._emit_intent(
            "CURSOR_MOVE",
            event.confidence,
            {"x_norm": x, "y_norm": y, "source": "gaze"},
        )
        self._last_gaze_y = y

    def _emit_intent(self, name: str, confidence: Optional[float], meta: dict) -> bool:
        ts = time.time()
        intent = Event(ts=ts, type="intent", name=name, confidence=confidence, meta=meta)
        if not self.safety_guard.approve(intent):
            return False
        try:
            self.bus.publish_threadsafe(intent)
        except Exception:
            pass
        self._execute(intent)
        return True

    def _execute(self, intent: Event) -> None:
        if intent.name == "CURSOR_MOVE":
            x = intent.meta.get("x_norm")
            y = intent.meta.get("y_norm")
            if x is None or y is None:
                if self._last_point:
                    x, y = self._last_point
            if x is not None and y is not None:
                self.mouse.move_absolute_norm(x, y, source=intent.meta.get("source"))
        elif intent.name == "CLICK":
            self.mouse.click()
        elif intent.name == "DOUBLE_CLICK":
            self.mouse.double_click()
        elif intent.name == "RIGHT_CLICK":
            self.mouse.right_click()
        elif intent.name == "DRAG_START":
            self.mouse.drag_start()
        elif intent.name == "DRAG_STOP":
            self.mouse.drag_stop()
        elif intent.name == "SCROLL":
            dy = intent.meta.get("dy", 0.0)
            self.mouse.scroll(dy)
        elif intent.name == "CANCEL":
            self.mouse.drag_stop()

    def _reset_pinch(self) -> None:
        self._pinch_start_ts = None

    def _reset_fist(self) -> None:
        self._fist_start_ts = None
