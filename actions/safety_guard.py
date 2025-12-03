from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from core.config import SafetyConfig
from core.event_bus import Event


@dataclass
class SafetyState:
    armed: bool = False
    emergency: bool = False
    last_click_ts: float = 0.0
    last_scroll_ts: float = 0.0
    drag_active: bool = False


class SafetyGuard:
    """Safety gating for intents with arming, cooldowns, and emergency stop."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self.state = SafetyState()

    def set_mode(self, mode: str) -> None:
        if mode.upper() == "ARMED":
            self.state.armed = True
        elif mode.upper() == "SAFE":
            self.state.armed = False

    def emergency_stop(self) -> None:
        self.state.emergency = True
        self.state.armed = False

    def clear_emergency(self) -> None:
        self.state.emergency = False

    def notify_tracking_lost(self) -> None:
        if self.config.auto_disarm_on_tracking_loss:
            self.state.armed = False

    def approve(self, intent: Event) -> bool:
        if self.state.emergency:
            return False
        if intent.confidence is not None and intent.confidence < self.config.min_confidence:
            return False

        name = intent.name.upper()
        now = time.time()

        # Only allow actions while armed.
        if name in {"CURSOR_MOVE", "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "DRAG_START", "DRAG_STOP", "SCROLL"}:
            if not self.state.armed:
                return False

        if name in {"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"}:
            if (now - self.state.last_click_ts) * 1000 < self.config.click_cooldown_ms:
                return False
            self.state.last_click_ts = now

        if name == "SCROLL":
            if (now - self.state.last_scroll_ts) * 1000 < self.config.click_cooldown_ms:
                return False
            self.state.last_scroll_ts = now

        if name == "DRAG_START":
            self.state.drag_active = True
        if name == "DRAG_STOP":
            self.state.drag_active = False

        return True
