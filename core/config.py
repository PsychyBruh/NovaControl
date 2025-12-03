from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class KeyboardConfig:
    arm_key: str = "space"
    emergency_key: str = "esc"
    safe_on_release: bool = False  # If True, SAFE on key release (hold-to-arm behavior)
    toggle_arm: bool = True  # If True, pressing arm key toggles ARMED/SAFE


@dataclass
class OverlayConfig:
    camera_index: int = 0
    preview_enabled: bool = True
    window_title: str = "NovaControl Overlay"
    target_fps: int = 30
    box_size: Tuple[int, int] = (640, 360)
    allow_resize: bool = True
    mirror: bool = True  # Flip camera horizontally so movements feel natural
    capture_size: Tuple[int, int] | None = None  # None = use camera default/max
    status_overlay_enabled: bool = True  # Small always-on-top status pill


@dataclass
class SafetyConfig:
    default_mode: str = "SAFE"
    click_cooldown_ms: int = 250
    drag_hold_ms: int = 600
    auto_disarm_on_tracking_loss: bool = True
    min_confidence: float = 0.6


@dataclass
class LoggingConfig:
    enable_metrics: bool = True
    metrics_interval_sec: float = 5.0
    log_dir: str = "logs"


@dataclass
class AppConfig:
    event_queue_size: int = 256
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config() -> AppConfig:
    """
    Centralized factory for config.
    Hook env var overrides or file-based overrides here later.
    """
    return AppConfig()
