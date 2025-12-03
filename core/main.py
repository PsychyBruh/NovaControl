from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Optional

from core.config import AppConfig, KeyboardConfig, load_config
from core.event_bus import EventBus
from actions.safety_guard import SafetyGuard
from actions.mouse_controller import MouseController
from intent.intent_engine import IntentEngine
from vision.hand_tracker import HandTracker
from vision.gaze_tracker import GazeTracker
from ui.preview_overlay import PreviewOverlay
from ui.status_overlay import StatusOverlay

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


class KeyboardHook:
    """Listens for arm/disarm and emergency stop keys and publishes system events."""

    def __init__(self, bus: EventBus, config: KeyboardConfig) -> None:
        self.bus = bus
        self.config = config
        self._listener: Optional[keyboard.Listener] = None if keyboard else None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._armed = False

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(self) -> None:
        if keyboard is None:
            logging.warning("pynput not installed; keyboard hook disabled.")
            return
        if self._loop is None:
            raise RuntimeError("KeyboardHook.start called before loop attachment.")
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        logging.info(
            "Keyboard hook active (arm=%s, emergency=%s)",
            self.config.arm_key,
            self.config.emergency_key,
        )

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if self._matches(key, self.config.emergency_key):
            self._publish("system", "EMERGENCY_STOP")
        elif self._matches(key, self.config.arm_key):
            if self.config.toggle_arm:
                self._armed = not self._armed
                self._publish("mode", "ARMED" if self._armed else "SAFE")
            elif not self._armed:
                self._armed = True
                self._publish("mode", "ARMED")

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if self._matches(key, self.config.arm_key) and self._armed and self.config.safe_on_release:
            self._armed = False
            self._publish("mode", "SAFE")

    def _matches(self, key: keyboard.Key | keyboard.KeyCode, name: str) -> bool:
        target = name.lower()
        if hasattr(key, "char") and key.char:
            return key.char.lower() == target
        return str(key).lower() == f"key.{target}"

    def _publish(self, event_type: str, name: str) -> None:
        payload = {"ts": time.time(), "type": event_type, "name": name}
        try:
            self.bus.publish_threadsafe(payload)
        except RuntimeError as exc:
            logging.error("KeyboardHook publish failed: %s", exc)


class Orchestrator:
    """Bootstraps the event bus, keyboard hook, and graceful shutdown."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.bus = EventBus(max_queue_size=config.event_queue_size)
        self.keyboard_hook = KeyboardHook(self.bus, config.keyboard)
        self._shutdown = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self.safety_guard = SafetyGuard(config.safety)
        self.mouse = MouseController()
        self.intent_engine = IntentEngine(self.bus, self.safety_guard, self.mouse, config)
        self.hand_tracker: Optional[HandTracker] = None
        self.gaze_tracker: Optional[GazeTracker] = None
        self.status_overlay: Optional[StatusOverlay] = None

    def request_shutdown(self) -> None:
        self._shutdown.set()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.bus.bind_loop(loop)
        self.keyboard_hook.attach_loop(loop)
        self.keyboard_hook.start()

        # Start intent engine
        await self.intent_engine.start()

        # Attach trackers (they publish into the bus)
        self.hand_tracker = HandTracker(bus=self.bus)
        self.gaze_tracker = GazeTracker(bus=self.bus)

        self._tasks.append(asyncio.create_task(self._log_events()))
        self._tasks.append(asyncio.create_task(self._watch_for_stop_events()))
        if self.config.overlay.preview_enabled:
            self._tasks.append(asyncio.create_task(asyncio.to_thread(self._run_overlay)))
        if self.config.overlay.status_overlay_enabled:
            self.status_overlay = StatusOverlay(self.bus)
            self.status_overlay.start()

        logging.info("Orchestrator started.")

    async def _log_events(self) -> None:
        async for event in self.bus.subscribe("*"):
            logging.debug("Event: %s | %s | conf=%s", event.type, event.name, event.confidence)

    async def _watch_for_stop_events(self) -> None:
        async for event in self.bus.subscribe("system"):
            if event.name == "EMERGENCY_STOP":
                logging.warning("Emergency stop received; shutting down.")
                self.request_shutdown()
                break

    async def stop(self) -> None:
        self.keyboard_hook.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.intent_engine.stop()
        await self.bus.close()
        if self.status_overlay:
            self.status_overlay.stop()
        logging.info("Orchestrator stopped.")

    async def run(self) -> None:
        await self.start()
        try:
            await self._shutdown.wait()
        finally:
            await self.stop()

    def _run_overlay(self) -> None:
        if not self.hand_tracker or not self.gaze_tracker:
            return
        overlay = PreviewOverlay(
            self.config.overlay,
            hand_tracker=self.hand_tracker,
            gaze_tracker=self.gaze_tracker,
            bus=self.bus,
        )
        try:
            overlay.run()
        except Exception as exc:
            logging.error("Overlay exited with error: %s", exc)


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = load_config()
    orchestrator = Orchestrator(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, orchestrator.request_shutdown)
        except NotImplementedError:
            # Windows may not support signal handlers for asyncio
            pass

    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
