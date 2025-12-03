import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, DefaultDict, Dict, Optional


@dataclass
class Event:
    """Canonical event structure flowing through the system."""

    ts: float
    type: str
    name: str
    confidence: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Lightweight asyncio-based event bus with per-subscriber queues and
    a latest-event cache. Subscribers receive only the event types they request.
    """

    def __init__(self, max_queue_size: int = 256) -> None:
        self._max_queue_size = max_queue_size
        self._subscribers: DefaultDict[str, list[asyncio.Queue[Event]]] = defaultdict(list)
        self._latest: Dict[str, Event] = {}
        self._closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the running event loop for thread-safe publishing."""
        self._loop = loop

    async def publish(self, event: Event | Dict[str, Any]) -> None:
        """Publish an event to all subscribers of its type (and '*' wildcard)."""
        if self._closed:
            return

        normalized = self._normalize_event(event)
        self._latest[normalized.type] = normalized

        for queue in (*self._subscribers.get(normalized.type, []), *self._subscribers.get("*", [])):
            self._offer(queue, normalized)

    def publish_threadsafe(self, event: Event | Dict[str, Any]) -> None:
        """Publish from a non-async context (e.g., keyboard listener thread)."""
        if self._loop is None:
            raise RuntimeError("EventBus.publish_threadsafe called before loop was bound.")
        asyncio.run_coroutine_threadsafe(self.publish(event), self._loop)

    async def subscribe(self, event_type: str = "*") -> AsyncIterator[Event]:
        """
        Subscribe to an event type. Use '*' to receive everything.
        Caller should iterate until cancelled; cleanup happens automatically.
        """
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers[event_type].append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers[event_type].remove(queue)

    def latest(self, event_type: str) -> Optional[Event]:
        """Return the most recent event of a type, if any."""
        return self._latest.get(event_type)

    async def close(self) -> None:
        """Stop accepting new events and clear subscribers."""
        self._closed = True
        self._subscribers.clear()

    def _normalize_event(self, event: Event | Dict[str, Any]) -> Event:
        if isinstance(event, Event):
            return event
        return Event(
            ts=event.get("ts", time.time()),
            type=event["type"],
            name=event["name"],
            confidence=event.get("confidence"),
            meta=event.get("meta") or {},
        )

    def _offer(self, queue: asyncio.Queue[Event], event: Event) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(event)

