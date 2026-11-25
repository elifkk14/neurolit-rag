"""In-memory sliding-window rate limiter.

Good enough for one server process (the free demo). With several processes or servers the
counts would have to live in a shared store such as Redis.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class SlidingWindowLimiter:
    """Allow at most `limit` events per `window` seconds for each key."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window
        self.clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()  # /ask runs in worker threads

    def retry_after(self, key: str) -> float:
        """Seconds until `key` may act again; 0 means allowed now (and the event is recorded)."""
        if self.limit <= 0:
            return 0.0
        now = self.clock()
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - self.window:
                events.popleft()
            if len(events) >= self.limit:
                return events[0] + self.window - now
            events.append(now)
            if len(self._events) > 10_000:  # keep memory bounded: forget idle clients
                for k in [k for k, v in self._events.items() if not v]:
                    del self._events[k]
            return 0.0

    def undo(self, key: str) -> None:
        """Remove the last recorded event (used when a later check rejects the request)."""
        with self._lock:
            if self._events.get(key):
                self._events[key].pop()
