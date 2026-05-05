"""Cost / rate guard for oracle calls.

Two limiters run in series:

* **Rate limit** — at most ``rps`` calls per second (sliding window).
* **Daily budget** — at most ``daily_calls`` per UTC day.

Both can be disabled by setting the limit to ``0`` or a negative number.
Counters live in-process (no Redis/DB) — the guard is intended to protect
a single dev/research box from runaway spend, not to coordinate a fleet.
A simple JSONL audit log is appended for every call so spend can be
inspected after the fact.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque


class OracleQuotaExceeded(RuntimeError):
    """Raised when the rate or daily-budget guard refuses a call."""


class CostGuard:
    """Thread-safe rate + daily-budget limiter with JSONL audit trail."""

    def __init__(
        self,
        *,
        rps: float = 0.5,
        daily_calls: int = 200,
        audit_path: Path | None = None,
    ) -> None:
        self.rps = rps
        self.daily_calls = daily_calls
        self.audit_path = audit_path
        self._lock = threading.Lock()
        self._recent: Deque[float] = deque()
        self._day_key: str = ""
        self._day_count: int = 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def acquire(self, *, reason: str = "") -> None:
        """Block briefly to respect ``rps`` and refuse if daily cap is hit.

        Raises :class:`OracleQuotaExceeded` when the daily budget is
        exhausted; the wake step is expected to swallow the error and
        proceed without oracle augmentation.
        """
        with self._lock:
            now = time.time()
            today = self._today()
            if today != self._day_key:
                self._day_key = today
                self._day_count = 0
            if self.daily_calls > 0 and self._day_count >= self.daily_calls:
                raise OracleQuotaExceeded(
                    f"daily oracle budget exhausted ({self.daily_calls} calls)"
                )

            if self.rps > 0:
                window = 1.0 / self.rps
                # Drop entries older than ``window``.
                while self._recent and now - self._recent[0] > window:
                    self._recent.popleft()
                if self._recent:
                    wait = window - (now - self._recent[0])
                    if wait > 0:
                        # Sleep outside the lock to avoid pinning other
                        # threads. Re-acquire afterwards to update counters.
                        self._lock.release()
                        try:
                            time.sleep(wait)
                        finally:
                            self._lock.acquire()
                        now = time.time()
                self._recent.append(now)

            self._day_count += 1

        # Audit (outside the lock).
        if self.audit_path is not None:
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "reason": reason,
                                "day_count": self._day_count,
                            }
                        )
                        + "\n"
                    )
            except OSError:
                # Audit failures must never break the actual oracle call.
                pass
