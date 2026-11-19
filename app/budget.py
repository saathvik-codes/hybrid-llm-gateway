import datetime as dt
import threading
import time
from typing import Callable


class BudgetPolicy:
    """Daily cloud-spend cap in USD (UTC day). A limit of 0 means no cap."""

    def __init__(self, daily_limit_usd: float = 0.0, clock: Callable[[], float] = time.time):
        self.daily_limit_usd = max(0.0, daily_limit_usd)
        self._clock = clock
        self._lock = threading.Lock()
        self._day = self._today()
        self._spent = 0.0

    def _today(self) -> dt.date:
        return dt.datetime.fromtimestamp(self._clock(), dt.timezone.utc).date()

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._spent = today, 0.0

    def record(self, cost_usd: float) -> None:
        with self._lock:
            self._roll()
            self._spent += max(0.0, cost_usd)

    @property
    def spent_today(self) -> float:
        with self._lock:
            self._roll()
            return self._spent

    @property
    def enabled(self) -> bool:
        return self.daily_limit_usd > 0

    def exceeded(self) -> bool:
        return self.enabled and self.spent_today >= self.daily_limit_usd

    def remaining(self) -> float:
        """USD left today, or -1 when there is no cap."""
        return max(0.0, self.daily_limit_usd - self.spent_today) if self.enabled else -1.0
