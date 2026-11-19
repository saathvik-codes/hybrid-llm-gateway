import pytest

from app.budget import BudgetPolicy


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_no_limit_never_exceeds():
    p = BudgetPolicy(0)
    p.record(1000)
    assert not p.enabled and not p.exceeded() and p.remaining() == -1.0


def test_limit_is_enforced_and_remaining_shrinks():
    p = BudgetPolicy(1.0)
    p.record(0.4)
    assert p.remaining() == pytest.approx(0.6) and not p.exceeded()
    p.record(0.6)
    assert p.exceeded() and p.remaining() == 0.0


def test_spend_resets_at_utc_midnight():
    clock = Clock()
    p = BudgetPolicy(1.0, clock=clock)
    p.record(2.0)
    assert p.exceeded()
    clock.t += 24 * 3600
    assert not p.exceeded() and p.spent_today == 0.0


def test_negative_cost_is_ignored():
    p = BudgetPolicy(1.0)
    p.record(-5)
    assert p.spent_today == 0.0
