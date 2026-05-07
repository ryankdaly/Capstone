"""Edge tests: event_rate_limiter — L4."""

import pytest
from conftest import load_artifact

m = load_artifact()

UINT32_MAX = (1 << 32) - 1   # 4294967295


def test_normal_rate_allows_events():
    rl = m.EventRateLimiter(max_events=3, window_ms=1000)
    assert rl.record_event(0) is True
    assert rl.record_event(100) is True
    assert rl.record_event(200) is True


def test_burst_at_limit_all_accepted():
    rl = m.EventRateLimiter(max_events=3, window_ms=1000)
    results = [rl.record_event(i * 10) for i in range(3)]
    assert all(results)


def test_burst_exceeding_limit_rejected():
    rl = m.EventRateLimiter(max_events=3, window_ms=1000)
    for i in range(3):
        rl.record_event(i * 10)
    # 4th event within same window: reject
    assert rl.record_event(50) is False


def test_events_outside_window_evicted():
    rl = m.EventRateLimiter(max_events=3, window_ms=100)
    for i in range(3):
        rl.record_event(i)
    # Jump forward by > window_ms; old events evicted
    assert rl.record_event(200) is True


def test_wraparound_basic():
    rl = m.EventRateLimiter(max_events=5, window_ms=100)
    # Timestamps spanning wraparound: UINT32_MAX - 5 → 10
    ts = [UINT32_MAX - 5, UINT32_MAX - 2, UINT32_MAX, 3, 10]
    results = [rl.record_event(t) for t in ts]
    # All should be accepted (5 events in 16ms span < 100ms window)
    assert all(results)


def test_wraparound_window_boundary():
    rl = m.EventRateLimiter(max_events=2, window_ms=50)
    rl.record_event(UINT32_MAX - 10)
    rl.record_event(UINT32_MAX)
    # Event at ts=3 is 13ms after UINT32_MAX-10 → within window still
    assert rl.record_event(3) is False  # 3rd event exceeds max_events=2


def test_rejected_event_not_recorded():
    rl = m.EventRateLimiter(max_events=1, window_ms=1000)
    rl.record_event(0)
    rl.record_event(100)  # rejected
    # Old event expires; next event should be accepted
    assert rl.record_event(1100) is True
