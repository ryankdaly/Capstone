"""Edge tests: tmr_vote — L3."""

import pytest
from conftest import load_artifact

m = load_artifact()


def test_all_agree_exact():
    value, ok = m.tmr_vote(5, 5, 5)
    assert ok is True
    assert value == 5


def test_first_two_agree():
    value, ok = m.tmr_vote(5, 5, 99)
    assert ok is True
    assert value == 5


def test_last_two_agree():
    value, ok = m.tmr_vote(99, 5, 5)
    assert ok is True
    assert value == 5


def test_first_and_last_agree():
    value, ok = m.tmr_vote(5, 99, 5)
    assert ok is True
    assert value == 5


def test_none_agree_returns_false():
    value, ok = m.tmr_vote(1, 2, 3)
    assert ok is False
    assert value == 0  # safe sentinel


def test_tolerance_boundary_exact():
    # |10 - 12| = 2 = tolerance → agree
    value, ok = m.tmr_vote(10, 12, 99, tolerance=2)
    assert ok is True


def test_tolerance_exceeded_by_one():
    # |10 - 12| = 2 > tolerance=1 → that pair doesn't agree
    # Also check other pairs: |10-99|=89 > 1, |12-99|=87 > 1 → none agree
    value, ok = m.tmr_vote(10, 12, 99, tolerance=1)
    assert ok is False


def test_negative_tolerance_raises():
    with pytest.raises((ValueError, AssertionError)):
        m.tmr_vote(1, 2, 3, tolerance=-1)
