"""Edge tests: clamp — L1."""

import pytest
from conftest import load_artifact

m = load_artifact()


def test_value_below_lo_returns_lo():
    assert m.clamp(-5, 0, 10) == 0


def test_value_above_hi_returns_hi():
    assert m.clamp(15, 0, 10) == 10


def test_value_at_lo_returns_lo():
    assert m.clamp(0, 0, 10) == 0


def test_value_at_hi_returns_hi():
    assert m.clamp(10, 0, 10) == 10


def test_value_in_range_unchanged():
    assert m.clamp(5, 0, 10) == 5


def test_invalid_bounds_raises():
    with pytest.raises(ValueError):
        m.clamp(5, 10, 0)


def test_lo_equals_hi():
    assert m.clamp(99, 7, 7) == 7


def test_lo_equals_hi_value_below():
    assert m.clamp(3, 7, 7) == 7
