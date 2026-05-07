"""Edge tests: in_range — L1."""

import pytest
from conftest import load_artifact

m = load_artifact()


def test_value_below_lo():
    assert m.in_range(0, 1, 10) is False


def test_value_above_hi():
    assert m.in_range(11, 1, 10) is False


def test_value_at_lo():
    assert m.in_range(1, 1, 10) is True


def test_value_at_hi():
    assert m.in_range(10, 1, 10) is True


def test_value_strictly_between():
    assert m.in_range(5, 1, 10) is True


def test_lo_equals_hi_value_matches():
    assert m.in_range(7, 7, 7) is True


def test_lo_equals_hi_value_misses():
    assert m.in_range(6, 7, 7) is False


def test_invalid_range_raises():
    with pytest.raises(ValueError):
        m.in_range(5, 10, 1)
