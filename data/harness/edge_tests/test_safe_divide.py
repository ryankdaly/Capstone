"""Edge tests: safe_divide — L1."""

import pytest
from conftest import load_artifact

m = load_artifact()

INT_MIN = -2_147_483_648


def test_normal_division():
    assert m.safe_divide(10, 3) == 3  # truncating toward zero


def test_exact_division():
    assert m.safe_divide(12, 4) == 3


def test_negative_dividend():
    assert m.safe_divide(-10, 3) == -3


def test_divide_by_zero_returns_none():
    assert m.safe_divide(100, 0) is None


def test_int_min_by_minus_one_returns_none():
    # C overflow: INT_MIN / -1 is undefined behaviour
    assert m.safe_divide(INT_MIN, -1) is None


def test_zero_numerator():
    assert m.safe_divide(0, 5) == 0


def test_denominator_one():
    assert m.safe_divide(INT_MIN, 1) == INT_MIN
