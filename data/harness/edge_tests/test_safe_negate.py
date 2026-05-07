"""Edge tests: safe_negate — L1."""

import pytest
from conftest import load_artifact

m = load_artifact()

INT_MIN = -2_147_483_648
INT_MAX = 2_147_483_647


def test_zero():
    assert m.safe_negate(0) == 0


def test_positive_one():
    assert m.safe_negate(1) == -1


def test_negative_one():
    assert m.safe_negate(-1) == 1


def test_int_max():
    assert m.safe_negate(INT_MAX) == -INT_MAX


def test_int_min_raises():
    with pytest.raises(OverflowError):
        m.safe_negate(INT_MIN)


def test_int_min_plus_one():
    # INT_MIN + 1 = -2147483647; negation = 2147483647 = INT_MAX; safe
    assert m.safe_negate(INT_MIN + 1) == INT_MAX
