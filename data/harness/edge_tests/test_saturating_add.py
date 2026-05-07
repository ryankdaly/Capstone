"""Edge tests: saturating_add — L2."""

import pytest
from conftest import load_artifact

m = load_artifact()

INT_MAX = 2_147_483_647
INT_MIN = -2_147_483_648


def test_normal_addition():
    assert m.saturating_add(100, 200) == 300


def test_positive_overflow_saturates():
    assert m.saturating_add(INT_MAX, 1) == INT_MAX


def test_negative_overflow_saturates():
    assert m.saturating_add(INT_MIN, -1) == INT_MIN


def test_int_max_plus_int_max():
    assert m.saturating_add(INT_MAX, INT_MAX) == INT_MAX


def test_int_min_plus_int_min():
    assert m.saturating_add(INT_MIN, INT_MIN) == INT_MIN


def test_add_zero_to_int_max():
    assert m.saturating_add(INT_MAX, 0) == INT_MAX


def test_add_zero_to_int_min():
    assert m.saturating_add(INT_MIN, 0) == INT_MIN


def test_result_at_exact_int_max():
    # INT_MAX - 1 + 1 = INT_MAX exactly, no saturation
    assert m.saturating_add(INT_MAX - 1, 1) == INT_MAX


def test_negative_numbers_in_range():
    assert m.saturating_add(-100, -200) == -300
