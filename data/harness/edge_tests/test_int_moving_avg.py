"""Edge tests: int_moving_avg — L2."""

import pytest
from conftest import load_artifact

m = load_artifact()

INT16_MAX = 32767
INT16_MIN = -32768


def test_single_value_window_one():
    avg = m.IntMovingAverage(1)
    avg.update(42)
    assert avg.get() == 42


def test_window_two_average():
    avg = m.IntMovingAverage(2)
    avg.update(10)
    avg.update(20)
    assert avg.get() == 15


def test_sliding_window_evicts_old_values():
    avg = m.IntMovingAverage(3)
    avg.update(0)
    avg.update(0)
    avg.update(0)
    avg.update(30)  # evicts first 0
    # window is [0, 0, 30] → avg = 10
    assert avg.get() == 10


def test_all_same_values():
    avg = m.IntMovingAverage(5)
    for _ in range(5):
        avg.update(7)
    assert avg.get() == 7


def test_truncation_not_rounding():
    avg = m.IntMovingAverage(2)
    avg.update(1)
    avg.update(2)
    # (1+2)//2 = 1 (truncation), not 1.5
    assert avg.get() == 1


def test_boundary_sum_no_overflow():
    # Worst case: window=256, all samples=INT16_MAX
    # sum = 256 * 32767 = 8,388,352 which is well within int32
    avg = m.IntMovingAverage(256)
    for _ in range(256):
        avg.update(INT16_MAX)
    assert avg.get() == INT16_MAX


def test_negative_samples():
    avg = m.IntMovingAverage(2)
    avg.update(-10)
    avg.update(-20)
    assert avg.get() == -15
