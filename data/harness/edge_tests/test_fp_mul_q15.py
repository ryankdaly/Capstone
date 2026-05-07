"""Edge tests: fp_mul_q15 (Q15.16 fixed-point multiply) — L4."""

import pytest
from conftest import load_artifact

m = load_artifact()

INT32_MAX = 2_147_483_647
INT32_MIN = -2_147_483_648
SCALE = 1 << 16  # 65536


def fp(x: float) -> int:
    """Convert float to Q15.16."""
    return m.float_to_fp(x)


def tofloat(x: int) -> float:
    return m.fp_to_float(x)


def test_known_vector_1point5_times_2():
    result = m.fp_mul(fp(1.5), fp(2.0))
    assert abs(tofloat(result) - 3.0) < 1e-4


def test_one_times_one():
    result = m.fp_mul(fp(1.0), fp(1.0))
    assert abs(tofloat(result) - 1.0) < 1e-4


def test_multiply_by_zero():
    result = m.fp_mul(fp(12345.0 / SCALE), fp(0.0))
    assert result == 0


def test_positive_overflow_saturates():
    # INT32_MAX * 2.0 should saturate to INT32_MAX
    result = m.fp_mul(INT32_MAX, fp(2.0))
    assert result == INT32_MAX


def test_negative_overflow_saturates():
    # INT32_MIN * 2.0 should saturate to INT32_MIN
    result = m.fp_mul(INT32_MIN, fp(2.0))
    assert result == INT32_MIN


def test_negative_times_positive():
    result = m.fp_mul(fp(-1.5), fp(2.0))
    assert abs(tofloat(result) - (-3.0)) < 1e-4


def test_fractional_multiply():
    result = m.fp_mul(fp(0.5), fp(0.5))
    assert abs(tofloat(result) - 0.25) < 1e-4
