"""Edge tests: rate_limiter — L2."""

import pytest
from conftest import load_artifact

m = load_artifact()


def test_negative_max_delta_raises():
    with pytest.raises(ValueError):
        m.make_rate_limiter(-1)


def test_first_call_returns_value_unchanged():
    limiter = m.make_rate_limiter(10)
    assert limiter(500) == 500


def test_ramp_up():
    limiter = m.make_rate_limiter(10)
    limiter(100)       # establish baseline
    assert limiter(200) == 110   # capped at +10


def test_ramp_down():
    limiter = m.make_rate_limiter(10)
    limiter(200)       # establish baseline
    assert limiter(100) == 190   # capped at -10


def test_no_change():
    limiter = m.make_rate_limiter(10)
    limiter(50)
    assert limiter(50) == 50


def test_exact_boundary_step():
    limiter = m.make_rate_limiter(10)
    limiter(100)
    assert limiter(110) == 110   # exactly max_delta; no clamping needed


def test_zero_max_delta_freezes():
    limiter = m.make_rate_limiter(0)
    limiter(100)
    assert limiter(200) == 100   # frozen


def test_ramp_reaches_target():
    limiter = m.make_rate_limiter(10)
    val = limiter(0)
    for _ in range(20):
        val = limiter(100)
    assert val == 100
