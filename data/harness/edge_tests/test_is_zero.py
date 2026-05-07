"""Edge tests: is_zero — L1 baseline."""

import pytest
from conftest import load_artifact

m = load_artifact()


def test_zero_returns_true():
    assert m.is_zero(0) is True


def test_positive_returns_false():
    assert m.is_zero(1) is False


def test_negative_returns_false():
    assert m.is_zero(-1) is False


def test_large_positive_returns_false():
    assert m.is_zero(2_147_483_647) is False


def test_large_negative_returns_false():
    assert m.is_zero(-2_147_483_648) is False


def test_return_type_is_bool():
    result = m.is_zero(0)
    assert isinstance(result, bool)
