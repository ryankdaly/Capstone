"""Edge tests: crc16_ccitt — L3."""

import pytest
from conftest import load_artifact

m = load_artifact()


def test_known_vector():
    # DO NOT change this value — it is the official CRC-16/CCITT test vector
    assert m.crc16_ccitt(b"123456789") == 0x29B1


def test_empty_bytes_with_default_initial():
    # CRC of empty data = initial value unchanged (no data processed)
    result = m.crc16_ccitt(b"")
    assert 0 <= result <= 0xFFFF


def test_single_byte():
    result = m.crc16_ccitt(b"\xFF")
    assert 0 <= result <= 0xFFFF


def test_result_is_16bit():
    result = m.crc16_ccitt(b"hello world")
    assert 0 <= result <= 0xFFFF


def test_initial_value_validation_rejects_out_of_range():
    with pytest.raises((ValueError, OverflowError, TypeError)):
        m.crc16_ccitt(b"data", initial=0x10000)


def test_different_initial_produces_different_result():
    r1 = m.crc16_ccitt(b"test", initial=0xFFFF)
    r2 = m.crc16_ccitt(b"test", initial=0x0000)
    # Different initials should produce different CRCs for non-empty data
    assert r1 != r2


def test_deterministic_for_same_input():
    r1 = m.crc16_ccitt(b"reproducible")
    r2 = m.crc16_ccitt(b"reproducible")
    assert r1 == r2
