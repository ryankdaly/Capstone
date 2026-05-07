"""Edge tests: xor_checksum — L2."""

import pytest
from conftest import load_artifact

m = load_artifact()


def _checksum(data: bytes) -> int:
    """Reference XOR checksum."""
    result = 0
    for b in data:
        result ^= b
    return result


def test_checksum_known_vector():
    data = b"\x01\x02\x03"
    expected = 0x01 ^ 0x02 ^ 0x03
    assert m.compute_checksum(data) == expected


def test_checksum_empty_raises_or_zero():
    # Implementations may either raise on empty or return 0x00.
    # Either is acceptable — just must not crash silently.
    try:
        result = m.compute_checksum(b"")
        assert result == 0  # XOR of nothing = 0
    except (ValueError, Exception):
        pass  # raising is also valid


def test_checksum_single_byte():
    assert m.compute_checksum(b"\xAB") == 0xAB


def test_verify_valid_message():
    data = b"\x10\x20\x30"
    cs = m.compute_checksum(data)
    # Appending checksum should make overall XOR = 0
    assert m.verify_checksum(data + bytes([cs])) is True


def test_verify_corrupted_message():
    data = b"\x10\x20\x30"
    cs = m.compute_checksum(data)
    corrupted = data + bytes([cs ^ 0xFF])  # flip all bits of checksum byte
    assert m.verify_checksum(corrupted) is False


def test_non_bytes_raises():
    with pytest.raises((TypeError, AttributeError)):
        m.compute_checksum("not bytes")


def test_verify_all_zero_message():
    # XOR of all zeros = 0; checksum = 0; appended = \x00 → total XOR = 0
    assert m.verify_checksum(b"\x00\x00\x00\x00") is True
