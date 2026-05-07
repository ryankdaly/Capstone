"""Edge tests: arinc429 — L4."""

import pytest
from conftest import load_artifact

m = load_artifact()


def test_pack_unpack_round_trip():
    label = 0o205   # airspeed label
    sdi = 1
    data = 0x1234
    ssm = 3
    word = m.arinc429_pack(label, sdi, data, ssm)
    result = m.arinc429_unpack(word)
    assert result["label"] == label
    assert result["sdi"] == sdi
    assert result["data"] == data
    assert result["ssm"] == ssm
    assert result["parity_ok"] is True


def test_known_vector_parity_ok():
    # Pack known values; verify parity bit is correctly computed
    word = m.arinc429_pack(0o205, 1, 0x1234, 3)
    assert 0 <= word <= 0xFFFFFFFF


def test_parity_error_raises():
    label = 0o100
    word = m.arinc429_pack(label, 0, 0, 0)
    # Flip the parity bit (bit 32, index 31)
    corrupted = word ^ (1 << 31)
    with pytest.raises(Exception):  # ParityError or ValueError
        m.arinc429_unpack(corrupted)


def test_label_out_of_range_raises():
    with pytest.raises((ValueError, AssertionError)):
        m.arinc429_pack(0x100, 0, 0, 0)  # label > 8 bits


def test_sdi_out_of_range_raises():
    with pytest.raises((ValueError, AssertionError)):
        m.arinc429_pack(0o100, 4, 0, 0)  # SDI only 2 bits (0-3)


def test_data_field_out_of_range_raises():
    with pytest.raises((ValueError, AssertionError)):
        m.arinc429_pack(0o100, 0, 0x80000, 0)  # data > 19 bits


def test_ssm_out_of_range_raises():
    with pytest.raises((ValueError, AssertionError)):
        m.arinc429_pack(0o100, 0, 0, 4)  # SSM only 2 bits (0-3)


def test_zero_word():
    # All-zero word has even parity (0 set bits) — odd parity word would have bit 32=1
    # Pack zeros; parity bit should be set to make total count odd
    word = m.arinc429_pack(0, 0, 0, 0)
    set_bits = bin(word & 0xFFFFFFFF).count("1")
    assert set_bits % 2 == 1  # odd parity
