"""Edge tests: bitfield_extract — L3."""

import pytest
from conftest import load_artifact

m = load_artifact()


def test_one_bit_field_unsigned():
    # bit 0 of 0xFFFFFFFF = 1
    assert m.extract_unsigned(0xFFFFFFFF, 0, 0) == 1


def test_one_bit_field_signed_is_minus_one():
    # 1-bit field value=1 → sign-extended = -1
    assert m.extract_signed(0xFFFFFFFF, 0, 0) == -1


def test_one_bit_field_signed_zero():
    # bit 0 of 0xFFFFFFFE = 0 → sign extended = 0
    assert m.extract_signed(0xFFFFFFFE, 0, 0) == 0


def test_16bit_field_bits_23_to_8():
    # Bits 23:8 of 0x00ABCD00 = 0xABCD
    word = 0x00ABCD00
    assert m.extract_unsigned(word, 23, 8) == 0xABCD


def test_full_32bit_field_unsigned():
    assert m.extract_unsigned(0xDEADBEEF, 31, 0) == 0xDEADBEEF


def test_msb_lt_lsb_raises():
    with pytest.raises((ValueError, AssertionError)):
        m.extract_unsigned(0xFFFFFFFF, 4, 8)  # msb < lsb


def test_bit_index_out_of_range_raises():
    with pytest.raises((ValueError, AssertionError)):
        m.extract_unsigned(0xFFFFFFFF, 32, 0)  # msb=32 out of [0,31]


def test_word_exceeds_32bits_raises():
    with pytest.raises((ValueError, AssertionError)):
        m.extract_unsigned(0x1_0000_0000, 31, 0)  # word > 32-bit
