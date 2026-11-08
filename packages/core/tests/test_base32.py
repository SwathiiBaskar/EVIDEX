from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evidex.core.base32 import (
    ALPHABET,
    RAW_BYTES,
    TEXT_CHARS,
    Base32Error,
    decode128,
    encode128,
)

PROPERTY = settings(max_examples=300, deadline=None)
RAW = st.binary(min_size=RAW_BYTES, max_size=RAW_BYTES)


def test_alphabet_is_ascending_and_excludes_ambiguous_letters() -> None:
    assert list(ALPHABET) == sorted(ALPHABET)
    assert len(set(ALPHABET)) == 32
    assert not set("ilou") & set(ALPHABET)


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        (bytes(16), "0" * 26),
        (b"\xff" * 16, "7" + "z" * 25),
        (bytes(15) + b"\x01", "0" * 25 + "1"),
        (b"\x80" + bytes(15), "4" + "0" * 25),
        (bytes(15) + b"\xff", "0" * 24 + "7z"),
    ],
)
def test_known_answers(raw: bytes, text: str) -> None:
    assert encode128(raw) == text
    assert decode128(text) == raw


@PROPERTY
@given(raw=RAW)
def test_round_trip(raw: bytes) -> None:
    assert decode128(encode128(raw)) == raw


@PROPERTY
@given(raw=RAW)
def test_length_and_leading_character(raw: bytes) -> None:
    text = encode128(raw)
    assert len(text) == TEXT_CHARS
    assert text[0] in "01234567"
    assert set(text) <= set(ALPHABET)


@PROPERTY
@given(first=RAW, second=RAW)
def test_text_order_equals_byte_order(first: bytes, second: bytes) -> None:
    assert (first < second) == (encode128(first) < encode128(second))
    assert (first == second) == (encode128(first) == encode128(second))


@pytest.mark.parametrize("length", [0, 1, 15, 17, 32])
def test_encoder_rejects_wrong_byte_length(length: int) -> None:
    with pytest.raises(Base32Error):
        encode128(bytes(length))


def test_encoder_rejects_non_bytes() -> None:
    with pytest.raises(Base32Error):
        encode128(bytearray(16))  # type: ignore[arg-type]


@pytest.mark.parametrize("length", [0, 1, 25, 27, 52])
def test_decoder_rejects_wrong_length(length: int) -> None:
    with pytest.raises(Base32Error):
        decode128("0" * length)


@pytest.mark.parametrize(
    "bad", ["i", "l", "o", "u", "A", "Z", "-", "_", " ", "\n", "\u00e9", "8", "9"]
)
def test_decoder_rejects_bad_characters_and_overflow(bad: str) -> None:
    # "8" and "9" are in the alphabet but overflow when they lead; elsewhere they are fine.
    text = bad + "0" * (TEXT_CHARS - 1)
    if bad in "89":
        with pytest.raises(Base32Error):
            decode128(text)
        assert decode128("0" + bad + "0" * (TEXT_CHARS - 2))
    else:
        with pytest.raises(Base32Error):
            decode128(text)
        with pytest.raises(Base32Error):
            decode128("0" * (TEXT_CHARS - 1) + bad)


def test_decoder_rejects_uppercase_form_of_a_valid_value() -> None:
    text = encode128(bytes(range(16)))
    assert any(char.isalpha() for char in text)
    with pytest.raises(Base32Error):
        decode128(text.upper())


def test_decoder_rejects_non_string() -> None:
    with pytest.raises(Base32Error):
        decode128(b"0" * 26)  # type: ignore[arg-type]
