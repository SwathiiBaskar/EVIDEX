"""Order-preserving base32 text form for 128-bit values.

The alphabet is Crockford's, in lower case, without the letters ``i``, ``l``, ``o`` and
``u``. Its characters are in ascending ASCII order, so comparing the text of two encoded
values gives the same result as comparing their bytes.

128 bits need 26 characters (130 bits). The first character carries only 3 bits and is
therefore always one of ``0``-``7``; the decoder rejects anything else instead of
silently discarding bits.
"""

from __future__ import annotations

from typing import Final

RAW_BYTES: Final = 16
TEXT_CHARS: Final = 26
ALPHABET: Final = "0123456789abcdefghjkmnpqrstvwxyz"

_BITS_PER_CHAR: Final = 5
_VALUES: Final = {char: value for value, char in enumerate(ALPHABET)}


class Base32Error(ValueError):
    """Raised when a value cannot be encoded or decoded."""


def encode128(raw: bytes) -> str:
    """Encode exactly 16 bytes as 26 characters."""
    if not isinstance(raw, bytes) or len(raw) != RAW_BYTES:
        raise Base32Error(f"expected exactly {RAW_BYTES} bytes")
    number = int.from_bytes(raw, "big")
    shifts = range(_BITS_PER_CHAR * (TEXT_CHARS - 1), -1, -_BITS_PER_CHAR)
    return "".join(ALPHABET[(number >> shift) & 0b11111] for shift in shifts)


def decode128(text: str) -> bytes:
    """Decode 26 characters into 16 bytes, rejecting anything that is not canonical."""
    if not isinstance(text, str) or len(text) != TEXT_CHARS:
        raise Base32Error(f"expected exactly {TEXT_CHARS} characters")
    number = 0
    for char in text:
        value = _VALUES.get(char)
        if value is None:
            raise Base32Error("character outside the base32 alphabet")
        number = (number << _BITS_PER_CHAR) | value
    if number >> (RAW_BYTES * 8):
        raise Base32Error("value does not fit in 128 bits")
    return number.to_bytes(RAW_BYTES, "big")
