"""Canonical, injective encoding of the parts an identifier is derived from.

Every part is written as ``tag || length || payload``: a one-byte type tag, an 8-byte
big-endian payload length, then the payload. Each element is therefore self-delimiting,
which makes the concatenation injective: two different sequences of parts can never
produce the same bytes, whatever their contents. The tag separates types that would
otherwise share a payload (``"1"`` versus ``b"1"`` versus ``1``).

Strings are encoded as UTF-8 exactly as given. No Unicode normalisation is applied:
callers decide whether two spellings of a name are the same name.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

_LENGTH_BYTES: Final = 8
_STR: Final = b"s"
_BYTES: Final = b"b"
_INT: Final = b"i"
_BUILTIN_TAGS: Final = frozenset({_STR, _BYTES, _INT})


@runtime_checkable
class Encodable(Protocol):
    """A value that supplies its own canonical ``(tag, payload)`` form."""

    def canonical_part(self) -> tuple[bytes, bytes]: ...


type Part = str | bytes | int | Encodable


def encode_parts(*parts: Part) -> bytes:
    """Return the canonical byte encoding of ``parts``.

    ``bool``, ``float``, ``None`` and every other unsupported type raise ``TypeError``.
    Integers are two's complement, big-endian, with a length derived only from the value.
    """
    encoded = bytearray()
    for part in parts:
        tag, payload = _tagged(part)
        encoded += tag
        encoded += len(payload).to_bytes(_LENGTH_BYTES, "big")
        encoded += payload
    return bytes(encoded)


def _tagged(part: object) -> tuple[bytes, bytes]:
    # bool is a subclass of int, so it has to be rejected before the int case.
    if isinstance(part, bool):
        raise TypeError("bool is not a valid identity part")
    if isinstance(part, str):
        return _STR, part.encode("utf-8")
    if isinstance(part, bytes):
        return _BYTES, part
    if isinstance(part, int):
        return _INT, part.to_bytes((part.bit_length() + 8) // 8, "big", signed=True)
    if isinstance(part, Encodable):
        tag, payload = part.canonical_part()
        if len(tag) != 1 or tag in _BUILTIN_TAGS:
            raise ValueError("a custom part needs a one-byte tag that is not a built-in tag")
        return tag, payload
    raise TypeError(f"unsupported identity part type: {type(part).__name__}")
