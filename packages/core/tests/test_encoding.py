from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evidex.core import IdKind, assign_id, hash_bytes
from evidex.core.encoding import Part, encode_parts

PROPERTY = settings(max_examples=400, deadline=None)

SIMPLE_PART = st.one_of(
    st.text(max_size=8),
    st.binary(max_size=8),
    st.integers(min_value=-(2**70), max_value=2**70),
)


class Custom:
    """A minimal Encodable with a configurable tag."""

    def __init__(self, tag: bytes, payload: bytes = b"") -> None:
        self._tag, self._payload = tag, payload

    def canonical_part(self) -> tuple[bytes, bytes]:
        return self._tag, self._payload


def _first_element(encoded: bytes) -> tuple[bytes, bytes]:
    length = int.from_bytes(encoded[1:9], "big")
    return encoded[:1], encoded[9 : 9 + length]


def test_known_answer_encoding() -> None:
    expected = (
        b"s" + (1).to_bytes(8, "big") + b"a"
        + b"i" + (1).to_bytes(8, "big") + b"\x01"
        + b"b" + (1).to_bytes(8, "big") + b"x"
    )  # fmt: skip
    assert encode_parts("a", 1, b"x") == expected


def test_empty_sequence_encodes_to_nothing() -> None:
    assert encode_parts() == b""


@PROPERTY
@given(first=st.lists(SIMPLE_PART, max_size=5), second=st.lists(SIMPLE_PART, max_size=5))
def test_encoding_is_injective_and_deterministic(first: list[Part], second: list[Part]) -> None:
    assert encode_parts(*first) == encode_parts(*first)
    if first != second:
        assert encode_parts(*first) != encode_parts(*second)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (("ab", "c"), ("a", "bc")),
        ((b"ab", b"c"), (b"a", b"bc")),
        (("", "a"), ("a", "")),
        ((), ("",)),
        (("",), (b"",)),
        (("1",), (b"1",)),
        (("1",), (1,)),
        ((b"\x01",), (1,)),
        ((0,), (b"\x00",)),
        ((0,), ("0",)),
        ((1, 2), (258,)),
        ((255,), (-1,)),
    ],
)
def test_boundary_and_type_ambiguities_are_resolved(
    left: tuple[Part, ...], right: tuple[Part, ...]
) -> None:
    assert encode_parts(*left) != encode_parts(*right)


@PROPERTY
@given(number=st.integers(min_value=-(2**100), max_value=2**100))
def test_integers_decode_back(number: int) -> None:
    tag, payload = _first_element(encode_parts(number))
    assert tag == b"i"
    assert int.from_bytes(payload, "big", signed=True) == number


def test_integer_edge_values_are_distinct() -> None:
    values = [0, 1, -1, 127, 128, 255, 256, -128, -129, 2**64, -(2**64)]
    assert len({encode_parts(value) for value in values}) == len(values)


def test_strings_are_not_normalised() -> None:
    composed, decomposed = "\u00e9", "e\u0301"
    assert encode_parts(composed) != encode_parts(decomposed)
    assert encode_parts(composed) == b"s" + (2).to_bytes(8, "big") + composed.encode("utf-8")


def test_lone_surrogates_are_rejected() -> None:
    with pytest.raises(UnicodeEncodeError):
        encode_parts("\ud800")


@pytest.mark.parametrize(
    "bad", [1.5, float("nan"), True, False, None, [1], ("a",), {"a"}, object()]
)
def test_unsupported_types_are_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        encode_parts(bad)  # type: ignore[arg-type]


def test_content_hash_and_id_never_collide_with_their_text() -> None:
    digest = hash_bytes(b"x")
    tenant = assign_id(IdKind.TEN, bytes(16))
    document = assign_id(IdKind.DOC, bytes(16))
    assert encode_parts(digest) != encode_parts(str(digest))
    assert encode_parts(digest) != encode_parts(digest.digest)
    assert encode_parts(tenant) != encode_parts(str(tenant))
    assert encode_parts(tenant) != encode_parts(tenant.raw)
    assert encode_parts(tenant) != encode_parts(document)  # same bytes, different kind


def test_custom_parts_need_a_free_one_byte_tag() -> None:
    assert encode_parts(Custom(b"x", b"p")) == b"x" + (1).to_bytes(8, "big") + b"p"
    for tag in (b"s", b"b", b"i", b"", b"xx"):
        with pytest.raises(ValueError, match="one-byte tag"):
            encode_parts(Custom(tag))
