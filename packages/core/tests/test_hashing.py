from __future__ import annotations

import hashlib
import io
import tracemalloc
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evidex.core import (
    ContentHash,
    ContentHashError,
    hash_bytes,
    hash_file,
    hash_stream,
)

PROPERTY = settings(max_examples=200, deadline=None)

# FIPS 180-4 / NIST example vectors for SHA-256.
EMPTY = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ABC = "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
TWO_BLOCK = "sha256:248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
MILLION_A = "sha256:cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"


class SpyStream:
    """A readable that records the largest read it was asked to serve."""

    def __init__(self, data: bytes) -> None:
        self._inner = io.BytesIO(data)
        self.largest_request = 0

    def read(self, size: int, /) -> bytes:
        self.largest_request = max(self.largest_request, size)
        return self._inner.read(size)


def test_known_answer_vectors() -> None:
    assert str(hash_bytes(b"")) == EMPTY
    assert str(hash_bytes(b"abc")) == ABC
    assert str(hash_bytes(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) == TWO_BLOCK
    assert str(hash_stream(io.BytesIO(b"a" * 1_000_000), chunk_size=65_536)) == MILLION_A


@PROPERTY
@given(data=st.binary(max_size=4096), chunk_size=st.integers(min_value=1, max_value=700))
def test_streaming_equals_one_shot_for_any_chunking(data: bytes, chunk_size: int) -> None:
    assert hash_stream(io.BytesIO(data), chunk_size=chunk_size) == hash_bytes(data)


@PROPERTY
@given(data=st.binary(max_size=2048))
def test_hash_bytes_accepts_any_buffer_type(data: bytes) -> None:
    expected = hash_bytes(data)
    assert hash_bytes(bytearray(data)) == expected
    assert hash_bytes(memoryview(data)) == expected


@PROPERTY
@given(data=st.binary(max_size=512))
def test_text_form_round_trips(data: bytes) -> None:
    value = hash_bytes(data)
    assert ContentHash.parse(str(value)) == value
    assert str(value) == f"sha256:{hashlib.sha256(data).hexdigest()}"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "sha256:",
        EMPTY.upper(),
        "sha256:" + EMPTY.split(":")[1].upper(),  # only the digits are upper case
        EMPTY.replace("sha256", "SHA256"),
        EMPTY[:-1],
        EMPTY + "0",
        " " + EMPTY,
        EMPTY + "\n",
        "md5:" + EMPTY.split(":")[1][:32],
        "sha512:" + EMPTY.split(":")[1],
        EMPTY.split(":")[1],
        "sha256:" + "g" * 64,
    ],
)
def test_malformed_text_is_rejected(text: str) -> None:
    with pytest.raises(ContentHashError):
        ContentHash.parse(text)


def test_non_string_text_is_rejected() -> None:
    with pytest.raises(ContentHashError):
        ContentHash.parse(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("digest", [b"", b"\x00" * 31, b"\x00" * 33])
def test_constructor_rejects_wrong_digest_length(digest: bytes) -> None:
    with pytest.raises(ContentHashError):
        ContentHash(digest)


def test_constructor_rejects_unknown_algorithm() -> None:
    with pytest.raises(ContentHashError):
        ContentHash(b"\x00" * 32, algorithm="md5")


def test_value_semantics() -> None:
    first, second = hash_bytes(b"x"), hash_bytes(b"x")
    assert first == second
    assert hash(first) == hash(second)
    assert {first, second} == {first}
    assert first != hash_bytes(b"y")
    assert repr(first) == f"ContentHash({str(first)!r})"
    with pytest.raises(AttributeError):
        first.digest = b"\x00" * 32  # type: ignore[misc]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size: int) -> None:
    with pytest.raises(ContentHashError):
        hash_stream(io.BytesIO(b"abc"), chunk_size=chunk_size)


def test_stream_reads_never_exceed_chunk_size() -> None:
    spy = SpyStream(bytes(range(256)) * 100)
    hash_stream(spy, chunk_size=1000)
    assert spy.largest_request == 1000


def test_hash_file_matches_hash_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")
    assert str(hash_file(path)) == ABC
    assert hash_file(str(path), chunk_size=1) == hash_bytes(b"abc")


def test_hash_file_uses_bounded_memory(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"\x5a" * (8 * 1024 * 1024))
    tracemalloc.start()
    try:
        result = hash_file(path, chunk_size=4096)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == hash_bytes(path.read_bytes())
    assert peak < 512 * 1024, f"hashing an 8 MiB file peaked at {peak} bytes"


def test_missing_file_raises_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "absent.bin")
