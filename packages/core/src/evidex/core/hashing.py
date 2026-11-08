"""Content hashing: streaming SHA-256 behind a typed, self-describing value."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Final, Protocol, Self

ALGORITHM: Final = "sha256"
DIGEST_BYTES: Final = 32
DEFAULT_CHUNK_SIZE: Final = 1 << 20

_TEXT: Final = re.compile(r"sha256:([0-9a-f]{64})")


class ContentHashError(ValueError):
    """Raised for malformed content hashes and invalid hashing parameters."""


class Readable(Protocol):
    """The part of a binary stream that hashing needs."""

    def read(self, size: int, /) -> bytes: ...


@dataclass(frozen=True, slots=True, repr=False)
class ContentHash:
    """A SHA-256 digest of some bytes, written ``sha256:<64 lower-case hex digits>``."""

    digest: bytes
    algorithm: str = ALGORITHM

    def __post_init__(self) -> None:
        if self.algorithm != ALGORITHM:
            raise ContentHashError("unsupported hash algorithm")
        if not isinstance(self.digest, bytes) or len(self.digest) != DIGEST_BYTES:
            raise ContentHashError(f"a {ALGORITHM} digest is exactly {DIGEST_BYTES} bytes")

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.digest.hex()}"

    def __repr__(self) -> str:
        return f"ContentHash({str(self)!r})"

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse the text form produced by ``str()``; anything else is rejected."""
        match = _TEXT.fullmatch(text) if isinstance(text, str) else None
        if match is None:
            raise ContentHashError("malformed content hash")
        return cls(bytes.fromhex(match.group(1)))

    def canonical_part(self) -> tuple[bytes, bytes]:
        """Canonical form used when a hash is an input to a derived identifier."""
        return b"h", str(self).encode("ascii")


def hash_bytes(data: bytes | bytearray | memoryview) -> ContentHash:
    """Hash a complete in-memory buffer."""
    return ContentHash(hashlib.sha256(data).digest())


def hash_stream(stream: Readable, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ContentHash:
    """Hash a binary stream, never requesting more than ``chunk_size`` bytes at a time."""
    if chunk_size <= 0:
        raise ContentHashError("chunk_size must be positive")
    hasher = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        hasher.update(chunk)
    return ContentHash(hasher.digest())


def hash_file(path: str | os.PathLike[str], *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ContentHash:
    """Hash a file in bounded memory."""
    with open(path, "rb") as stream:
        return hash_stream(stream, chunk_size=chunk_size)
