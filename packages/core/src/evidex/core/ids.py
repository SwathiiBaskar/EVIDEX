"""Typed, tenant-rooted identifiers.

An identifier is a 128-bit value with a kind. Its text form is ``<prefix>_<26 characters>``
(see :mod:`evidex.core.base32`). Every kind has exactly one *construction mode*:

``ASSIGNED``
    The value is supplied by an authority and merely validated here. ``ten`` is always
    assigned and is the only kind without a parent: **a tenant id can never be derived**.
    A ``doc`` is assigned when it has no stable source key.
``DERIVED``
    The value is a pure function of a parent id and further parts: the first 128 bits of
    SHA-256 over the canonical encoding (:mod:`evidex.core.encoding`) of the scheme tag, the
    kind prefix, the parent and the parts. A ``doc`` with a source key, and every ``rev``,
    ``ext``, ``el``, ``cks`` and ``chk``, are derived. Because each derived kind has a fixed
    parent kind, every derived id descends from a ``ten`` or an assigned ``doc``, and ids
    derived under different tenants never coincide.
``CONTENT``
    ``blob`` ids depend on a content hash alone. They identify stored bytes, not anything a
    tenant owns.

The scheme tag is part of every derivation. Changing the derivation therefore requires a
new scheme tag; the tag in use is :data:`SCHEME`.
"""

from __future__ import annotations

import enum
import hashlib
import re
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, Self, overload

from evidex.core.base32 import RAW_BYTES, Base32Error, decode128, encode128
from evidex.core.encoding import Part, encode_parts
from evidex.core.hashing import ContentHash

SCHEME: Final = "evidex.id.v1"

_TEXT: Final = re.compile(r"([a-z]+)_([0-9a-z]{26})")


class IdError(ValueError):
    """Raised for malformed identifiers and for constructions the registry forbids."""


class Construction(enum.Flag):
    """How the value of an identifier of a given kind comes into being."""

    ASSIGNED = enum.auto()
    DERIVED = enum.auto()
    CONTENT = enum.auto()


class IdKind(enum.Enum):
    """The closed registry of identifier kinds; the value is the text prefix."""

    TEN = "ten"  # tenant
    DOC = "doc"  # logical document
    REV = "rev"  # source revision
    EXT = "ext"  # extracted representation
    EL = "el"  # structural element of an extraction
    CKS = "cks"  # chunk set
    CHK = "chk"  # chunk
    BLOB = "blob"  # stored bytes

    @property
    def prefix(self) -> str:
        return self.value

    @property
    def parent(self) -> IdKind | None:
        """The kind this kind is derived from, or ``None`` for ``ten`` and ``blob``."""
        return _PARENT.get(self)

    @property
    def construction(self) -> Construction:
        return _CONSTRUCTION[self]


_PARENT: Final = {
    IdKind.DOC: IdKind.TEN,
    IdKind.REV: IdKind.DOC,
    IdKind.EXT: IdKind.REV,
    IdKind.EL: IdKind.EXT,
    IdKind.CKS: IdKind.EXT,
    IdKind.CHK: IdKind.CKS,
}

_CONSTRUCTION: Final = {
    IdKind.TEN: Construction.ASSIGNED,
    IdKind.DOC: Construction.ASSIGNED | Construction.DERIVED,
    IdKind.REV: Construction.DERIVED,
    IdKind.EXT: Construction.DERIVED,
    IdKind.EL: Construction.DERIVED,
    IdKind.CKS: Construction.DERIVED,
    IdKind.CHK: Construction.DERIVED,
    IdKind.BLOB: Construction.CONTENT,
}

_BY_PREFIX: Final = {kind.prefix: kind for kind in IdKind}


@dataclass(frozen=True, slots=True, repr=False)
class Id:
    """Base class of the typed identifiers; it cannot be instantiated itself."""

    kind: ClassVar[IdKind]
    raw: bytes

    def __post_init__(self) -> None:
        if type(self) is Id:
            raise TypeError("Id is abstract; use one of the typed id classes")
        if not isinstance(self.raw, bytes) or len(self.raw) != RAW_BYTES:
            raise IdError(f"an identifier is exactly {RAW_BYTES} bytes")

    @property
    def prefix(self) -> str:
        return self.kind.prefix

    def __str__(self) -> str:
        return f"{self.kind.prefix}_{encode128(self.raw)}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse the text form of this class's kind."""
        value = parse_id(text, cls.kind)
        if not isinstance(value, cls):  # pragma: no cover - parse_id already checked the kind
            raise IdError("identifier of the wrong kind")
        return value

    def canonical_part(self) -> tuple[bytes, bytes]:
        """Canonical form used when an id is the parent or a part of a derivation."""
        return b"d", self.prefix.encode("ascii") + b"_" + self.raw


@dataclass(frozen=True, slots=True, repr=False)
class TenantId(Id):
    kind: ClassVar[IdKind] = IdKind.TEN


@dataclass(frozen=True, slots=True, repr=False)
class DocumentId(Id):
    kind: ClassVar[IdKind] = IdKind.DOC


@dataclass(frozen=True, slots=True, repr=False)
class RevisionId(Id):
    kind: ClassVar[IdKind] = IdKind.REV


@dataclass(frozen=True, slots=True, repr=False)
class ExtractionId(Id):
    kind: ClassVar[IdKind] = IdKind.EXT


@dataclass(frozen=True, slots=True, repr=False)
class ElementId(Id):
    kind: ClassVar[IdKind] = IdKind.EL


@dataclass(frozen=True, slots=True, repr=False)
class ChunkSetId(Id):
    kind: ClassVar[IdKind] = IdKind.CKS


@dataclass(frozen=True, slots=True, repr=False)
class ChunkId(Id):
    kind: ClassVar[IdKind] = IdKind.CHK


@dataclass(frozen=True, slots=True, repr=False)
class BlobId(Id):
    kind: ClassVar[IdKind] = IdKind.BLOB


_CLASSES: Final[dict[IdKind, type[Id]]] = {
    IdKind.TEN: TenantId,
    IdKind.DOC: DocumentId,
    IdKind.REV: RevisionId,
    IdKind.EXT: ExtractionId,
    IdKind.EL: ElementId,
    IdKind.CKS: ChunkSetId,
    IdKind.CHK: ChunkId,
    IdKind.BLOB: BlobId,
}


def _digest(kind: IdKind, parent: Id | None, parts: tuple[Part, ...]) -> bytes:
    pieces: list[Part] = [SCHEME, kind.prefix]
    if parent is not None:
        pieces.append(parent)
    pieces.extend(parts)
    return hashlib.sha256(encode_parts(*pieces)).digest()[:RAW_BYTES]


@overload
def derive_id(kind: Literal[IdKind.DOC], parent: TenantId, *parts: Part) -> DocumentId: ...
@overload
def derive_id(kind: Literal[IdKind.REV], parent: DocumentId, *parts: Part) -> RevisionId: ...
@overload
def derive_id(kind: Literal[IdKind.EXT], parent: RevisionId, *parts: Part) -> ExtractionId: ...
@overload
def derive_id(kind: Literal[IdKind.EL], parent: ExtractionId, *parts: Part) -> ElementId: ...
@overload
def derive_id(kind: Literal[IdKind.CKS], parent: ExtractionId, *parts: Part) -> ChunkSetId: ...
@overload
def derive_id(kind: Literal[IdKind.CHK], parent: ChunkSetId, *parts: Part) -> ChunkId: ...
def derive_id(kind: IdKind, parent: Id, *parts: Part) -> Id:
    """Derive the id of ``kind`` from its registry-fixed parent kind and at least one part.

    ``ten`` is never derivable and ``blob`` is derived only from a content hash
    (:func:`derive_blob_id`); both are rejected, and the overloads make the call a static
    type error as well.
    """
    if not isinstance(kind, IdKind):
        raise IdError("kind must be an IdKind")
    if kind is IdKind.TEN:
        raise IdError("a tenant id is assigned, never derived")
    if kind is IdKind.BLOB:
        raise IdError("a blob id comes from a content hash; use derive_blob_id")
    expected = kind.parent
    if expected is None or not kind.construction & Construction.DERIVED:  # pragma: no cover
        raise IdError(f"{kind.prefix!r} ids cannot be derived")
    if not isinstance(parent, Id) or parent.kind is not expected:
        raise IdError(f"a {kind.prefix!r} id is derived from a {expected.prefix!r} parent")
    if not parts:
        raise IdError("a derivation needs at least one part")
    return _CLASSES[kind](_digest(kind, parent, parts))


def derive_blob_id(content: ContentHash) -> BlobId:
    """The id of stored bytes: a function of their content hash and nothing else."""
    if not isinstance(content, ContentHash):
        raise IdError("a blob id is derived from a ContentHash")
    return BlobId(_digest(IdKind.BLOB, None, (content,)))


@overload
def assign_id(kind: Literal[IdKind.TEN], raw: bytes) -> TenantId: ...
@overload
def assign_id(kind: Literal[IdKind.DOC], raw: bytes) -> DocumentId: ...
def assign_id(kind: IdKind, raw: bytes) -> Id:
    """Accept a 128-bit value supplied by an authority as an id of an ASSIGNED kind.

    Nothing is generated here, and nothing proves which tenant an assigned ``doc``
    belongs to: that association is a fact recorded by whoever owns the document.
    """
    if not isinstance(kind, IdKind) or not kind.construction & Construction.ASSIGNED:
        raise IdError("this kind of identifier cannot be assigned")
    return _CLASSES[kind](raw)


def parse_id(text: str, kind: IdKind | None = None) -> Id:
    """Parse any identifier text, optionally requiring a particular kind."""
    match = _TEXT.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise IdError("malformed identifier")
    found = _BY_PREFIX.get(match.group(1))
    if found is None:
        raise IdError(f"unknown identifier prefix {match.group(1)!r}")
    if kind is not None and found is not kind:
        raise IdError(f"expected a {kind.prefix!r} identifier, got {found.prefix!r}")
    try:
        raw = decode128(match.group(2))
    except Base32Error as error:
        raise IdError("malformed identifier") from error
    return _CLASSES[found](raw)
