from __future__ import annotations

import hashlib
import os
import re
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evidex.core import (
    SCHEME,
    ChunkId,
    Construction,
    ContentHash,
    DocumentId,
    Id,
    IdError,
    IdKind,
    RevisionId,
    TenantId,
    assign_id,
    derive_blob_id,
    derive_id,
    hash_bytes,
    ids,
    parse_id,
)
from evidex.core.base32 import ALPHABET, encode128

PROPERTY = settings(max_examples=200, deadline=None)
RAW = st.binary(min_size=16, max_size=16)
NAME = st.text(min_size=1, max_size=12)

TENANT_ONE = assign_id(IdKind.TEN, bytes(range(16)))
TENANT_TWO = assign_id(IdKind.TEN, bytes(range(16, 32)))
HELLO = hash_bytes(b"hello")
PARAGRAPH = hash_bytes(b"first paragraph")

# Known answers for scheme "evidex.id.v1". If any of these changes, identifier derivation
# changed, which silently re-identifies everything derived before: use a new scheme tag.
GOLDEN = {
    "tenant1": "ten_00041061050r3gg28a1c60t3gf",
    "tenant2": "ten_0g2491650n2rbhg68t3ce1t7gz",
    "doc1": "doc_7v3vta71bz4d80p0w8z8gvcs2v",
    "rev1": "rev_6fyq414ewxdm0yjcxdedpjzpb3",
    "ext1": "ext_0z9wwgr3vp4ffhsrtezppmvp1m",
    "el1": "el_66z34hgrv340kvc8rywa5f9dhe",
    "cks1": "cks_6eaf20dhyzwxe1erk2hq7ytkbv",
    "chk1": "chk_5js19hws09p3hxf2be3r7ndgz1",
    "doc2": "doc_5a30f1j0sz054ttzg2svetkm5z",
    "rev2": "rev_6epc6d6bazg9rh1qp1twzqwkz3",
    "ext2": "ext_08fr5wfypjsfgs25rktkpgaqnx",
    "el2": "el_7112bdsv2m70kxk00m62s5znqn",
    "cks2": "cks_229yxdbvcw81pkmerx7zteyths",
    "chk2": "chk_6z371h24q66g12k244fh2wrzc0",
    "blob": "blob_6cv2vd5qf75jeq5apbmzs0cpmk",
}


def _chain(tenant: TenantId, content: bytes = b"hello", ext_config: str = "parser=plain;v=1",
           chunk_config: str = "fixed-token;v=1") -> dict[str, Id]:  # fmt: skip
    doc = derive_id(IdKind.DOC, tenant, "main", "docs/a.pdf")
    rev = derive_id(IdKind.REV, doc, hash_bytes(content))
    ext = derive_id(IdKind.EXT, rev, ext_config)
    element = derive_id(IdKind.EL, ext, "/0/1", "paragraph", PARAGRAPH)
    chunk_set = derive_id(IdKind.CKS, ext, chunk_config, "size=256")
    chunk = derive_id(IdKind.CHK, chunk_set, 0, PARAGRAPH)
    return {"doc": doc, "rev": rev, "ext": ext, "el": element, "cks": chunk_set, "chk": chunk}


# --- registry -----------------------------------------------------------------------------


def test_registry_is_closed_and_prefixes_are_unique() -> None:
    assert {kind.prefix for kind in IdKind} == {
        "ten", "doc", "rev", "ext", "el", "cks", "chk", "blob",
    }  # fmt: skip
    assert len({kind.prefix for kind in IdKind}) == len(IdKind) == 8


def test_construction_modes_and_parents() -> None:
    assert IdKind.TEN.construction == Construction.ASSIGNED
    assert IdKind.DOC.construction == Construction.ASSIGNED | Construction.DERIVED
    assert IdKind.BLOB.construction == Construction.CONTENT
    for kind in (IdKind.REV, IdKind.EXT, IdKind.EL, IdKind.CKS, IdKind.CHK):
        assert kind.construction == Construction.DERIVED
    assert {k for k in IdKind if k.parent is None} == {IdKind.TEN, IdKind.BLOB}
    assert IdKind.TEN.parent is None
    assert IdKind.DOC.parent is IdKind.TEN
    assert IdKind.EL.parent is IdKind.EXT
    assert IdKind.CKS.parent is IdKind.EXT
    assert IdKind.CHK.parent is IdKind.CKS


def test_every_derived_kind_descends_from_a_tenant() -> None:
    for kind in IdKind:
        if not kind.construction & Construction.DERIVED:
            continue
        seen, current = {kind}, kind
        while current.parent is not None:
            current = current.parent
            assert current not in seen
            seen.add(current)
        assert current is IdKind.TEN


# --- derivation ---------------------------------------------------------------------------


def test_golden_ids_for_the_current_scheme() -> None:
    assert SCHEME == "evidex.id.v1"
    one, two = _chain(TENANT_ONE), _chain(TENANT_TWO)
    actual = {"tenant1": TENANT_ONE, "tenant2": TENANT_TWO, "blob": derive_blob_id(HELLO)}
    actual |= {f"{level}1": value for level, value in one.items()}
    actual |= {f"{level}2": value for level, value in two.items()}
    assert {name: str(value) for name, value in actual.items()} == GOLDEN


def _reference_id(prefix: str, *elements: bytes) -> str:
    """The scheme written out independently of the library: stdlib only."""

    def element(tag: bytes, payload: bytes) -> bytes:
        return tag + struct.pack(">Q", len(payload)) + payload

    material = element(b"s", SCHEME.encode()) + element(b"s", prefix.encode())
    digest = hashlib.sha256(material + b"".join(elements)).digest()[:16]
    return f"{prefix}_{encode128(digest)}"


@PROPERTY
@given(
    tenant=RAW,
    collection=st.text(max_size=10),
    key=st.text(max_size=10),
    content=st.binary(max_size=32),
)
def test_derivation_matches_an_independent_reference(
    tenant: bytes, collection: str, key: str, content: bytes
) -> None:
    def part(tag: bytes, payload: bytes) -> bytes:
        return tag + struct.pack(">Q", len(payload)) + payload

    tenant_id = assign_id(IdKind.TEN, tenant)
    doc = derive_id(IdKind.DOC, tenant_id, collection, key)
    expected_doc = _reference_id(
        "doc",
        part(b"d", b"ten_" + tenant),
        part(b"s", collection.encode()),
        part(b"s", key.encode()),
    )
    assert str(doc) == expected_doc
    rev = derive_id(IdKind.REV, doc, hash_bytes(content))
    content_text = f"sha256:{hashlib.sha256(content).hexdigest()}".encode()
    assert str(rev) == _reference_id("rev", part(b"d", b"doc_" + doc.raw), part(b"h", content_text))


def test_derivation_is_deterministic() -> None:
    assert _chain(TENANT_ONE) == _chain(TENANT_ONE)


def test_kind_separates_domains() -> None:
    doc = derive_id(IdKind.DOC, TENANT_ONE, "collection", "key")
    ext = derive_id(IdKind.EXT, derive_id(IdKind.REV, doc, HELLO), "config")
    assert derive_id(IdKind.EL, ext, "x").raw != derive_id(IdKind.CKS, ext, "x").raw


def test_scheme_tag_is_part_of_every_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _chain(TENANT_ONE)
    before_blob = derive_blob_id(HELLO)
    monkeypatch.setattr(ids, "SCHEME", "evidex.id.v2")
    after = _chain(TENANT_ONE)
    assert all(before[level] != after[level] for level in before)
    assert before_blob != derive_blob_id(HELLO)


def test_parent_and_part_order_matter() -> None:
    doc_one = derive_id(IdKind.DOC, TENANT_ONE, "c", "k1")
    doc_two = derive_id(IdKind.DOC, TENANT_ONE, "c", "k2")
    assert derive_id(IdKind.REV, doc_one, HELLO) != derive_id(IdKind.REV, doc_two, HELLO)
    assert derive_id(IdKind.DOC, TENANT_ONE, "a", "b") != derive_id(
        IdKind.DOC, TENANT_ONE, "b", "a"
    )
    assert derive_id(IdKind.DOC, TENANT_ONE, "ab", "c") != derive_id(
        IdKind.DOC, TENANT_ONE, "a", "bc"
    )
    assert len({derive_id(IdKind.DOC, TENANT_ONE, p) for p in ("1", 1, b"1")}) == 3


@PROPERTY
@given(
    first=st.lists(
        st.one_of(st.text(max_size=6), st.binary(max_size=6), st.integers()), min_size=1, max_size=4
    ),
    second=st.lists(
        st.one_of(st.text(max_size=6), st.binary(max_size=6), st.integers()), min_size=1, max_size=4
    ),
)
def test_distinct_parts_give_distinct_ids(
    first: list[str | bytes | int], second: list[str | bytes | int]
) -> None:
    if first != second:
        assert derive_id(IdKind.DOC, TENANT_ONE, *first) != derive_id(
            IdKind.DOC, TENANT_ONE, *second
        )


def test_wrong_parent_kind_is_rejected() -> None:
    chain = _chain(TENANT_ONE)
    parents: dict[str, Id] = {"ten": TENANT_ONE, **chain}
    expected_parent = {
        IdKind.DOC: "ten", IdKind.REV: "doc", IdKind.EXT: "rev",
        IdKind.EL: "ext", IdKind.CKS: "ext", IdKind.CHK: "cks",
    }  # fmt: skip
    for kind, right in expected_parent.items():
        for name, parent in parents.items():
            if name == right:
                derive_id(kind, parent, "part")  # type: ignore[call-overload]
            else:
                with pytest.raises(IdError, match="is derived from a"):
                    derive_id(kind, parent, "part")  # type: ignore[call-overload]


def test_derivation_needs_a_parent_object_and_a_part() -> None:
    with pytest.raises(IdError, match="at least one part"):
        derive_id(IdKind.DOC, TENANT_ONE)
    for bad_parent in (HELLO, "ten_x", None, TENANT_ONE.raw):
        with pytest.raises(IdError):
            derive_id(IdKind.DOC, bad_parent, "x")  # type: ignore[call-overload]
    with pytest.raises(IdError):
        derive_id("doc", TENANT_ONE, "x")  # type: ignore[call-overload]


def test_ten_and_blob_are_not_derivable() -> None:
    with pytest.raises(IdError, match="assigned, never derived"):
        derive_id(IdKind.TEN, TENANT_ONE, "slug")  # type: ignore[call-overload]
    with pytest.raises(IdError, match="derive_blob_id"):
        derive_id(IdKind.BLOB, TENANT_ONE, "x")  # type: ignore[call-overload]


def test_blob_ids_come_only_from_content_hashes() -> None:
    assert derive_blob_id(HELLO) == derive_blob_id(hash_bytes(b"hello"))
    assert derive_blob_id(HELLO) != derive_blob_id(hash_bytes(b"hello!"))
    for bad in (str(HELLO), HELLO.digest, None):
        with pytest.raises(IdError):
            derive_blob_id(bad)  # type: ignore[arg-type]
    assert isinstance(HELLO, ContentHash)


# --- assignment -----------------------------------------------------------------------------


def test_only_ten_and_doc_can_be_assigned() -> None:
    assert isinstance(assign_id(IdKind.TEN, bytes(16)), TenantId)
    assert isinstance(assign_id(IdKind.DOC, bytes(16)), DocumentId)
    for kind in (IdKind.REV, IdKind.EXT, IdKind.EL, IdKind.CKS, IdKind.CHK, IdKind.BLOB):
        with pytest.raises(IdError, match="cannot be assigned"):
            assign_id(kind, bytes(16))  # type: ignore[call-overload]
    with pytest.raises(IdError):
        assign_id("ten", bytes(16))  # type: ignore[call-overload]


@pytest.mark.parametrize("raw", [b"", bytes(15), bytes(17), bytearray(16), "x" * 16, None])
def test_assigned_values_must_be_exactly_16_bytes(raw: object) -> None:
    with pytest.raises(IdError):
        assign_id(IdKind.TEN, raw)  # type: ignore[call-overload]


def test_base_id_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Id(bytes(16))


def test_ids_of_different_kinds_never_compare_equal() -> None:
    tenant: object = assign_id(IdKind.TEN, bytes(16))
    document: object = assign_id(IdKind.DOC, bytes(16))
    assert tenant != document
    assert TenantId(bytes(16)) == TenantId(bytes(16))
    assert len({TenantId(bytes(16)), DocumentId(bytes(16))}) == 2


# --- text form ------------------------------------------------------------------------------


@PROPERTY
@given(raw=RAW)
def test_text_round_trips_for_every_assignable_kind(raw: bytes) -> None:
    for kind in (IdKind.TEN, IdKind.DOC):
        value = assign_id(kind, raw)
        text = str(value)
        assert re.fullmatch(rf"{kind.prefix}_[0-7][{ALPHABET}]{{25}}", text)
        assert parse_id(text) == value
        assert parse_id(text, kind) == value
        assert type(value).parse(text) == value


def test_text_round_trips_for_derived_and_blob_ids() -> None:
    for value in (*_chain(TENANT_ONE).values(), derive_blob_id(HELLO)):
        assert parse_id(str(value)) == value
        assert type(value).parse(str(value)) == value
        assert repr(value) == f"{type(value).__name__}({str(value)!r})"


def test_parse_enforces_the_requested_kind() -> None:
    text = str(_chain(TENANT_ONE)["rev"])
    assert isinstance(parse_id(text, IdKind.REV), RevisionId)
    with pytest.raises(IdError, match="expected a 'doc'"):
        parse_id(text, IdKind.DOC)
    with pytest.raises(IdError, match="expected a 'chk'"):
        ChunkId.parse(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "doc",
        "doc_",
        "_" + "0" * 26,
        "zzz_" + "0" * 26,
        "doc_" + "0" * 25,
        "doc_" + "0" * 27,
        "DOC_" + "0" * 26,
        "doc_" + "A" * 26,
        "doc_" + "8" + "0" * 25,
        "doc_" + "i" + "0" * 25,
        "doc-" + "0" * 26,
        " doc_" + "0" * 26,
        "doc_" + "0" * 26 + "\n",
        "doc_" + "0" * 26 + " ",
        "doc__" + "0" * 25,
    ],
)
def test_malformed_text_is_rejected(text: str) -> None:
    with pytest.raises(IdError):
        parse_id(text)


def test_unknown_prefix_is_a_closed_registry_error() -> None:
    with pytest.raises(IdError, match="unknown identifier prefix"):
        parse_id("zzz_" + "0" * 26)


def test_non_string_text_is_rejected() -> None:
    with pytest.raises(IdError):
        parse_id(None)  # type: ignore[arg-type]


# --- levels are independent -----------------------------------------------------------------


def test_extraction_changes_do_not_touch_document_or_revision() -> None:
    base = _chain(TENANT_ONE)
    changed = _chain(TENANT_ONE, ext_config="parser=layout;v=2")
    assert (base["doc"], base["rev"]) == (changed["doc"], changed["rev"])
    for level in ("ext", "el", "cks", "chk"):
        assert base[level] != changed[level]


def test_chunking_changes_do_not_touch_the_extraction() -> None:
    base = _chain(TENANT_ONE)
    changed = _chain(TENANT_ONE, chunk_config="recursive;v=3")
    for level in ("doc", "rev", "ext", "el"):
        assert base[level] == changed[level]
    for level in ("cks", "chk"):
        assert base[level] != changed[level]


def test_source_changes_leave_only_the_document_unchanged() -> None:
    base = _chain(TENANT_ONE)
    changed = _chain(TENANT_ONE, content=b"hello, edited")
    assert base["doc"] == changed["doc"]
    for level in ("rev", "ext", "el", "cks", "chk"):
        assert base[level] != changed[level]


def test_returning_to_earlier_content_returns_to_the_same_revision() -> None:
    first = _chain(TENANT_ONE, content=b"a")
    _chain(TENANT_ONE, content=b"b")
    assert _chain(TENANT_ONE, content=b"a") == first


# --- static typing --------------------------------------------------------------------------

SNIPPET = textwrap.dedent(
    """
    from evidex.core import (
        ChunkId, DocumentId, IdKind, TenantId, assign_id, derive_blob_id, derive_id, hash_bytes,
    )

    tenant = assign_id(IdKind.TEN, bytes(16))
    doc = derive_id(IdKind.DOC, tenant, "collection", "key")
    rev = derive_id(IdKind.REV, doc, hash_bytes(b"x"))
    ext = derive_id(IdKind.EXT, rev, "config")
    chunk_set = derive_id(IdKind.CKS, ext, "chunker")
    chunk = derive_id(IdKind.CHK, chunk_set, 0)
    blob = derive_blob_id(hash_bytes(b"x"))


    def needs_document(value: DocumentId) -> None: ...
    def needs_tenant(value: TenantId) -> None: ...
    def needs_chunk(value: ChunkId) -> None: ...


    needs_document(doc)
    needs_tenant(tenant)
    needs_chunk(chunk)
    needs_document(chunk)  # expect-error
    needs_tenant(doc)  # expect-error
    needs_chunk(blob)  # expect-error
    derive_id(IdKind.REV, tenant, hash_bytes(b"x"))  # expect-error
    derive_id(IdKind.CHK, ext, 0)  # expect-error
    derive_id(IdKind.TEN, tenant, "slug")  # expect-error
    derive_id(IdKind.BLOB, tenant, "x")  # expect-error
    assign_id(IdKind.REV, bytes(16))  # expect-error
    assign_id(IdKind.BLOB, bytes(16))  # expect-error
    """
)


def test_misuse_is_a_static_type_error(tmp_path: Path) -> None:
    snippet = tmp_path / "snippet.py"
    snippet.write_text(SNIPPET, encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1] / "src"
    completed = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-error-summary",
         "--cache-dir", str(tmp_path / "cache"), str(snippet)],
        cwd=tmp_path,
        env={**os.environ, "MYPYPATH": str(source_root)},
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    reported = {int(m.group(1)) for m in re.finditer(r":(\d+): error:", completed.stdout)}
    expected = {
        number
        for number, line in enumerate(SNIPPET.splitlines(), start=1)
        if "# expect-error" in line
    }
    assert len(expected) == 9  # the fixtures must really be there
    assert reported == expected, completed.stdout
