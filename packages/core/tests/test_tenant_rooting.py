from __future__ import annotations

import inspect

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evidex.core import (
    ContentHash,
    Id,
    IdError,
    IdKind,
    TenantId,
    assign_id,
    derive_blob_id,
    derive_id,
    hash_bytes,
)

PROPERTY = settings(max_examples=200, deadline=None)
RAW = st.binary(min_size=16, max_size=16)
NAME = st.text(min_size=1, max_size=12)


def _chain(
    tenant: TenantId, collection: str, key: str, content: bytes, ext_config: str, chunk_config: str
) -> dict[str, Id]:
    doc = derive_id(IdKind.DOC, tenant, collection, key)
    rev = derive_id(IdKind.REV, doc, hash_bytes(content))
    ext = derive_id(IdKind.EXT, rev, ext_config)
    element = derive_id(IdKind.EL, ext, "/0", "paragraph", hash_bytes(b"text"))
    chunk_set = derive_id(IdKind.CKS, ext, chunk_config)
    chunk = derive_id(IdKind.CHK, chunk_set, 0, hash_bytes(b"text"))
    return {"doc": doc, "rev": rev, "ext": ext, "el": element, "cks": chunk_set, "chk": chunk}


@PROPERTY
@given(
    first=RAW,
    second=RAW,
    collection=NAME,
    key=NAME,
    content=st.binary(max_size=64),
    ext_config=NAME,
    chunk_config=NAME,
)
def test_no_id_in_the_chain_is_shared_between_tenants(
    first: bytes,
    second: bytes,
    collection: str,
    key: str,
    content: bytes,
    ext_config: str,
    chunk_config: str,
) -> None:
    if first == second:
        return
    one = _chain(assign_id(IdKind.TEN, first), collection, key, content, ext_config, chunk_config)
    two = _chain(assign_id(IdKind.TEN, second), collection, key, content, ext_config, chunk_config)
    assert one.keys() == two.keys()
    for level in one:
        assert one[level] != two[level], f"{level} coincides across tenants"
        assert str(one[level]) != str(two[level])


@PROPERTY
@given(content=st.binary(max_size=64))
def test_blob_ids_are_content_only_and_therefore_shared(content: bytes) -> None:
    assert derive_blob_id(hash_bytes(content)) == derive_blob_id(hash_bytes(content))
    assert derive_blob_id(hash_bytes(content)) != derive_blob_id(hash_bytes(content + b"\x00"))


def test_only_blob_ids_can_be_built_from_a_content_hash() -> None:
    digest = hash_bytes(b"x")
    for kind in IdKind:
        with pytest.raises(IdError):
            derive_id(kind, digest, "part")  # type: ignore[call-overload]
    assert derive_blob_id(digest).kind is IdKind.BLOB
    assert [k for k in IdKind if k.prefix == derive_blob_id(digest).prefix] == [IdKind.BLOB]


def test_no_derivation_path_exists_without_a_parent() -> None:
    parameters = inspect.signature(derive_id).parameters
    assert list(parameters) == ["kind", "parent", "parts"]
    assert parameters["parent"].default is inspect.Parameter.empty
    assert inspect.signature(derive_blob_id).parameters["content"].annotation in (
        ContentHash,
        "ContentHash",
    )


def test_a_tenant_id_cannot_be_derived_from_anything() -> None:
    tenant = assign_id(IdKind.TEN, bytes(16))
    doc = assign_id(IdKind.DOC, bytes(16))
    rev = derive_id(IdKind.REV, doc, hash_bytes(b"x"))
    ext = derive_id(IdKind.EXT, rev, "config")
    parents: list[Id] = [
        tenant,
        doc,
        rev,
        ext,
        derive_id(IdKind.EL, ext, "p"),
        derive_id(IdKind.CKS, ext, "c"),
    ]
    for parent in parents:
        with pytest.raises(IdError, match="assigned, never derived"):
            derive_id(IdKind.TEN, parent, "operator-slug")  # type: ignore[call-overload]


def test_assigning_an_id_takes_no_tenant_information() -> None:
    assert list(inspect.signature(assign_id).parameters) == ["kind", "raw"]
    raw = bytes(range(16))
    assert assign_id(IdKind.DOC, raw) == assign_id(IdKind.DOC, raw)
    as_document: object = assign_id(IdKind.DOC, raw)
    as_tenant: object = assign_id(IdKind.TEN, raw)
    assert as_document != as_tenant


def test_children_of_an_assigned_document_are_unique_to_it() -> None:
    first, second = (assign_id(IdKind.DOC, bytes([n]) * 16) for n in (1, 2))
    content = hash_bytes(b"same bytes")
    assert derive_id(IdKind.REV, first, content) != derive_id(IdKind.REV, second, content)
