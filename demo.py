from evidex.core import IdError, IdKind, assign_id, derive_blob_id, derive_id, hash_bytes

tenant = assign_id(IdKind.TEN, bytes(range(16)))
doc = derive_id(IdKind.DOC, tenant, "main", "docs/a.pdf")
rev = derive_id(IdKind.REV, doc, hash_bytes(b"hello"))
ext_a = derive_id(IdKind.EXT, rev, "parser=plain;v=1")
ext_b = derive_id(IdKind.EXT, rev, "parser=layout;v=2")
print("tenant   ", tenant); print("document ", doc); print("revision ", rev)
print("extract A", ext_a); print("extract B", ext_b)
print("parser change keeps document and revision:", ext_a != ext_b)
other = assign_id(IdKind.TEN, bytes(range(16, 32)))
other_doc = derive_id(IdKind.DOC, other, "main", "docs/a.pdf")
other_rev = derive_id(IdKind.REV, other_doc, hash_bytes(b"hello"))
print("same file, other tenant -> different ids:", other_doc != doc and other_rev != rev)
print("blob (content only)      ", derive_blob_id(hash_bytes(b"hello")))
try:
    derive_id(IdKind.TEN, tenant, "operator-slug")  # type: ignore[call-overload]
except IdError as error:
    print("deriving a tenant id is refused:", error)