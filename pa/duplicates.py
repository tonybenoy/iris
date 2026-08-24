"""Finding photos stored more than once, and knowing when that is deliberate.

Exact duplicates fall out of the content hash for free: the same bytes in three
folders are already one photo with three files. This module is about the other
case -- visually identical images that are not byte-identical, found by
perceptual hash.

The catch is that "visually identical but not the same file" is also what a
RAW + JPEG pair looks like, and what an iPhone's HEIC + JPG pair looks like.
Those are not waste; they are the camera doing what it was told. Reporting them
as duplicates trains you to ignore the duplicates screen.
"""
from __future__ import annotations

import sqlite3

from pa.db.repo import RAW_EXTS


def _split(filename: str) -> tuple[str, str]:
    stem, dot, ext = (filename or "").rpartition(".")
    return (stem.casefold(), ext.casefold()) if dot else ("", "")


def same_shot_different_format(filenames: list[str]) -> bool:
    """Is this group one photo saved in several formats, rather than copies?

    The test is a shared stem with no repeated extension: DSC_0042.NEF next to
    DSC_0042.JPG is the camera writing RAW+JPEG, while two files both called
    holiday.jpg are genuinely the same thing kept twice.

    Requiring at least one RAW or HEIC member keeps this narrow. Two unrelated
    exports that happen to share a stem, say cover.jpg and cover.png, are still
    worth telling you about.
    """
    pairs = [_split(f) for f in filenames]
    if len(pairs) < 2 or any(not stem for stem, _ in pairs):
        return False
    stems = {stem for stem, _ in pairs}
    exts = [ext for _, ext in pairs]
    if len(stems) != 1 or len(set(exts)) != len(exts):
        return False
    return any(f".{e}" in RAW_EXTS or e in ("heic", "heif") for e in exts)


def near_duplicates(conn: sqlite3.Connection, max_distance: int,
                    include_format_pairs: bool = False) -> tuple[list[dict], int]:
    """Group photos whose perceptual hashes are within `max_distance` bits.

    Returns (groups, format_pairs_skipped).

    Brute-force pairwise, but only over photos that HAVE a pHash and using
    numpy popcount over a packed array, so 500k photos is one vectorised pass
    rather than 125 billion Python-level comparisons.
    """
    import numpy as np

    rows = conn.execute(
        """SELECT p.id, p.phash, (SELECT filename FROM file f WHERE f.photo_id=p.id LIMIT 1) fn
           FROM photo p WHERE p.phash IS NOT NULL""").fetchall()
    if len(rows) < 2:
        return [], 0
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    names = {r["id"]: r["fn"] for r in rows}
    hashes = np.array([r["phash"] & 0xFFFFFFFFFFFFFFFF for r in rows], dtype=np.uint64)
    packed = hashes.view(np.uint8).reshape(len(hashes), 8)

    seen: set[int] = set()
    groups: list[dict] = []
    skipped = 0
    for i in range(len(ids)):
        if int(ids[i]) in seen:
            continue
        dist = np.unpackbits(packed[i] ^ packed, axis=1).sum(axis=1)
        hit = np.where(dist <= max_distance)[0]
        if len(hit) < 2:
            continue
        members = [int(ids[j]) for j in hit]
        seen.update(members)
        pair = same_shot_different_format([names.get(m) or "" for m in members])
        if pair and not include_format_pairs:
            skipped += 1
            continue
        groups.append({"ids": members, "format_pair": pair,
                       "example": names.get(members[0]) or str(members[0])})
    return groups, skipped
