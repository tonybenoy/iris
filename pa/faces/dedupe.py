"""Telling one detection of a face from a second detection of the same face.

Re-running detection on a photo finds the same faces again, and the ones you
already decided about -- named, or ignored -- are deliberately kept rather than
deleted. Without a way to recognise that a fresh detection *is* one of those,
every re-run adds a second copy of every face you ever named: the person shows
up twice on the photo, their count doubles, and every stranger you dismissed
comes back to the naming queue.
"""
from __future__ import annotations

import sqlite3

# Intersection over union at which two boxes are the same face rather than two
# faces near each other. 0.5 is the usual line in detection work, and detectors
# are stable enough between runs that a re-detection of one face scores far
# above it -- typically 1.0, the box being identical.
SAME_FACE_IOU = 0.5


def normalised(box: tuple[float, float, float, float],
               src: tuple[float | None, float | None]) -> tuple[float, float, float, float]:
    """A box as fractions of its image, so boxes measured against different
    sizes can be compared. Rows from before source dimensions were recorded have
    nothing to scale by and are left in pixels -- which is what comparing them
    at all already assumes."""
    x, y, w, h = box
    src_w, src_h = src
    if not src_w or not src_h:
        return x, y, w, h
    return x / src_w, y / src_h, w / src_w, h / src_h


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = overlap_w * overlap_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def decided_boxes(conn: sqlite3.Connection, photo_id: int) -> list[tuple]:
    """Boxes on this photo that the user has ruled on, named or ignored."""
    return [
        normalised((r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]),
                   (r["src_w"], r["src_h"]))
        for r in conn.execute(
            """SELECT bbox_x, bbox_y, bbox_w, bbox_h, src_w, src_h FROM face
               WHERE photo_id=? AND (confirmed=1 OR rejected=1)""", (photo_id,))
    ]


def is_already_decided(box: tuple, src: tuple, decided: list[tuple]) -> bool:
    here = normalised(box, src)
    return any(iou(here, other) >= SAME_FACE_IOU for other in decided)


def _decided(row: sqlite3.Row) -> bool:
    return bool(row["confirmed"] or row["rejected"])


def _disagree(kept: sqlite3.Row, other: sqlite3.Row) -> bool:
    """Whether two records of one face say different things about it.

    One ignored and one named is a contradiction, and so is naming two different
    people. Which is right is not something to guess at, so both are left where
    they are for a person to sort out.

    A record nobody has ruled on holds no opinion and so contradicts nothing --
    including a machine's guess at who it is, which the decision it duplicates
    already answers better.
    """
    if not _decided(other):
        return False
    if bool(kept["rejected"]) != bool(other["rejected"]):
        return True
    return (kept["person_id"] is not None and other["person_id"] is not None
            and kept["person_id"] != other["person_id"])


def find_duplicates(conn: sqlite3.Connection) -> list[int]:
    """Faces that are a second record of a face already on the same photo.

    Ranked so the record a person acted on is the one that survives: anything
    decided outranks anything undecided, and the older row wins between equals,
    it being the one that has been referred to for longer.

    Copies made before detection knew to skip decided faces are usually still
    undecided, but not always -- naming a proposed group after a re-run marks
    the copy confirmed too, leaving two identical named boxes. Both are the same
    claim about the same face, so keeping one loses nothing.
    """
    rows = conn.execute(
        """SELECT id, photo_id, person_id, confirmed, rejected,
                  bbox_x, bbox_y, bbox_w, bbox_h, src_w, src_h
           FROM face ORDER BY photo_id, id""").fetchall()
    by_photo: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_photo.setdefault(row["photo_id"], []).append(row)

    duplicates: list[int] = []
    for faces in by_photo.values():
        if len(faces) < 2:
            continue
        kept: list[tuple[sqlite3.Row, tuple]] = []
        for row in sorted(faces, key=lambda r: (not _decided(r), r["id"])):
            box = normalised((row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]),
                             (row["src_w"], row["src_h"]))
            match = next((k for k, kbox in kept if iou(box, kbox) >= SAME_FACE_IOU), None)
            if match is None or _disagree(match, row):
                kept.append((row, box))
            else:
                duplicates.append(row["id"])
    return duplicates


def remove_duplicates(conn: sqlite3.Connection) -> int:
    ids = find_duplicates(conn)
    conn.executemany("DELETE FROM face WHERE id=?", [(i,) for i in ids])
    conn.commit()
    return len(ids)
