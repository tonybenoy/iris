"""Data access. Every write the indexer performs goes through here."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

STAGES = ("thumbs", "embed", "faces", "caption")

# Which annotation wins when a photo has several. A hand edit always outranks
# model output, however recently the model ran: re-captioning a library must
# never quietly shadow something the user wrote. Every read of `annotation`
# orders by this.
ANNOTATION_ORDER = "(a.model = 'manual') DESC, a.created_at DESC"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif",
              ".bmp", ".tif", ".tiff", ".avif"}
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef"}


def now() -> int:
    return int(time.time())


# ------------------------------------------------------------------ volumes
def upsert_volume(conn: sqlite3.Connection, uuid: str, label: str,
                  mountpoint: str | None) -> int:
    conn.execute(
        """INSERT INTO volume (uuid, label, last_mount, last_seen_at, online)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(uuid) DO UPDATE SET
             label=excluded.label, last_mount=excluded.last_mount,
             last_seen_at=excluded.last_seen_at, online=excluded.online""",
        (uuid, label, mountpoint, now(), 1 if mountpoint else 0))
    return conn.execute("SELECT id FROM volume WHERE uuid=?", (uuid,)).fetchone()["id"]


def set_volume_online(conn: sqlite3.Connection, volume_id: int,
                      mountpoint: str | None) -> None:
    conn.execute("UPDATE volume SET online=?, last_mount=?, last_seen_at=? WHERE id=?",
                 (1 if mountpoint else 0, mountpoint, now(), volume_id))


# -------------------------------------------------------------------- roots
def add_root(conn: sqlite3.Connection, volume_id: int, rel_path: str,
             label: str | None, exclude_globs: list[str] | None) -> int:
    conn.execute(
        """INSERT INTO root (volume_id, rel_path, label, exclude_globs, added_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(volume_id, rel_path) DO UPDATE SET
             label=excluded.label, enabled=1""",
        (volume_id, rel_path, label, json.dumps(exclude_globs or []), now()))
    return conn.execute("SELECT id FROM root WHERE volume_id=? AND rel_path=?",
                        (volume_id, rel_path)).fetchone()["id"]


def list_roots(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT r.*, v.uuid AS volume_uuid, v.label AS volume_label,
                  v.online, v.last_mount
           FROM root r JOIN volume v ON v.id = r.volume_id
           ORDER BY v.label, r.rel_path""").fetchall()


# ------------------------------------------------------------------- photos
def get_photo_by_hash(conn: sqlite3.Connection, blake3: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM photo WHERE blake3=?", (blake3,)).fetchone()


def insert_photo(conn: sqlite3.Connection, blake3: str, meta, phash: int | None,
                 size: int) -> int:
    cur = conn.execute(
        """INSERT INTO photo (blake3, phash, width, height, bytes, mime, taken_at,
                              taken_at_source, camera_make, camera_model, lens, iso,
                              f_number, exposure_s, focal_len, orientation,
                              gps_lat, gps_lon, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (blake3, phash, meta.width, meta.height, size, meta.mime, meta.taken_at,
         meta.taken_at_source, meta.camera_make, meta.camera_model, meta.lens,
         meta.iso, meta.f_number, meta.exposure_s, meta.focal_len, meta.orientation,
         meta.gps_lat, meta.gps_lon, now()))
    return cur.lastrowid


def link_file(conn: sqlite3.Connection, photo_id: int, root_id: int, rel_path: str,
              filename: str, mtime: int, size: int) -> None:
    conn.execute(
        """INSERT INTO file (photo_id, root_id, rel_path, filename, mtime, size,
                             state, seen_at)
           VALUES (?,?,?,?,?,?, 'present', ?)
           ON CONFLICT(root_id, rel_path) DO UPDATE SET
             photo_id=excluded.photo_id, mtime=excluded.mtime, size=excluded.size,
             state='present', seen_at=excluded.seen_at""",
        (photo_id, root_id, rel_path, filename, mtime, size, now()))


def file_unchanged(conn: sqlite3.Connection, root_id: int, rel_path: str,
                   mtime: int, size: int) -> bool:
    """True if we have seen this exact path at this exact mtime+size before.
    Lets a rescan skip hashing entirely for files that cannot have changed."""
    row = conn.execute(
        "SELECT mtime, size FROM file WHERE root_id=? AND rel_path=?",
        (root_id, rel_path)).fetchone()
    return row is not None and row["mtime"] == mtime and row["size"] == size


def mark_missing(conn: sqlite3.Connection, root_id: int, seen_before: int) -> int:
    cur = conn.execute(
        "UPDATE file SET state='missing' WHERE root_id=? AND seen_at < ? AND state='present'",
        (root_id, seen_before))
    return cur.rowcount


def set_root_files_offline(conn: sqlite3.Connection, root_id: int, offline: bool) -> None:
    conn.execute("UPDATE file SET state=? WHERE root_id=? AND state IN ('present','offline')",
                 ("offline" if offline else "present", root_id))


def prune_orphans(conn: sqlite3.Connection, drop_missing: bool = True) -> int:
    """Delete photos that no longer exist anywhere on disk.

    Two ways a photo ends up with no readable file:
      - every `file` row is gone, because its root was removed (rows cascade,
        but the `photo` row is deliberately not tied to any one root);
      - the rows survive but every one is marked 'missing', because a rescan
        found the file deleted.

    Both leave a photo that still answers searches and still shows a cached
    thumbnail, with nothing behind it to open. `drop_missing=False` keeps the
    second kind, which is what you want if a drive was merely unplugged during
    the last scan.
    """
    clause = ("NOT EXISTS (SELECT 1 FROM file f WHERE f.photo_id = p.id AND "
              "f.state != 'missing')" if drop_missing else
              "NOT EXISTS (SELECT 1 FROM file f WHERE f.photo_id = p.id)")
    ids = [r["id"] for r in conn.execute(f"SELECT p.id FROM photo p WHERE {clause}")]
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM photo_fts WHERE rowid IN ({marks})", ids)
    conn.execute(f"DELETE FROM photo WHERE id IN ({marks})", ids)
    return len(ids)


def remove_root(conn: sqlite3.Connection, root_id: int) -> tuple[int, int]:
    """Forget a folder. Returns (files removed, photos left with no location)."""
    n_files = conn.execute("SELECT COUNT(*) FROM file WHERE root_id=?",
                           (root_id,)).fetchone()[0]
    conn.execute("DELETE FROM root WHERE id=?", (root_id,))
    return n_files, prune_orphans(conn)


# --------------------------------------------------------------------- jobs
def enqueue(conn: sqlite3.Connection, photo_id: int, stages=STAGES,
            priority: int = 100, reset: bool = False) -> None:
    for stage in stages:
        if reset:
            conn.execute(
                """INSERT INTO job (photo_id, stage, state, priority, created_at)
                   VALUES (?,?,'pending',?,?)
                   ON CONFLICT(photo_id, stage) DO UPDATE SET
                     state='pending', attempts=0, error=NULL, priority=excluded.priority,
                     started_at=NULL, finished_at=NULL""",
                (photo_id, stage, priority, now()))
        else:
            conn.execute(
                """INSERT INTO job (photo_id, stage, state, priority, created_at)
                   VALUES (?,?,'pending',?,?)
                   ON CONFLICT(photo_id, stage) DO NOTHING""",
                (photo_id, stage, priority, now()))


def claim_jobs(conn: sqlite3.Connection, stage: str, limit: int) -> list[sqlite3.Row]:
    """Atomically take up to `limit` pending jobs. RETURNING keeps the select and
    the state change in one statement, so two workers cannot claim the same job."""
    return conn.execute(
        """UPDATE job SET state='running', started_at=?, attempts=attempts+1
           WHERE id IN (SELECT id FROM job WHERE stage=? AND state='pending'
                        ORDER BY priority, id LIMIT ?)
           RETURNING *""",
        (now(), stage, limit)).fetchall()


def finish_job(conn: sqlite3.Connection, job_id: int, state: str,
               error: str | None = None, model_version: str | None = None) -> None:
    conn.execute(
        "UPDATE job SET state=?, error=?, model_version=?, finished_at=? WHERE id=?",
        (state, error, model_version, now(), job_id))


def requeue_stale(conn: sqlite3.Connection) -> int:
    """Jobs left 'running' by a killed process are pending again on restart."""
    cur = conn.execute(
        "UPDATE job SET state='pending' WHERE state='running' AND attempts < 5")
    return cur.rowcount


def job_stats(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for row in conn.execute("SELECT stage, state, COUNT(*) n FROM job GROUP BY stage, state"):
        stats.setdefault(row["stage"], {})[row["state"]] = row["n"]
    return stats


# --------------------------------------------------------------------- tags
def get_or_create_tag(conn: sqlite3.Connection, name: str, kind: str = "ai") -> int:
    name = name.strip().lower()
    row = conn.execute("SELECT id FROM tag WHERE name=?", (name,)).fetchone()
    if row:
        return row["id"]
    return conn.execute("INSERT INTO tag (name, kind, created_at) VALUES (?,?,?)",
                        (name, kind, now())).lastrowid


def set_tags(conn: sqlite3.Connection, photo_id: int, names: list[str],
             source: str = "ai", confidence: float | None = None) -> None:
    """Replace this source's tags for the photo. Deliberately scoped to `source`
    so re-running the AI stage never deletes a tag the user typed."""
    conn.execute("DELETE FROM photo_tag WHERE photo_id=? AND source=?", (photo_id, source))
    for name in names:
        if not name.strip():
            continue
        tag_id = get_or_create_tag(conn, name, "manual" if source == "manual" else "ai")
        conn.execute(
            """INSERT INTO photo_tag (photo_id, tag_id, source, confidence, created_at)
               VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING""",
            (photo_id, tag_id, source, confidence, now()))


def save_annotation(conn: sqlite3.Connection, photo_id: int, ann: dict[str, Any],
                    model: str, version: str) -> None:
    conn.execute(
        """INSERT INTO annotation (photo_id, caption, scene, setting, people_count,
                                   ocr_text, raw_json, model, model_version, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(photo_id, model, model_version) DO UPDATE SET
             caption=excluded.caption, scene=excluded.scene, setting=excluded.setting,
             people_count=excluded.people_count, ocr_text=excluded.ocr_text,
             raw_json=excluded.raw_json, created_at=excluded.created_at""",
        (photo_id, ann.get("caption"), ann.get("scene"), ann.get("setting"),
         ann.get("people_count"), ann.get("ocr_text"), json.dumps(ann), model,
         version, now()))


# ------------------------------------------------------------------- search
def reindex_fts(conn: sqlite3.Connection, photo_id: int) -> None:
    """Rebuild this photo's FTS row from its current annotation, tags and people."""
    ann = conn.execute(
        "SELECT caption, ocr_text, scene FROM annotation a WHERE photo_id=? "
        f"ORDER BY {ANNOTATION_ORDER} LIMIT 1", (photo_id,)).fetchone()
    tags = [r["name"] for r in conn.execute(
        "SELECT t.name FROM tag t JOIN photo_tag pt ON pt.tag_id=t.id WHERE pt.photo_id=?",
        (photo_id,))]
    people = [r["name"] for r in conn.execute(
        "SELECT DISTINCT p.name FROM person p JOIN face f ON f.person_id=p.id "
        "WHERE f.photo_id=? AND p.name IS NOT NULL", (photo_id,))]
    files = conn.execute(
        "SELECT filename, rel_path FROM file WHERE photo_id=? LIMIT 4", (photo_id,)).fetchall()
    place = conn.execute("SELECT place_name FROM photo WHERE id=?", (photo_id,)).fetchone()

    conn.execute("DELETE FROM photo_fts WHERE rowid=?", (photo_id,))
    conn.execute(
        """INSERT INTO photo_fts (rowid, caption, tags, ocr_text, people, filename, folder, place)
           VALUES (?,?,?,?,?,?,?,?)""",
        (photo_id,
         " ".join(filter(None, [ann["caption"] if ann else None, ann["scene"] if ann else None])),
         " ".join(tags),
         (ann["ocr_text"] if ann else "") or "",
         " ".join(people),
         " ".join(f["filename"] for f in files),
         " ".join(str(Path(f["rel_path"]).parent) for f in files),
         (place["place_name"] if place else "") or ""))
