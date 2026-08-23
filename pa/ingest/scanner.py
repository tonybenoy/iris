"""Filesystem scanning: walk roots, hash new/changed files, enqueue work."""
from __future__ import annotations

import contextlib
import fnmatch
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pa.db import repo
from pa.db.connection import transaction
from pa.ingest import exif, hashing, thumbs, volumes

DEFAULT_EXCLUDES = [
    "*/.*", "*/@eaDir/*", "*/Thumbs.db", "*/.thumbnails/*", "*/node_modules/*",
    "*/$RECYCLE.BIN/*", "*/System Volume Information/*", "*/lost+found/*",
]


@dataclass
class ScanStats:
    seen: int = 0
    skipped: int = 0
    new_photos: int = 0
    new_files: int = 0
    duplicates: int = 0
    missing: int = 0
    errors: list[str] = field(default_factory=list)


def walk_images(base: Path, excludes: list[str], recursive: bool = True) -> Iterator[Path]:
    exts = repo.IMAGE_EXTS | repo.RAW_EXTS
    patterns = DEFAULT_EXCLUDES + excludes
    stack = [base]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            path_str = str(entry)
            if any(fnmatch.fnmatch(path_str, pat) for pat in patterns):
                continue
            try:
                if entry.is_dir():
                    if recursive:
                        stack.append(entry)
                elif entry.suffix.lower() in exts:
                    yield entry
            except OSError:
                continue


def scan_root(conn: sqlite3.Connection, root_row: sqlite3.Row, cfg,
              recursive: bool = True, subdir: Path | None = None,
              force: bool = False, on_progress=None) -> ScanStats:
    """Scan one root (or a subdirectory of it) and record what is there."""
    stats = ScanStats()
    mount = volumes.current_mountpoint(root_row["volume_uuid"])
    if mount is None:
        # Drive is unplugged. Its rows stay put and go 'offline' so search still
        # finds these photos and the UI can say which drive to reconnect.
        repo.set_root_files_offline(conn, root_row["id"], True)
        repo.set_volume_online(conn, root_row["volume_id"], None)
        conn.commit()
        stats.errors.append(f"volume {root_row['volume_label']} is not mounted")
        return stats

    repo.set_volume_online(conn, root_row["volume_id"], str(mount))
    base = mount / root_row["rel_path"] if root_row["rel_path"] else mount
    target = subdir or base
    if not target.exists():
        stats.errors.append(f"path not found: {target}")
        return stats

    excludes = json.loads(root_row["exclude_globs"] or "[]")
    scan_started = repo.now()

    for path in walk_images(target, excludes, recursive):
        stats.seen += 1
        try:
            st = path.stat()
            rel = str(path.relative_to(base))
            if not force and repo.file_unchanged(conn, root_row["id"], rel,
                                                 int(st.st_mtime), st.st_size):
                stats.skipped += 1
                conn.execute("UPDATE file SET seen_at=?, state='present' "
                             "WHERE root_id=? AND rel_path=?",
                             (repo.now(), root_row["id"], rel))
                continue
            _ingest_one(conn, root_row, path, rel, st, cfg, stats)
        except Exception as exc:
            stats.errors.append(f"{path}: {exc}")
        if on_progress and stats.seen % 25 == 0:
            on_progress(stats)

    with transaction(conn):
        if subdir is None and recursive:
            stats.missing = repo.mark_missing(conn, root_row["id"], scan_started)
        conn.execute("UPDATE root SET last_scan_at=? WHERE id=?",
                     (repo.now(), root_row["id"]))
    return stats


def _ingest_one(conn: sqlite3.Connection, root_row: sqlite3.Row, path: Path,
                rel: str, st, cfg, stats: ScanStats) -> None:
    digest = hashing.content_hash(path)
    existing = repo.get_photo_by_hash(conn, digest)

    if existing is not None:
        # Same bytes already known: this is a copy on another drive or a moved
        # file. Link the path, but never re-run annotation.
        with transaction(conn):
            repo.link_file(conn, existing["id"], root_row["id"], rel, path.name,
                           int(st.st_mtime), st.st_size)
        stats.duplicates += 1
        stats.new_files += 1
        return

    phash = None
    try:
        with thumbs.open_image(path) as img:
            meta = exif.extract(path, img)
            phash = hashing.perceptual_hash(img)
    except Exception:
        meta = exif.PhotoMeta()
        with contextlib.suppress(Exception):
            meta = exif.extract(path)

    with transaction(conn):
        photo_id = repo.insert_photo(conn, digest, meta, phash, st.st_size)
        repo.link_file(conn, photo_id, root_row["id"], rel, path.name,
                       int(st.st_mtime), st.st_size)
        repo.enqueue(conn, photo_id)
        repo.reindex_fts(conn, photo_id)
    stats.new_photos += 1
    stats.new_files += 1


def resolve_file_path(conn: sqlite3.Connection, photo_id: int) -> Path | None:
    """An on-disk path for this photo, preferring a drive that is plugged in."""
    rows = conn.execute(
        """SELECT f.rel_path, r.rel_path AS root_rel, v.uuid AS vuuid, f.state
           FROM file f JOIN root r ON r.id=f.root_id JOIN volume v ON v.id=r.volume_id
           WHERE f.photo_id=? ORDER BY (f.state='present') DESC""", (photo_id,)).fetchall()
    for row in rows:
        mount = volumes.current_mountpoint(row["vuuid"])
        if mount is None:
            continue
        base = mount / row["root_rel"] if row["root_rel"] else mount
        candidate = base / row["rel_path"]
        if candidate.exists():
            return candidate
    return None
