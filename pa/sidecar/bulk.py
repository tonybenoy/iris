"""Whole-library sidecar export and import.

Both the CLI and the web UI offer these, so the loop lives here rather than in
either one. The callers differ only in how they report progress: the CLI draws
a spinner, the server publishes counters for the browser to poll.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from pa.ingest.scanner import resolve_file_path
from pa.sidecar import xmp


@dataclass
class ExportResult:
    total: int = 0
    written: int = 0
    skipped: int = 0
    offline: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    found: int = 0
    tags: int = 0


def _photo_ids(conn: sqlite3.Connection, root_id: int | None, limit: int = 0) -> list[int]:
    sql = """SELECT DISTINCT p.id FROM photo p JOIN file f ON f.photo_id=p.id
             WHERE f.state='present'"""
    args: list = []
    if root_id is not None:
        sql += " AND f.root_id=?"
        args.append(root_id)
    sql += " ORDER BY p.id"
    if limit:
        sql += " LIMIT ?"
        args.append(limit)
    return [r["id"] for r in conn.execute(sql, args)]


def export(conn: sqlite3.Connection, cfg, root_id: int | None = None, limit: int = 0,
           overwrite: bool = False, on_progress=None) -> ExportResult:
    """Write caption, tags and named faces to .xmp files.

    Photos on a disconnected drive are counted as `offline` rather than failed:
    the sidecar's location is derived from where the original lives, so there is
    nowhere to put it until the drive is back.
    """
    ids = _photo_ids(conn, root_id, limit)
    res = ExportResult(total=len(ids))
    for i, photo_id in enumerate(ids, 1):
        path = resolve_file_path(conn, photo_id)
        if path is None:
            res.offline += 1
        elif not overwrite and xmp.resolve_path(cfg, conn, photo_id, path).exists():
            res.skipped += 1
        else:
            try:
                if xmp.write(conn, photo_id, path, cfg):
                    res.written += 1
            except OSError as exc:
                res.errors.append(f"{path.name}: {exc}")
        if on_progress and i % 20 == 0:
            on_progress(res, i)
    if on_progress:
        on_progress(res, len(ids))
    return res


def read_back(conn: sqlite3.Connection, cfg, root_id: int | None = None,
              on_progress=None) -> ImportResult:
    """Read keywords and ratings from existing .xmp files into the library."""
    ids = _photo_ids(conn, root_id)
    res = ImportResult()
    for i, photo_id in enumerate(ids, 1):
        path = resolve_file_path(conn, photo_id)
        if path is None:
            continue
        sc = xmp.read_into(conn, photo_id, path, cfg)
        if sc is not None:
            res.found += 1
            res.tags += len(sc.tags)
        if on_progress and i % 20 == 0:
            on_progress(res, i)
    conn.commit()
    if on_progress:
        on_progress(res, len(ids))
    return res
