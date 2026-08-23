"""Hybrid search: structured filters + FTS5 keyword + vector similarity, fused.

The three signals answer different questions and none is sufficient alone:
  - filters  are exact and cheap  (person, date, camera, folder)
  - FTS5     nails names and literal words that appear in captions or OCR
  - vectors  handle "mountains", "golden hour" - meaning, not spelling

Results are combined with Reciprocal Rank Fusion, which needs only each engine's
ranking rather than its scores. That matters because BM25 and cosine similarity
are on wholly different scales and normalising them against each other is
guesswork that breaks whenever a model changes.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from pa.search.parser import Query, parse

RRF_K = 60  # standard damping constant; larger = flatter contribution from rank


@dataclass
class Hit:
    photo_id: int
    score: float
    caption: str | None = None
    blake3: str | None = None
    taken_at: int | None = None
    filename: str | None = None
    sources: tuple[str, ...] = ()


def _fts_expr(text: str) -> str | None:
    """Turn free text into an FTS5 MATCH expression, with a prefix on the last
    token so search-as-you-type matches partial words."""
    tokens = [t for t in re.findall(r"\w+", text) if t]
    if not tokens:
        return None
    quoted = [f'"{t}"' for t in tokens[:-1]]
    quoted.append(f'"{tokens[-1]}"*')
    return " AND ".join(quoted)


def _filter_sql(q: Query) -> tuple[str, list]:
    clauses, params = [], []
    if q.people:
        for name in q.people:
            clauses.append("""EXISTS (SELECT 1 FROM face f JOIN person pe ON pe.id=f.person_id
                                      WHERE f.photo_id=p.id AND pe.name LIKE ?)""")
            params.append(f"%{name}%")
    for tag in q.tags:
        clauses.append("""EXISTS (SELECT 1 FROM photo_tag pt JOIN tag t ON t.id=pt.tag_id
                                  WHERE pt.photo_id=p.id AND t.name=?)""")
        params.append(tag.lower())
    if q.camera:
        clauses.append("(p.camera_model LIKE ? OR p.camera_make LIKE ?)")
        params += [f"%{q.camera}%", f"%{q.camera}%"]
    if q.place:
        clauses.append("p.place_name LIKE ?")
        params.append(f"%{q.place}%")
    if q.folder:
        clauses.append("EXISTS (SELECT 1 FROM file fi "
                       "WHERE fi.photo_id=p.id AND fi.rel_path LIKE ?)")
        params.append(f"%{q.folder}%")
    if q.filename:
        clauses.append("EXISTS (SELECT 1 FROM file fi "
                       "WHERE fi.photo_id=p.id AND fi.filename LIKE ?)")
        params.append(f"%{q.filename}%")
    if q.date_from:
        clauses.append("p.taken_at >= ?")
        params.append(q.date_from)
    if q.date_to:
        clauses.append("p.taken_at <= ?")
        params.append(q.date_to)
    if q.favorite:
        clauses.append("p.favorite = 1")
    if q.has_faces is not None:
        op = "EXISTS" if q.has_faces else "NOT EXISTS"
        clauses.append(f"{op} (SELECT 1 FROM face f WHERE f.photo_id=p.id)")
    clauses.append("p.hidden = 0")
    # A photo whose every file is 'missing' was deleted from disk. Its row and
    # cached thumbnail survive so `pa prune` can report it, but offering it as a
    # search result means handing back something that cannot be opened.
    # 'offline' is deliberately still included: that drive is merely unplugged.
    clauses.append("EXISTS (SELECT 1 FROM file f WHERE f.photo_id=p.id "
                   "AND f.state != 'missing')")
    return " AND ".join(clauses), params


def _fts_ranked(conn, q: Query, where: str, params: list, limit: int) -> list[int]:
    expr = _fts_expr(q.text)
    if not expr:
        return []
    sql = f"""SELECT p.id FROM photo_fts JOIN photo p ON p.id = photo_fts.rowid
              WHERE photo_fts MATCH ? AND {where}
              ORDER BY bm25(photo_fts, 8.0, 6.0, 3.0, 10.0, 2.0, 1.0, 4.0)
              LIMIT ?"""
    try:
        return [r[0] for r in conn.execute(sql, [expr, *params, limit])]
    except sqlite3.OperationalError:
        return []


def _filtered_ranked(conn, where: str, params: list, limit: int) -> list[int]:
    """Filters only, newest first - the answer when there is no free text at all
    ("photos of Sarah last summer" is entirely structured)."""
    return [r[0] for r in conn.execute(
        f"SELECT p.id FROM photo p WHERE {where} ORDER BY p.taken_at DESC LIMIT ?",
        [*params, limit])]


def _rrf(rankings: dict[str, list[int]],
         k: int = RRF_K) -> dict[int, tuple[float, tuple[str, ...]]]:
    scores: dict[int, float] = {}
    sources: dict[int, list[str]] = {}
    for name, ids in rankings.items():
        for rank, pid in enumerate(ids):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(pid, []).append(name)
    return {pid: (score, tuple(sources[pid])) for pid, score in scores.items()}


def search(conn: sqlite3.Connection, raw: str, limit: int = 60,
           vector_search=None) -> list[Hit]:
    q = parse(raw)
    where, params = _filter_sql(q)
    pool = max(limit * 4, 200)

    rankings: dict[str, list[int]] = {}
    if not q.text:
        # Nothing fuzzy to match -- the query is entirely structured
        # ("person:Sarah last summer"), so the filters ARE the answer.
        rankings["filter"] = _filtered_ranked(conn, where, params, pool)
    else:
        rankings["fts"] = _fts_ranked(conn, q, where, params, pool)
        if vector_search is not None:
            allowed = set(_filtered_ranked(conn, where, params, 100_000))
            rankings["vec"] = [pid for pid in vector_search(q.text, pool) if pid in allowed]
        elif not rankings["fts"]:
            # Degraded mode: no vector index built yet, so FTS is the only judge
            # of relevance and it only knows literal words. Falling back to the
            # filters beats claiming the library holds nothing.
            #
            # When the vector index IS available we deliberately do NOT do this:
            # both engines reporting no match is a real answer ("no mountains in
            # this library"), and answering it with every photo you own is worse
            # than answering it with nothing.
            rankings["filter"] = _filtered_ranked(conn, where, params, pool)

    fused = _rrf(rankings)
    top = sorted(fused.items(), key=lambda kv: -kv[1][0])[:limit]
    if not top:
        return []

    ids = [pid for pid, _ in top]
    placeholders = ",".join("?" * len(ids))
    rows = {r["id"]: r for r in conn.execute(
        f"""SELECT p.id, p.blake3, p.taken_at,
                   (SELECT caption FROM annotation a WHERE a.photo_id=p.id
                    ORDER BY (a.model='manual') DESC, a.created_at DESC LIMIT 1) caption,
                   (SELECT filename FROM file f WHERE f.photo_id=p.id LIMIT 1) filename
            FROM photo p WHERE p.id IN ({placeholders})""", ids)}

    return [Hit(photo_id=pid, score=score, sources=srcs,
                caption=rows[pid]["caption"] if pid in rows else None,
                blake3=rows[pid]["blake3"] if pid in rows else None,
                taken_at=rows[pid]["taken_at"] if pid in rows else None,
                filename=rows[pid]["filename"] if pid in rows else None)
            for pid, (score, srcs) in top if pid in rows]
