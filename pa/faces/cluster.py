"""Group face embeddings into people.

Two-phase, and the split matters:

  1. Faces already assigned to a named person act as *anchors*. Any new face
     close enough to a person's centroid joins that person directly. This is
     what makes naming someone once keep working as new photos arrive, instead
     of re-asking every import.

  2. Whatever is left over is clustered among itself to propose new people.

Faces the user confirmed or reassigned by hand are never moved by either phase.
Clustering is a suggestion engine; the user's decisions are data.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from pa.db import repo


@dataclass
class ClusterStats:
    anchored: int = 0
    new_clusters: int = 0
    clustered_faces: int = 0
    unassigned: int = 0


def _load_faces(conn: sqlite3.Connection) -> tuple[np.ndarray, list[sqlite3.Row]]:
    rows = conn.execute(
        "SELECT id, photo_id, person_id, confirmed, embedding FROM face WHERE rejected=0"
    ).fetchall()
    if not rows:
        return np.empty((0, 512), dtype=np.float32), []
    vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    # ArcFace embeddings arrive normalised, but re-normalising costs nothing and
    # makes the dot-product-as-cosine assumption below unconditionally true.
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    return vecs, rows


def person_centroids(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    """Mean embedding per named person, built only from confirmed faces so one
    bad auto-assignment cannot drag a person's centroid off target."""
    out: dict[int, list[np.ndarray]] = {}
    for row in conn.execute(
            "SELECT person_id, embedding FROM face "
            "WHERE person_id IS NOT NULL AND confirmed=1 AND rejected=0"):
        out.setdefault(row["person_id"], []).append(
            np.frombuffer(row["embedding"], dtype=np.float32))
    centroids = {}
    for pid, vecs in out.items():
        c = np.mean(np.stack(vecs), axis=0)
        centroids[pid] = c / max(float(np.linalg.norm(c)), 1e-12)
    return centroids


def assign_to_known(conn: sqlite3.Connection, cfg) -> int:
    """Phase 1: attach unassigned faces to people who already have a name."""
    centroids = person_centroids(conn)
    if not centroids:
        return 0
    rows = conn.execute(
        "SELECT id, embedding FROM face WHERE person_id IS NULL AND rejected=0").fetchall()
    if not rows:
        return 0

    people = list(centroids)
    matrix = np.stack([centroids[p] for p in people])
    threshold = 1.0 - cfg.face.cluster_eps  # cosine distance -> similarity

    assigned = 0
    for row in rows:
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
        sims = matrix @ vec
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            conn.execute("UPDATE face SET person_id=?, confirmed=0 WHERE id=?",
                         (people[best], row["id"]))
            assigned += 1
    return assigned


def cluster_unassigned(conn: sqlite3.Connection, cfg) -> tuple[int, int, int]:
    """Phase 2: group the remaining faces into proposed clusters."""
    rows = conn.execute(
        "SELECT id, embedding FROM face WHERE person_id IS NULL AND rejected=0").fetchall()
    if len(rows) < cfg.face.min_cluster_size:
        return 0, 0, len(rows)

    vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    from sklearn.cluster import DBSCAN

    # DBSCAN rather than HDBSCAN or k-means: ArcFace distances have a genuinely
    # meaningful absolute scale (~0.4 cosine is the standard same-person
    # threshold), we do not know the number of people up front, and DBSCAN is
    # free to label a one-off stranger as noise instead of forcing them into
    # somebody's cluster.
    labels = DBSCAN(eps=cfg.face.cluster_eps, min_samples=cfg.face.min_cluster_size,
                    metric="cosine", n_jobs=-1).fit_predict(vecs)

    next_cluster = (conn.execute("SELECT COALESCE(MAX(cluster_id), 0) FROM face")
                    .fetchone()[0]) + 1
    n_clusters = 0
    n_faces = 0
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = [rows[i]["id"] for i in np.where(labels == label)[0]]
        cid = next_cluster + n_clusters
        conn.executemany("UPDATE face SET cluster_id=? WHERE id=?",
                         [(cid, fid) for fid in members])
        n_clusters += 1
        n_faces += len(members)
    noise = int(np.sum(labels == -1))
    return n_clusters, n_faces, noise


def recluster(conn: sqlite3.Connection, cfg) -> ClusterStats:
    stats = ClusterStats()
    # Clear only machine-made cluster proposals; named and confirmed faces stay.
    conn.execute("UPDATE face SET cluster_id=NULL WHERE person_id IS NULL")
    stats.anchored = assign_to_known(conn, cfg)
    stats.new_clusters, stats.clustered_faces, stats.unassigned = cluster_unassigned(conn, cfg)
    conn.commit()
    return stats


def name_cluster(conn: sqlite3.Connection, cluster_id: int, name: str) -> int:
    """Turn a proposed cluster into a named person. This is the single most
    valuable user action in the app, so it is one call."""
    row = conn.execute("SELECT id FROM person WHERE name=?", (name,)).fetchone()
    person_id = row["id"] if row else conn.execute(
        "INSERT INTO person (name, created_at) VALUES (?,?)",
        (name, repo.now())).lastrowid

    faces = conn.execute("SELECT id, photo_id FROM face WHERE cluster_id=?",
                         (cluster_id,)).fetchall()
    conn.executemany("UPDATE face SET person_id=?, confirmed=1, cluster_id=NULL WHERE id=?",
                     [(person_id, f["id"]) for f in faces])
    if not row:
        cover = conn.execute(
            "SELECT id FROM face WHERE person_id=? ORDER BY det_score DESC LIMIT 1",
            (person_id,)).fetchone()
        if cover:
            conn.execute("UPDATE person SET cover_face_id=? WHERE id=?", (cover["id"], person_id))
    for f in faces:
        repo.reindex_fts(conn, f["photo_id"])
    conn.commit()
    return len(faces)
