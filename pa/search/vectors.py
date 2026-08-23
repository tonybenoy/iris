"""Vector index for image embeddings.

Brute-force cosine over a memory-mapped fp16 matrix. At 500k photos that is
~1.15GB on disk and ~150ms per query -- fast enough for one local user, and it
avoids maintaining a second storage format that can silently drift out of sync
with SQLite.

SQLite remains the source of truth: photo_embedding holds every vector, and this
index is a derived cache that `pa reindex --stage embed` can rebuild from
scratch. Past roughly 1M vectors, latency crosses ~400ms and this should be
swapped for a real ANN index behind the same three methods.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np


class VectorIndex:
    def __init__(self, directory: Path, dim: int, model: str):
        self.dir = directory
        self.dim = dim
        self.model = model
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    @property
    def _vec_path(self) -> Path:
        return self.dir / "image_vectors.f16"

    @property
    def _id_path(self) -> Path:
        return self.dir / "image_ids.i64"

    @property
    def _meta_path(self) -> Path:
        return self.dir / "meta.json"

    def build(self, conn: sqlite3.Connection, batch: int = 8192) -> int:
        """Rebuild from photo_embedding. Streams so memory stays flat."""
        total = conn.execute(
            "SELECT COUNT(*) FROM photo_embedding WHERE model=?", (self.model,)).fetchone()[0]
        if total == 0:
            for p in (self._vec_path, self._id_path, self._meta_path):
                p.unlink(missing_ok=True)
            self._vectors = self._ids = None
            return 0

        vecs = np.lib.format.open_memmap(
            self._vec_path, mode="w+", dtype=np.float16, shape=(total, self.dim))
        ids = np.empty(total, dtype=np.int64)
        offset = 0
        cur = conn.execute(
            "SELECT photo_id, embedding FROM photo_embedding WHERE model=? ORDER BY photo_id",
            (self.model,))
        while rows := cur.fetchmany(batch):
            block = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            vecs[offset:offset + len(rows)] = block.astype(np.float16)
            ids[offset:offset + len(rows)] = [r["photo_id"] for r in rows]
            offset += len(rows)
        vecs.flush()
        ids.tofile(self._id_path)
        self._meta_path.write_text(json.dumps(
            {"model": self.model, "dim": self.dim, "count": total}))
        self._vectors = self._ids = None
        return total

    def _ensure_loaded(self) -> bool:
        if self._vectors is not None:
            return True
        if not self._vec_path.exists() or not self._meta_path.exists():
            return False
        meta = json.loads(self._meta_path.read_text())
        if meta.get("model") != self.model or meta.get("dim") != self.dim:
            return False  # embedding model changed; index is stale until rebuilt
        self._vectors = np.load(self._vec_path, mmap_mode="r")
        self._ids = np.fromfile(self._id_path, dtype=np.int64)
        return True

    def search(self, query_vec: np.ndarray, limit: int = 200,
               min_score: float = 0.0, rel_score: float = 0.0) -> list[int]:
        """Photo ids ranked by cosine similarity, best first.

        Nearest-neighbour search always returns *something*, so without a cutoff
        "a dog on a beach" ranks every screenshot you own and the fusion step
        then presents the whole library as results. Two cutoffs, because either
        alone fails:

        `min_score` -- an absolute floor. SigLIP's sigmoid training objective
        gives its scores a meaningful absolute scale (a real match lands around
        0.12, unrelated content below 0.06), so "nothing here looks like that"
        becomes expressible. Fails when a query scores high across the board.

        `rel_score` -- a fraction of the best hit. Handles queries whose whole
        distribution is shifted up, keeping the good matches and dropping the
        long mediocre tail. Fails alone when nothing matches, since the best of
        a bad set still passes.
        """
        if not self._ensure_loaded() or self._vectors is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        scores = self._vectors.astype(np.float32) @ q
        k = min(limit, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        if len(top) == 0:
            return []
        cutoff = max(min_score, float(scores[top[0]]) * rel_score)
        return [int(self._ids[i]) for i in top if scores[i] >= cutoff]

    def count(self) -> int:
        if not self._meta_path.exists():
            return 0
        return int(json.loads(self._meta_path.read_text()).get("count", 0))
