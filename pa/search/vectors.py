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

import contextlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

# The names an older library wrote before builds were generation-stamped.
LEGACY_VECTORS = "image_vectors.f16"
LEGACY_IDS = "image_ids.i64"


class VectorIndex:
    """Reader and writer of the index files.

    Several of these exist at once over the same directory -- the web app holds
    one for searching, the indexer makes its own to rebuild with, and the CLI a
    third -- which is why a build never writes over the file a reader may have
    mapped. See build().
    """

    def __init__(self, directory: Path, dim: int, model: str):
        self.dir = directory
        self.dim = dim
        self.model = model
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._loaded: tuple[str, str] | None = None   # the files currently mapped

    @property
    def _meta_path(self) -> Path:
        return self.dir / "meta.json"

    def _meta(self) -> dict:
        if not self._meta_path.exists():
            return {}
        try:
            return json.loads(self._meta_path.read_text())
        except (OSError, ValueError):
            return {}   # half-written by a build that was killed

    def release(self) -> None:
        """Drop the memory map. On Windows a mapped file cannot be replaced or
        deleted, so anything about to rewrite the index has to let go first."""
        self._vectors = self._ids = None
        self._loaded = None

    def build(self, conn: sqlite3.Connection, batch: int = 8192) -> int:
        """Rebuild from photo_embedding. Streams so memory stays flat.

        Each build writes new files under a fresh name and points meta.json at
        them, rather than overwriting the ones already there.

        Writing in place is fine on Linux, where unlinking a mapped file just
        defers the delete until the last reader closes it. Windows has no such
        thing: a file another handle has mapped cannot be truncated, replaced or
        removed, and any search since the server started leaves exactly such a
        map behind. Rebuilding then died with "OSError: [Errno 22] Invalid
        argument" on image_vectors.f16 -- which, being how a redo of the embed
        stage finishes, made the whole model change impossible to complete.

        Naming each build separately means a reader never blocks a writer. The
        superseded files are deleted afterwards if they can be, and otherwise
        left for the next build to collect.
        """
        self.release()
        total = conn.execute(
            "SELECT COUNT(*) FROM photo_embedding WHERE model=?", (self.model,)).fetchone()[0]
        if total == 0:
            with contextlib.suppress(OSError):
                self._meta_path.unlink(missing_ok=True)
            self._sweep(keep=())
            return 0

        stamp = time.time_ns()
        vec_name, id_name = f"image_vectors.{stamp}.f16", f"image_ids.{stamp}.i64"
        vecs = np.lib.format.open_memmap(
            self.dir / vec_name, mode="w+", dtype=np.float16, shape=(total, self.dim))
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
        del vecs   # closes this build's own map before anything tries to tidy up
        ids.tofile(self.dir / id_name)
        # Written last: until meta.json names them, the new files are invisible
        # and a reader carries on with the old index rather than half of this one.
        self._meta_path.write_text(json.dumps(
            {"model": self.model, "dim": self.dim, "count": total,
             "vectors": vec_name, "ids": id_name}))
        self._sweep(keep=(vec_name, id_name))
        return total

    def _sweep(self, keep: tuple[str, ...]) -> None:
        """Delete superseded index files, best effort.

        A file some other reader still has mapped will refuse to go on Windows.
        That is not worth failing a build over: it is a few megabytes until the
        next one, which will find it unmapped and take it then.
        """
        for path in [*self.dir.glob("image_vectors.*"), *self.dir.glob("image_ids.*"),
                     self.dir / LEGACY_VECTORS, self.dir / LEGACY_IDS]:
            if path.name not in keep:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)

    def _ensure_loaded(self) -> bool:
        """Map the files meta.json currently names, if they are not already.

        Checked on every search rather than once, because a build now writes new
        files instead of overwriting these -- so an instance that mapped the old
        ones would go on answering from them forever. The web app holds exactly
        such an instance for the life of the process, which would mean searching
        a freshly re-embedded library and getting the previous model's answers.
        Reading one small file per query costs nothing next to the matrix
        multiply it precedes.
        """
        meta = self._meta()
        if meta.get("model") != self.model or meta.get("dim") != self.dim:
            return False  # embedding model changed; index is stale until rebuilt
        names = (meta.get("vectors", LEGACY_VECTORS), meta.get("ids", LEGACY_IDS))
        if self._vectors is not None and self._loaded == names:
            return True
        vec_path, id_path = self.dir / names[0], self.dir / names[1]
        if not vec_path.exists() or not id_path.exists():
            return False
        try:
            self._vectors = np.load(vec_path, mmap_mode="r")
            self._ids = np.fromfile(id_path, dtype=np.int64)
        except (OSError, ValueError):
            # Truncated or unreadable: no index is a working search that finds
            # less, where a raised error is a search screen that shows nothing.
            self.release()
            return False
        self._loaded = names
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
        return int(self._meta().get("count", 0))
