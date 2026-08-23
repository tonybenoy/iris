"""The cutoff logic is what stops a nearest-neighbour search from returning the
entire library for a query nothing matches."""
import numpy as np
import pytest

from pa.search.vectors import VectorIndex


class FakeIndex(VectorIndex):
    def __init__(self, scores):
        self._fake = np.asarray(scores, dtype=np.float32)
        self._ids = np.arange(len(scores), dtype=np.int64)
        self.dim = 1

    def _ensure_loaded(self):
        self._vectors = self._fake.reshape(-1, 1)
        return True


def run(scores, min_score=0.07, rel_score=0.6, limit=100):
    q = np.array([1.0], dtype=np.float32)
    return FakeIndex(scores).search(q, limit, min_score, rel_score)


def test_nothing_matches_returns_nothing():
    # Real measurement: "a dog on a beach" against a library of screenshots.
    assert run([0.012, 0.008, -0.03, 0.001]) == []


def test_weak_best_match_still_rejected():
    # "mountains" topped out at 0.0555 with no mountain in the library.
    assert run([0.0555, 0.0508, 0.03]) == []


def test_genuine_matches_survive():
    # "a person outdoors": three real hits well clear of the noise floor.
    assert run([0.125, 0.122, 0.118, 0.02, -0.01]) == [0, 1, 2]


def test_relative_cutoff_trims_mediocre_tail():
    # All above the absolute floor, but only the top cluster is a real match.
    assert run([0.30, 0.29, 0.09, 0.08]) == [0, 1]


def test_results_are_ranked_best_first():
    assert run([0.10, 0.30, 0.20], min_score=0.0, rel_score=0.0) == [1, 2, 0]


@pytest.mark.parametrize("scores", [[], [0.5]])
def test_edge_sizes(scores):
    got = run(scores)
    assert got == ([] if not scores else [0])


def test_no_match_means_no_results_not_whole_library(tmp_path):
    """A free-text query both engines reject must return nothing.

    Regression: an over-eager fallback answered "mountains" with all 133 photos
    in the library, which is worse than an honest empty result.
    """
    import sqlite3

    from pa.db.connection import init_db
    from pa.search.query import search

    conn = init_db(tmp_path / "t.db")
    conn.execute("INSERT INTO photo (blake3, created_at) VALUES ('abc', 0)")
    # A photo needs at least one present file to be searchable at all: search
    # excludes photos whose every copy has been deleted from disk.
    conn.execute("INSERT INTO volume (id, uuid, label) VALUES (1,'u','Disk')")
    conn.execute("INSERT INTO root (id, volume_id, rel_path, added_at) VALUES (1,1,'r',0)")
    conn.execute(
        """INSERT INTO file (photo_id, root_id, rel_path, filename, mtime, size, seen_at)
           VALUES (1,1,'a.png','a.png',0,0,0)""")
    conn.execute("INSERT INTO photo_fts (rowid, caption, tags, ocr_text, people, "
                 "filename, folder, place) VALUES (1,'a screenshot','ui','','','a.png','','')")
    conn.commit()

    assert search(conn, "screenshot", vector_search=lambda t, n: [1]), "real match should hit"
    assert search(conn, "mountains", vector_search=lambda t, n: []) == [], \
        "no engine matched: must return empty, not the library"
    assert isinstance(conn, sqlite3.Connection)
