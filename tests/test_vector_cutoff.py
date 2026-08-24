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


# ------------------------------------------------------- rebuilding the index
# Several VectorIndex objects live over one directory at once: the web app holds
# one for searching, the indexer makes its own to rebuild with, the CLI a third.
# On Linux a build could overwrite the files under a reader and nothing minded.
# On Windows a mapped file cannot be truncated, replaced or deleted, so the same
# build raised "OSError: [Errno 22] Invalid argument" -- which is how a redo of
# the embed stage finishes, making a model change impossible to complete.
@pytest.fixture
def library(tmp_path):
    from pa.db.connection import init_db

    conn = init_db(tmp_path / "library.db")
    for photo_id in (1, 2, 3):
        conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (?,?,0)",
                     (photo_id, f"hash{photo_id}"))
    conn.commit()
    return conn


def _embed(conn, photo_id, vector, model="m1"):
    vec = np.asarray(vector, dtype=np.float32)
    conn.execute("INSERT INTO photo_embedding (photo_id, embedding, model, dim, created_at) "
                 "VALUES (?,?,?,?,0)", (photo_id, vec.tobytes(), model, len(vec)))
    conn.commit()


def test_a_build_does_not_write_over_a_file_a_reader_has_mapped(tmp_path, library):
    """The Windows failure cannot be reproduced on Linux -- there, overwriting a
    mapped file is allowed -- so this asserts the mechanism that avoids it: the
    second build must land somewhere the first one is not."""
    import json

    vectors = tmp_path / "vectors"
    reader = VectorIndex(vectors, 2, "m1")
    _embed(library, 1, [1.0, 0.0])
    VectorIndex(vectors, 2, "m1").build(library)

    # Searching is what maps the file, and that map is what blocks the rebuild.
    assert reader.search(np.array([1.0, 0.0], dtype=np.float32)) == [1]
    was = json.loads((vectors / "meta.json").read_text())["vectors"]

    _embed(library, 2, [0.0, 1.0])
    assert VectorIndex(vectors, 2, "m1").build(library) == 2
    now = json.loads((vectors / "meta.json").read_text())["vectors"]
    assert now != was, "a rebuild wrote over the file a reader had open"


def test_a_reader_picks_up_the_rebuild(tmp_path, library):
    """The web app keeps one index for the life of the process. Mapping the old
    files forever would mean searching a re-embedded library and being answered
    by the model it replaced."""
    reader = VectorIndex(tmp_path / "vectors", 2, "m1")
    _embed(library, 1, [1.0, 0.0])
    VectorIndex(tmp_path / "vectors", 2, "m1").build(library)
    query = np.array([0.0, 1.0], dtype=np.float32)
    assert reader.search(query, min_score=0.5) == []

    _embed(library, 2, [0.0, 1.0])
    VectorIndex(tmp_path / "vectors", 2, "m1").build(library)
    assert reader.search(query, min_score=0.5) == [2], "still answering from the old files"


def test_superseded_files_are_cleared_away(tmp_path, library):
    vectors = tmp_path / "vectors"
    _embed(library, 1, [1.0, 0.0])
    for _ in range(3):
        VectorIndex(vectors, 2, "m1").build(library)
    assert len(list(vectors.glob("image_vectors.*"))) == 1
    assert len(list(vectors.glob("image_ids.*"))) == 1


def test_an_index_from_an_older_version_still_loads(tmp_path, library):
    """meta.json used to name no files at all -- the two were always called the
    same thing. An upgrade must not silently lose the index it already had."""
    import json

    from pa.search.vectors import LEGACY_IDS, LEGACY_VECTORS

    vectors = tmp_path / "vectors"
    _embed(library, 1, [1.0, 0.0])
    VectorIndex(vectors, 2, "m1").build(library)
    # Rename it back to what the previous version wrote.
    meta = json.loads((vectors / "meta.json").read_text())
    (vectors / meta["vectors"]).rename(vectors / LEGACY_VECTORS)
    (vectors / meta["ids"]).rename(vectors / LEGACY_IDS)
    (vectors / "meta.json").write_text(json.dumps(
        {"model": "m1", "dim": 2, "count": 1}))

    assert VectorIndex(vectors, 2, "m1").search(
        np.array([1.0, 0.0], dtype=np.float32)) == [1]


def test_an_empty_library_leaves_no_index_behind(tmp_path, library):
    vectors = tmp_path / "vectors"
    _embed(library, 1, [1.0, 0.0])
    VectorIndex(vectors, 2, "m1").build(library)
    library.execute("DELETE FROM photo_embedding")
    library.commit()

    index = VectorIndex(vectors, 2, "m1")
    assert index.build(library) == 0
    assert index.count() == 0
    assert list(vectors.glob("image_*")) == []


def test_a_half_written_meta_is_not_an_error(tmp_path, library):
    """A build killed part way used to take search down with it."""
    vectors = tmp_path / "vectors"
    vectors.mkdir(parents=True)
    (vectors / "meta.json").write_text('{"model": "m1", "dim"')
    index = VectorIndex(vectors, 2, "m1")
    assert index.count() == 0
    assert index.search(np.array([1.0, 0.0], dtype=np.float32)) == []
