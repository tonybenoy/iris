"""Removing a folder must not leave photos that point at nothing."""
from pa.db import repo
from pa.db.connection import init_db


def _library(tmp_path):
    conn = init_db(tmp_path / "t.db")
    conn.execute("INSERT INTO volume (id, uuid, label) VALUES (1,'u','Disk')")
    conn.execute("INSERT INTO root (id, volume_id, rel_path, added_at) VALUES (1,1,'a',0)")
    conn.execute("INSERT INTO root (id, volume_id, rel_path, added_at) VALUES (2,1,'b',0)")
    # Photo 1 lives only in root 1; photo 2 is the same image backed up in both.
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h1',0)")
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (2,'h2',0)")
    for pid, rid, name in [(1, 1, "one.jpg"), (2, 1, "two.jpg"), (2, 2, "two-backup.jpg")]:
        conn.execute(
            """INSERT INTO file (photo_id, root_id, rel_path, filename, mtime, size, seen_at)
               VALUES (?,?,?,?,0,0,0)""", (pid, rid, name, name))
    conn.commit()
    return conn


def test_removing_a_root_drops_only_photos_with_no_other_copy(tmp_path):
    conn = _library(tmp_path)
    files, orphans = repo.remove_root(conn, 1)
    conn.commit()
    assert files == 2
    assert orphans == 1, "photo 1 existed only in that root"
    remaining = [r["id"] for r in conn.execute("SELECT id FROM photo ORDER BY id")]
    assert remaining == [2], "the backed-up photo must survive"


def test_prune_is_idempotent(tmp_path):
    conn = _library(tmp_path)
    repo.remove_root(conn, 1)
    conn.commit()
    assert repo.prune_orphans(conn) == 0


def test_prune_leaves_a_healthy_library_alone(tmp_path):
    conn = _library(tmp_path)
    assert repo.prune_orphans(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM photo").fetchone()[0] == 2


def test_deleted_photos_leave_search_but_offline_drives_do_not(tmp_path):
    """A deleted file and an unplugged drive look similar in the database and
    must not be treated the same: one is gone for good, the other is coming
    back when you reconnect it."""
    from pa.search.query import search

    conn = _library(tmp_path)
    conn.execute("INSERT INTO photo_fts (rowid, caption, tags, ocr_text, people, "
                 "filename, folder, place) VALUES (1,'a beach','beach','','','one.jpg','','')")
    conn.execute("INSERT INTO photo_fts (rowid, caption, tags, ocr_text, people, "
                 "filename, folder, place) VALUES (2,'a beach','beach','','','two.jpg','','')")
    conn.commit()
    assert len(search(conn, "beach")) == 2

    conn.execute("UPDATE file SET state='offline' WHERE photo_id=1")
    conn.commit()
    assert len(search(conn, "beach")) == 2, "an unplugged drive must still be searchable"

    conn.execute("UPDATE file SET state='missing' WHERE photo_id=1")
    conn.commit()
    assert len(search(conn, "beach")) == 1, "a deleted photo must drop out of search"


def test_prune_can_keep_missing_photos(tmp_path):
    conn = _library(tmp_path)
    conn.execute("UPDATE file SET state='missing' WHERE photo_id=1")
    conn.commit()
    assert repo.prune_orphans(conn, drop_missing=False) == 0
    assert repo.prune_orphans(conn, drop_missing=True) == 1


def test_a_hand_edit_outranks_a_later_model_run(tmp_path):
    """Re-captioning a library must never quietly shadow something you wrote.

    Regression: the display picked the newest annotation row, so re-running the
    caption stage buried the user's edit under fresh model output. The edit was
    still in the database, which made it look preserved while being invisible.
    """
    from pa.db.connection import init_db
    from pa.search.query import search

    conn = init_db(tmp_path / "t.db")
    conn.execute("INSERT INTO volume (id, uuid, label) VALUES (1,'u','Disk')")
    conn.execute("INSERT INTO root (id, volume_id, rel_path, added_at) VALUES (1,1,'r',0)")
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h',0)")
    conn.execute(
        """INSERT INTO file (photo_id, root_id, rel_path, filename, mtime, size, seen_at)
           VALUES (1,1,'a.jpg','a.jpg',0,0,0)""")

    repo.save_annotation(conn, 1, {"caption": "a yellow building"}, "gemma", "v1")
    repo.save_annotation(conn, 1, {"caption": "Sarus cranes at dawn"}, "manual", "edited")
    # The model runs again afterwards and writes a newer row.
    repo.save_annotation(conn, 1, {"caption": "a yellow building again"}, "gemma", "v2")
    repo.reindex_fts(conn, 1)
    conn.commit()

    # A vector stub keeps this out of the FTS-only degraded mode, where a query
    # that matches nothing deliberately falls back to the filters.
    def vec(text, n):
        return []

    hits = search(conn, "cranes", vector_search=vec)
    assert hits and hits[0].caption == "Sarus cranes at dawn", \
        "the hand edit must still be the caption after a later model run"

    conn.execute("DELETE FROM annotation WHERE photo_id=1 AND model='manual'")
    repo.reindex_fts(conn, 1)
    conn.commit()
    assert search(conn, "cranes", vector_search=vec) == [], \
        "reverting must fall back to the model"
