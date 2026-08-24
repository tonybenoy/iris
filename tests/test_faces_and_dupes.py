"""Correcting what the machine guessed about people, and about duplicates.

Naming is the most valuable action in the app, and it only works if the queue
can actually be emptied. Group photos are mostly strangers, so without a way to
say "not someone I need to name" every clustering run proposes the same faces
again and the screen never finishes.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pa.api.app import create_app
from pa.config import Config
from pa.duplicates import same_shot_different_format


@pytest.fixture
def client(tmp_path, monkeypatch):
    import pa.config as cfgmod

    cfg = Config()
    cfg.paths.data_dir = tmp_path
    cfg.paths.cache_dir = tmp_path / "cache"
    for d in (cfg.paths.data_dir, cfg.paths.cache_dir, cfg.paths.thumbs_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfgmod, "_cfg", cfg)

    from pa.db.connection import init_db
    conn = init_db(cfg.paths.db_path)
    vec = np.zeros(512, dtype=np.float32).tobytes()
    for photo_id in (1, 2):
        conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (?,?,0)",
                     (photo_id, f"hash{photo_id}"))
    # Two proposed groups, as clustering would leave them.
    for face_id, cluster in ((1, 7), (2, 7), (3, 9)):
        conn.execute(
            """INSERT INTO face (id, photo_id, cluster_id, bbox_x, bbox_y, bbox_w, bbox_h,
                                 det_score, embedding, model, created_at)
               VALUES (?,1,?,0,0,10,10,0.9,?, 'test', 0)""", (face_id, cluster, vec))
    conn.commit()
    return TestClient(create_app())


def _queue(client):
    return {c["cluster_id"] for c in client.get("/api/clusters").json()["clusters"]}


# ----------------------------------------------------------------- ignoring
def test_ignoring_a_group_clears_it_from_the_queue(client):
    assert _queue(client) == {7, 9}
    r = client.post("/api/clusters/7/ignore")
    assert r.status_code == 200, r.text
    assert r.json()["ignored"] == 2
    assert _queue(client) == {9}, "an ignored group must not still ask to be named"


def test_ignored_faces_are_listed_so_it_can_be_undone(client):
    client.post("/api/clusters/7/ignore")
    d = client.get("/api/faces/ignored").json()
    assert [g["cluster_id"] for g in d["groups"]] == [7]
    assert d["groups"][0]["count"] == 2


def test_restoring_puts_the_same_group_back(client):
    client.post("/api/clusters/7/ignore")
    assert client.post("/api/faces/ignored/7/restore").json()["restored"] == 2
    assert _queue(client) == {7, 9}, "undo must restore the group, not loose faces"
    assert client.get("/api/faces/ignored").json()["groups"] == []


def test_ignoring_one_face_leaves_the_rest_of_its_group(client):
    assert client.post("/api/faces/1/ignore").status_code == 200
    clusters = {c["cluster_id"]: c["count"] for c in
                client.get("/api/clusters").json()["clusters"]}
    assert clusters == {7: 1, 9: 1}
    # It had no group of its own, so it is counted loose rather than restorable.
    d = client.get("/api/faces/ignored").json()
    assert d["loose"] == 1 and d["groups"] == []


def test_ignored_faces_stay_out_of_reclustering(client, tmp_path):
    """The whole point. If clustering can see them again, ignoring achieves
    nothing beyond a single screen refresh."""
    from pa.db.connection import init_db
    from pa.faces.cluster import _load_faces

    client.post("/api/clusters/7/ignore")
    conn = init_db(Config(paths={"data_dir": tmp_path}).paths.db_path)
    _, rows = _load_faces(conn)
    assert {r["id"] for r in rows} == {3}


def test_ignoring_an_unknown_group_is_a_404(client):
    assert client.post("/api/clusters/999/ignore").status_code == 404
    assert client.post("/api/faces/ignored/999/restore").status_code == 404


# -------------------------------------------------------------------- merge
def test_merging_folds_groups_together(client):
    r = client.post("/api/clusters/merge", json={"cluster_ids": [7, 9]})
    assert r.status_code == 200, r.text
    assert _queue(client) == {7}, "merged groups become one, keeping the lowest id"
    assert client.get("/api/clusters").json()["clusters"][0]["count"] == 3


def test_merging_with_a_name_creates_the_person(client):
    client.post("/api/clusters/merge", json={"cluster_ids": [7, 9], "name": "Sarah"})
    assert [p["name"] for p in client.get("/api/people").json()["people"]] == ["Sarah"]
    assert _queue(client) == set()


def test_merging_needs_two_groups(client):
    assert client.post("/api/clusters/merge", json={"cluster_ids": [7]}).status_code == 400


# --------------------------------------------------------------- duplicates
@pytest.mark.parametrize("names, expected", [
    (["DSC_0042.NEF", "DSC_0042.JPG"], True),       # camera shooting RAW+JPEG
    (["IMG_1234.HEIC", "IMG_1234.jpg"], True),      # what a phone does
    (["dsc_0042.nef", "DSC_0042.JPG"], True),       # case must not matter
    (["holiday.jpg", "holiday.jpg"], False),        # genuinely the same file twice
    (["cover.jpg", "cover.png"], False),            # no RAW involved: still report it
    (["a.NEF", "b.JPG"], False),                    # different shots
    (["DSC_0042.NEF", "DSC_0042.JPG", "DSC_0042.CR2"], True),
    (["noextension", "DSC_0042.JPG"], False),
])
def test_raw_and_jpeg_of_one_shot_is_not_a_duplicate(names, expected):
    assert same_shot_different_format(names) is expected


def test_near_duplicates_skips_format_pairs(client, tmp_path):
    """A NEF and its JPEG have all but identical pHashes, so they land in the
    same near-duplicate group and used to be reported as waste."""
    from pa.db.connection import init_db
    from pa.duplicates import near_duplicates

    conn = init_db(Config(paths={"data_dir": tmp_path}).paths.db_path)
    conn.execute("UPDATE photo SET phash=? WHERE id=1", (12345678,))
    conn.execute("UPDATE photo SET phash=? WHERE id=2", (12345678,))
    conn.execute("INSERT INTO volume (id, uuid, label) VALUES (1,'u','v')")
    conn.execute("INSERT INTO root (id, volume_id, rel_path, added_at) VALUES (1,1,'',0)")
    for photo_id, name in ((1, "DSC_0042.NEF"), (2, "DSC_0042.JPG")):
        conn.execute(
            """INSERT INTO file (photo_id, root_id, rel_path, filename, mtime, size,
                                 state, seen_at)
               VALUES (?,1,?,?,0,1,'present',0)""", (photo_id, name, name))
    conn.commit()

    groups, skipped = near_duplicates(conn, 6)
    assert groups == [] and skipped == 1

    groups, skipped = near_duplicates(conn, 6, include_format_pairs=True)
    assert len(groups) == 1 and groups[0]["format_pair"] is True


# --------------------------------------------------------------- migrations
def test_an_existing_library_upgrades_without_losing_anything(tmp_path):
    """schema.sql names columns that older databases only reach by migration.

    It used to be executed on every open, before migrations ran, so its
    CREATE INDEX on a migration-added column raised "no such column" and no
    library from a previous version could be opened at all.
    """
    from pa.db.connection import SCHEMA_VERSION, init_db

    db = tmp_path / "library.db"
    conn = init_db(db)
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'keepme',0)")
    # Wind the database back to what version 2 actually looked like, rather than
    # relabelling a current one -- the bug was in opening a genuinely older shape.
    conn.execute("DROP INDEX IF EXISTS idx_face_ignored")
    conn.execute("ALTER TABLE face DROP COLUMN ignored_as")
    conn.execute("UPDATE schema_version SET version=2")
    conn.commit()
    conn.close()

    import pa.db.connection as conmod
    conmod._local.__dict__.clear()  # force a fresh handle, not the cached one

    conn = init_db(db)
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("SELECT blake3 FROM photo WHERE id=1").fetchone()[0] == "keepme"
    assert "ignored_as" in [r[1] for r in conn.execute("PRAGMA table_info(face)")]


# ------------------------------------------------------- merging named people
def test_naming_a_group_with_an_existing_name_joins_that_person(client):
    """The common correction: two groups are the same person already named."""
    client.post("/api/clusters/7/name", json={"name": "Sarah"})
    client.post("/api/clusters/9/name", json={"name": "Sarah"})
    people = client.get("/api/people").json()["people"]
    assert len(people) == 1, "one name must mean one person, not two rows"
    assert people[0]["n"] == 3


def test_merging_two_named_people(client):
    client.post("/api/clusters/7/name", json={"name": "Sarah"})
    client.post("/api/clusters/9/name", json={"name": "Sarah B"})
    ids = {p["name"]: p["id"] for p in client.get("/api/people").json()["people"]}

    r = client.post(f"/api/people/{ids['Sarah B']}/merge?into={ids['Sarah']}")
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "moved": 1, "name": "Sarah"}

    people = client.get("/api/people").json()["people"]
    assert [(p["name"], p["n"]) for p in people] == [("Sarah", 3)]


def test_renaming_onto_a_taken_name_offers_the_merge(client):
    """Refusing with a bare conflict left no way to fix a split person."""
    client.post("/api/clusters/7/name", json={"name": "Sarah"})
    client.post("/api/clusters/9/name", json={"name": "Sarah B"})
    ids = {p["name"]: p["id"] for p in client.get("/api/people").json()["people"]}

    r = client.patch(f"/api/people/{ids['Sarah B']}", json={"name": "Sarah"})
    assert r.status_code == 409
    assert r.json()["detail"]["merge_into"] == ids["Sarah"]


def test_a_person_cannot_be_merged_into_themselves(client):
    client.post("/api/clusters/7/name", json={"name": "Sarah"})
    pid = client.get("/api/people").json()["people"][0]["id"]
    assert client.post(f"/api/people/{pid}/merge?into={pid}").status_code == 400
    assert client.post(f"/api/people/{pid}/merge?into=9999").status_code == 404
