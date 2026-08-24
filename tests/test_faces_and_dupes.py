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


# ------------------------------------------------------------- face crops
def _make_view_thumb(cfg, digest: str, size: tuple[int, int]) -> None:
    from PIL import Image

    from pa.ingest import thumbs
    path = thumbs.thumb_path(cfg.paths.thumbs_dir, digest, "view",
                             cfg.thumbs.format.lower())
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "grey").save(path, cfg.thumbs.format)


def test_face_crop_rescales_a_box_detected_on_a_bigger_image(client, tmp_path):
    """Faces are detected on whatever _stage_source hands over -- the original
    if no thumbnail existed yet -- but crops always come from the view
    thumbnail. Cropping with the detector's coordinates then asks for a region
    far outside a 400px image, and PIL raises "right is less than left": a 500
    on every face card on the People screen."""
    import pa.config as cfgmod

    cfg = cfgmod._cfg
    _make_view_thumb(cfg, "hash1", (400, 300))

    from pa.db.connection import init_db
    conn = init_db(cfg.paths.db_path)
    # As if detected on a 6000x4000 original: a face near the right-hand edge.
    conn.execute("""UPDATE face SET bbox_x=5200, bbox_y=1000, bbox_w=400, bbox_h=400,
                                    src_w=6000, src_h=4000 WHERE id=1""")
    conn.commit()

    r = client.get("/api/face/1")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/webp"


def test_face_crop_survives_a_box_it_cannot_place(client, tmp_path):
    """Libraries indexed before src_w was recorded have no way to rescale. A
    wrong crop is still nameable; a 500 is not."""
    import pa.config as cfgmod

    cfg = cfgmod._cfg
    _make_view_thumb(cfg, "hash1", (400, 300))

    from pa.db.connection import init_db
    conn = init_db(cfg.paths.db_path)
    conn.execute("""UPDATE face SET bbox_x=5200, bbox_y=-40, bbox_w=400, bbox_h=400,
                                    src_w=NULL, src_h=NULL WHERE id=1""")
    conn.commit()

    assert client.get("/api/face/1").status_code == 200


# ------------------------------------------------------- suggesting a name
def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _face_vectors(conn, vectors: dict[int, np.ndarray]) -> None:
    for face_id, vec in vectors.items():
        conn.execute("UPDATE face SET embedding=? WHERE id=?",
                     (vec.astype(np.float32).tobytes(), face_id))
    conn.commit()


def test_a_group_that_resembles_a_named_person_is_suggested(client, tmp_path):
    """Recognising a name you are shown is far easier than recalling one from a
    face crop, and it is what stops one person being entered twice."""
    import pa.config as cfgmod
    from pa.db.connection import init_db
    conn = init_db(cfgmod._cfg.paths.db_path)
    sarah, stranger = _unit(1), _unit(2)
    _face_vectors(conn, {1: sarah, 2: sarah, 3: stranger})
    client.post("/api/clusters/7/name", json={"name": "Sarah"})

    # A third group, half-way between the two: too unsure for clustering to
    # attach on its own, which is exactly the case a suggestion is for.
    mix = (sarah + stranger) / 2
    conn.execute(
        """INSERT INTO face (id, photo_id, cluster_id, bbox_x, bbox_y, bbox_w, bbox_h,
                             det_score, embedding, model, created_at)
           VALUES (11,1,11,0,0,10,10,0.9,?,'test',0)""",
        ((mix / np.linalg.norm(mix)).astype(np.float32).tobytes(),))
    conn.commit()

    by_id = {c["cluster_id"]: c for c in client.get("/api/clusters").json()["clusters"]}
    assert [g["name"] for g in by_id[11]["suggestions"]] == ["Sarah"]
    assert by_id[11]["suggestions"][0]["score"] > 0.5
    assert by_id[9]["suggestions"] == [], "an unrelated face must not be guessed at"


def test_nothing_is_suggested_before_anyone_is_named(client):
    for c in client.get("/api/clusters").json()["clusters"]:
        assert c["suggestions"] == []


def test_a_weak_resemblance_is_left_alone(client, tmp_path):
    """The floor is a setting because the cost of a wrong guess is a whole group
    filed under the wrong person."""
    import pa.config as cfgmod
    from pa.db.connection import init_db
    cfg = cfgmod._cfg
    conn = init_db(cfg.paths.db_path)
    _face_vectors(conn, {1: _unit(1), 2: _unit(1), 3: _unit(2)})
    client.post("/api/clusters/7/name", json={"name": "Sarah"})

    cfg.face.suggest_min_similarity = 0.99
    assert client.get("/api/clusters").json()["clusters"][0]["suggestions"] == []
    cfg.face.suggest_min_similarity = -1.0
    assert client.get("/api/clusters").json()["clusters"][0]["suggestions"] != []


# ------------------------------------------------ seeing a group as photos
def test_a_group_lists_the_photos_its_faces_came_from(client):
    """Five crops are enough to say "that is Sarah" and not enough to notice
    that the group is a face on a poster."""
    d = client.get("/api/clusters/7/faces").json()
    assert d["cluster_id"] == 7
    assert [f["id"] for f in d["faces"]] == [1, 2]
    assert all(f["photo_id"] == 1 and f["blake3"] == "hash1" for f in d["faces"])


def test_an_unknown_group_has_no_photos(client):
    assert client.get("/api/clusters/999/faces").status_code == 404


# ------------------------------------------- the same face, detected twice
# Faces you named or ignored survive a re-run of detection on purpose. The
# detector then finds them again, and inserting that result puts a second copy
# of every decided face on the photo -- the person listed twice, their count
# doubled, and every dismissed stranger back in the naming queue.


def _add_face(conn, face_id, photo_id, box, *, confirmed=0, rejected=0, person=None,
              src=(1000, 1000)):
    vec = np.zeros(512, dtype=np.float32).tobytes()
    conn.execute(
        """INSERT INTO face (id, photo_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                             det_score, embedding, model, src_w, src_h,
                             confirmed, rejected, created_at)
           VALUES (?,?,?,?,?,?,?,0.9,?,'test',?,?,?,?,0)""",
        (face_id, photo_id, person, *box, vec, src[0], src[1], confirmed, rejected))
    conn.commit()


def test_a_redetection_of_a_named_face_is_recognised(client, tmp_path):
    import pa.config as cfgmod
    from pa.db.connection import init_db
    from pa.faces.dedupe import decided_boxes, is_already_decided
    conn = init_db(cfgmod._cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    _add_face(conn, 20, 1, (100, 100, 200, 200), confirmed=1)

    decided = decided_boxes(conn, 1)
    # The same face found again: identical, and jittered as a detector would.
    assert is_already_decided((100, 100, 200, 200), (1000, 1000), decided)
    assert is_already_decided((104, 96, 198, 205), (1000, 1000), decided)
    # A different person standing next to them is not the same face.
    assert not is_already_decided((400, 100, 200, 200), (1000, 1000), decided)


def test_a_box_measured_against_another_size_still_matches(client, tmp_path):
    """Detection runs on the view thumbnail when there is one and the original
    when there is not, so the two copies of one face can be recorded in
    different coordinate spaces."""
    import pa.config as cfgmod
    from pa.db.connection import init_db
    from pa.faces.dedupe import decided_boxes, is_already_decided
    conn = init_db(cfgmod._cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    _add_face(conn, 21, 1, (100, 100, 200, 200), confirmed=1, src=(1000, 1000))

    # The same face on a 4000px original: four times the numbers.
    assert is_already_decided((400, 400, 800, 800), (4000, 4000), decided_boxes(conn, 1))


def test_cleaning_up_keeps_every_decision(client, tmp_path):
    import pa.config as cfgmod
    from pa.db.connection import init_db
    conn = init_db(cfgmod._cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    _add_face(conn, 30, 1, (100, 100, 200, 200), confirmed=1)      # named
    _add_face(conn, 31, 1, (600, 100, 200, 200), rejected=1)       # ignored
    _add_face(conn, 32, 1, (100, 100, 200, 200))                   # copy of the named one
    _add_face(conn, 33, 1, (600, 100, 200, 200))                   # copy of the ignored one
    _add_face(conn, 34, 1, (100, 700, 150, 150))                   # a genuinely new face

    assert client.post("/api/faces/dedupe").json()["removed"] == 2
    assert {r[0] for r in conn.execute("SELECT id FROM face")} == {30, 31, 34}


def test_cleaning_up_twice_removes_nothing_the_second_time(client, tmp_path):
    import pa.config as cfgmod
    from pa.db.connection import init_db
    conn = init_db(cfgmod._cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    _add_face(conn, 40, 1, (100, 100, 200, 200), confirmed=1)
    _add_face(conn, 41, 1, (100, 100, 200, 200))

    assert client.post("/api/faces/dedupe").json()["removed"] == 1
    assert client.post("/api/faces/dedupe").json()["removed"] == 0


def test_running_detection_again_does_not_add_a_second_copy(client, tmp_path):
    """The end to end version: the stage itself must not re-add what it kept."""
    from dataclasses import dataclass

    from PIL import Image

    import pa.config as cfgmod
    from pa.db import repo
    from pa.db.connection import init_db
    from pa.ingest import thumbs as thumbmod
    from pa.jobs import worker

    cfg = cfgmod._cfg
    conn = init_db(cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    _add_face(conn, 50, 1, (100, 100, 200, 200), confirmed=1)   # you named this one
    _add_face(conn, 51, 1, (600, 100, 200, 200), rejected=1)    # and dismissed this one
    path = thumbmod.thumb_path(cfg.paths.thumbs_dir, "hash1", "view",
                               cfg.thumbs.format.lower())
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1000, 1000)).save(path, cfg.thumbs.format)
    repo.enqueue(conn, 1, stages=("faces",), reset=True)
    conn.commit()

    @dataclass
    class Detected:
        bbox: tuple
        det_score: float
        embedding: np.ndarray
        src_size: tuple

    class Analyzer:
        model_version = "test"

        def detect(self, _blob):
            zero = np.zeros(512, dtype=np.float32)
            return [
                Detected((100, 100, 200, 200), 0.9, zero, (1000, 1000)),  # the named one
                Detected((600, 100, 200, 200), 0.9, zero, (1000, 1000)),  # the dismissed one
                Detected((100, 700, 150, 150), 0.9, zero, (1000, 1000)),  # someone new
            ]

    res = worker.run_faces(conn, cfg, Analyzer())
    assert res.done == 1, res.errors

    rows = conn.execute(
        "SELECT id, confirmed, rejected FROM face WHERE photo_id=1 ORDER BY id").fetchall()
    assert [r["id"] for r in rows][:2] == [50, 51], "a decision must survive re-detection"
    assert len(rows) == 3, "the two it kept were detected again and added a second time"


def test_a_copy_that_was_later_named_is_still_a_copy(client, tmp_path):
    """Naming a proposed group after a re-run marks the duplicate confirmed too,
    leaving two identical named boxes. Both say the same thing about the same
    face, so keeping one loses nothing."""
    import pa.config as cfgmod
    from pa.db.connection import init_db
    conn = init_db(cfgmod._cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    conn.execute("INSERT INTO person (id, name, created_at) VALUES (1,'Ada',0)")
    _add_face(conn, 60, 1, (100, 100, 200, 200), confirmed=1, person=1)
    _add_face(conn, 61, 1, (100, 100, 200, 200), confirmed=1, person=1)

    assert client.post("/api/faces/dedupe").json()["removed"] == 1
    assert [r[0] for r in conn.execute("SELECT id FROM face")] == [60], "the older wins"


def test_two_people_in_one_box_is_left_for_a_human(client, tmp_path):
    """Same box, two different names, is a contradiction rather than a copy.
    Guessing which is right would quietly delete somebody's correction."""
    import pa.config as cfgmod
    from pa.db.connection import init_db
    conn = init_db(cfgmod._cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    conn.execute("INSERT INTO person (id, name, created_at) VALUES (1,'Ada',0)")
    conn.execute("INSERT INTO person (id, name, created_at) VALUES (2,'Grace',0)")
    _add_face(conn, 70, 1, (100, 100, 200, 200), confirmed=1, person=1)
    _add_face(conn, 71, 1, (100, 100, 200, 200), confirmed=1, person=2)

    assert client.post("/api/faces/dedupe").json()["removed"] == 0
    assert len(conn.execute("SELECT id FROM face").fetchall()) == 2


def test_an_ignored_face_and_a_named_one_are_left_alone(client, tmp_path):
    import pa.config as cfgmod
    from pa.db.connection import init_db
    conn = init_db(cfgmod._cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    _add_face(conn, 80, 1, (100, 100, 200, 200), confirmed=1)
    _add_face(conn, 81, 1, (100, 100, 200, 200), rejected=1)

    assert client.post("/api/faces/dedupe").json()["removed"] == 0


def test_two_undecided_copies_keep_one(client, tmp_path):
    """A photo with nobody named on it duplicates just the same, and the copies
    double every stranger in the naming queue."""
    import pa.config as cfgmod
    from pa.db.connection import init_db
    conn = init_db(cfgmod._cfg.paths.db_path)
    conn.execute("DELETE FROM face")
    _add_face(conn, 90, 1, (100, 100, 200, 200))
    _add_face(conn, 91, 1, (100, 100, 200, 200))
    _add_face(conn, 92, 1, (500, 500, 200, 200))

    assert client.post("/api/faces/dedupe").json()["removed"] == 1
    assert {r[0] for r in conn.execute("SELECT id FROM face")} == {90, 92}
