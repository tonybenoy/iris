"""XMP sidecars are the escape hatch from this app: what the pipeline worked out
travels with the photo, readable by Lightroom, digiKam and exiftool."""
from pathlib import Path

from pa.db.connection import init_db
from pa.sidecar import xmp


def test_sidecar_path_keeps_original_extension():
    # digiKam and exiftool default to appending, not replacing: this keeps a
    # RAW+JPEG pair from sharing one sidecar.
    assert xmp.sidecar_path(Path("/p/IMG_1.jpg")) == Path("/p/IMG_1.jpg.xmp")
    assert xmp.sidecar_path(Path("/p/IMG_1.CR2")) == Path("/p/IMG_1.CR2.xmp")


def test_roundtrip():
    sc = xmp.Sidecar(caption="A man by a wall", tags=["man", "graffiti"], rating=4,
                     regions=[{"name": "Tony", "x": 0.5, "y": 0.4, "w": 0.2, "h": 0.3}])
    got = xmp.parse(xmp.build(sc))
    assert got.caption == "A man by a wall"
    assert got.tags == ["man", "graffiti"]
    assert got.rating == 4
    assert got.people == ["Tony"]


def test_regions_are_centre_based_and_normalised(tmp_path):
    """MWG areas are centre-based. Emitting top-left coordinates puts every face
    box down and to the right in every other application that reads them."""
    conn = init_db(tmp_path / "t.db")
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h',0)")
    conn.execute("INSERT INTO person (id, name, created_at) VALUES (1,'Tony',0)")
    conn.execute(
        """INSERT INTO face (photo_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                             det_score, embedding, model, src_w, src_h, created_at)
           VALUES (1,1, 100,200, 100,100, 0.9, x'00', 'm', 1000, 1000, 0)""")
    conn.commit()

    sc = xmp.collect(conn, 1)
    r = sc.regions[0]
    assert r["name"] == "Tony"
    assert r["x"] == 0.15  # (100 + 100/2) / 1000
    assert r["y"] == 0.25  # (200 + 100/2) / 1000
    assert r["w"] == 0.1 and r["h"] == 0.1


def test_face_without_source_dims_is_skipped(tmp_path):
    """A box with no known coordinate space cannot be normalised; emitting it
    would place the region arbitrarily."""
    conn = init_db(tmp_path / "t.db")
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h',0)")
    conn.execute("INSERT INTO person (id, name, created_at) VALUES (1,'Tony',0)")
    conn.execute(
        """INSERT INTO face (photo_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                             det_score, embedding, model, created_at)
           VALUES (1,1, 10,10, 50,50, 0.9, x'00', 'm', 0)""")
    conn.commit()
    assert xmp.collect(conn, 1).regions == []


def test_write_skips_photos_with_nothing_to_say(tmp_path):
    conn = init_db(tmp_path / "t.db")
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h',0)")
    conn.commit()
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    assert xmp.write(conn, 1, photo) is None
    assert not xmp.sidecar_path(photo).exists()


def test_import_adds_to_manual_tags_without_dropping_existing(tmp_path):
    from pa.db import repo

    conn = init_db(tmp_path / "t.db")
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h',0)")
    repo.set_tags(conn, 1, ["mine"], source="manual")
    repo.set_tags(conn, 1, ["ai-guess"], source="ai")
    conn.commit()

    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    xmp.sidecar_path(photo).write_bytes(
        xmp.build(xmp.Sidecar(tags=["Holiday", "Kerala"], rating=3)))
    xmp.read_into(conn, 1, photo)
    conn.commit()

    tags = {r["name"]: r["source"] for r in conn.execute(
        "SELECT t.name, pt.source FROM photo_tag pt JOIN tag t ON t.id=pt.tag_id "
        "WHERE pt.photo_id=1")}
    assert tags["mine"] == "manual"
    assert tags["holiday"] == "manual"
    assert tags["kerala"] == "manual"
    assert tags["ai-guess"] == "ai", "importing keywords must not touch model output"
    assert conn.execute("SELECT rating FROM photo WHERE id=1").fetchone()[0] == 3


def test_malformed_xmp_does_not_raise():
    assert xmp.parse(b"<not-xml").caption == ""
