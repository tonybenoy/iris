"""Drives must be identified by what they ARE, not where they happen to appear.

External drives get a different letter every time Windows feels like it, and
Linux automounts move between /media and /run/media. Keying a library on the
mount path would turn a replugged disk into thousands of "deleted" photos and
re-annotate the lot when it came back.
"""
from pathlib import Path

import pytest

from pa.db import repo
from pa.db.connection import init_db
from pa.ingest import volumes
from pa.ingest.volumes import Mount

SERIAL_A = "40E8-3749"   # the photo drive
SERIAL_B = "1111-2222"   # some other disk


def fake_world(monkeypatch, letter_to_serial: dict[str, str]):
    """Present a mount table where each drive letter maps to a given serial."""
    mounts = [
        Mount(Path(f"/mnt/{letter.lower()}"), f"{letter}:\\134", "9p",
              f"rw,noatime,aname=drvfs;path={letter}:" + chr(92) + ";uid=1000")
        for letter in letter_to_serial
    ]
    monkeypatch.setattr(volumes, "_mounts", lambda: mounts)

    def fake_vol(cmd, **kw):
        letter = cmd[-1][0].upper()
        serial = letter_to_serial.get(letter)

        class R:
            stdout = (f" Volume in drive {letter} is PhotoDisk\n"
                      f" Volume Serial Number is {serial}\n") if serial else ""
        return R()

    monkeypatch.setattr(volumes.subprocess, "run", fake_vol)
    return mounts


def test_same_disk_found_after_the_letter_changes(monkeypatch):
    fake_world(monkeypatch, {"D": SERIAL_A})
    uuid, label, mount = volumes.identify(Path("/mnt/d"))
    assert uuid == f"win-{SERIAL_A}"
    assert mount.mountpoint == Path("/mnt/d")

    # Unplug, replug: Windows now calls the very same disk G:.
    fake_world(monkeypatch, {"G": SERIAL_A})
    assert volumes.current_mountpoint(uuid) == Path("/mnt/g"), \
        "the drive moved letters and must still be found by its serial"


def test_a_different_disk_at_the_old_letter_is_not_mistaken_for_it(monkeypatch):
    """The dangerous case. If some other disk takes over D:, treating it as the
    original would index a stranger's files into that library and mark every
    real photo missing."""
    fake_world(monkeypatch, {"D": SERIAL_A})
    uuid, _, _ = volumes.identify(Path("/mnt/d"))

    fake_world(monkeypatch, {"D": SERIAL_B})
    assert volumes.current_mountpoint(uuid) is None, \
        "a different disk at the same letter must not resolve"


def test_unplugged_drive_resolves_to_nothing(monkeypatch):
    fake_world(monkeypatch, {"D": SERIAL_A})
    uuid, _, _ = volumes.identify(Path("/mnt/d"))
    fake_world(monkeypatch, {"C": "50E4-2151"})
    assert volumes.current_mountpoint(uuid) is None


def test_photo_paths_survive_a_letter_change(tmp_path, monkeypatch):
    """End to end: a file recorded while the disk was D: must be readable once
    the same disk comes back as G:."""
    fake_world(monkeypatch, {"D": SERIAL_A})
    conn = init_db(tmp_path / "t.db")
    uuid, label, mount = volumes.identify(Path("/mnt/d"))
    vol_id = repo.upsert_volume(conn, uuid, label, str(mount.mountpoint))
    root_id = repo.add_root(conn, vol_id, "Photos/2024", "Trip", [])
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h',0)")
    repo.link_file(conn, 1, root_id, "IMG_1.jpg", "IMG_1.jpg", 0, 0)
    conn.commit()

    # The drive returns as G:, and the file is where G: now says it is.
    fake_world(monkeypatch, {"G": SERIAL_A})
    target = Path("/mnt/g/Photos/2024/IMG_1.jpg")
    monkeypatch.setattr(Path, "exists", lambda self: self == target)

    from pa.ingest.scanner import resolve_file_path
    assert resolve_file_path(conn, 1) == target


def test_relative_paths_only(tmp_path, monkeypatch):
    """Nothing stored may contain a mount point or drive letter."""
    fake_world(monkeypatch, {"D": SERIAL_A})
    mount = volumes.find_mount(Path("/mnt/d"))
    assert volumes.relative_to_mount(Path("/mnt/d/Photos/2024"), mount) == "Photos/2024"
    assert volumes.relative_to_mount(Path("/mnt/d"), mount) == ""


@pytest.mark.parametrize("mountpoint", ["/mnt/My Drive", "/media/tony/Backup 2024"])
def test_mount_points_with_spaces(monkeypatch, mountpoint):
    """/proc/mounts octal-escapes spaces; external drives very often have them."""
    escaped = mountpoint.replace(" ", r"\040")
    monkeypatch.setattr(
        Path, "read_text",
        lambda self, *a, **k: f"/dev/sdb1 {escaped} ext4 rw 0 0\n")
    got = [m.mountpoint for m in volumes._mounts()]
    assert Path(mountpoint) in got


# --------------------------------------------------------- browsing upwards
def test_the_picker_can_always_walk_back_up(tmp_path):
    """The picker had no way out of a folder but the breadcrumb, which was built
    by splitting the path on "/" -- so on Windows, where the server spells paths
    with backslashes, it produced one crumb holding the whole path and browsing
    only ever went downwards."""
    from pa.api.roots import browse

    deep = tmp_path / "photos" / "2024" / "summer"
    deep.mkdir(parents=True)

    seen = []
    at = str(deep)
    for _ in range(20):
        listing = browse(at)
        seen.append(listing.path)
        if listing.parent is None:
            break
        # Whatever it names as the parent has to be somewhere we can browse to.
        assert browse(listing.parent).path == listing.parent
        at = listing.parent

    assert seen[0] == str(deep)
    assert str(tmp_path / "photos") in seen
    assert browse(seen[-1]).parent is None, "walking up must end at a root, not loop"
