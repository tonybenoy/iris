"""Folder browsing and library management for the web UI.

A browser file picker can only choose files on the machine running the browser,
which is the wrong machine when the photos live on the server's drives. So the
server offers its own directory listing instead.

That listing is deliberately narrow: directories only, no file contents, no
reading anything. It still reveals your folder structure to whoever can reach
the port, which is why the app binds to localhost by default.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pa.db import repo

# Noise that is never a photo library, hidden from the picker to keep it usable.
SKIP_DIRS = {
    "$RECYCLE.BIN", "System Volume Information", "node_modules", "__pycache__",
    ".git", "lost+found", "AppData", "Windows", "Program Files",
    "Program Files (x86)", "ProgramData", "$WinREAgent", "Recovery",
}


@dataclass
class Entry:
    name: str
    path: str
    photos: int = 0          # images directly inside, a hint that this is the folder
    children: int = 0
    readable: bool = True
    library: str | None = None   # see library_state()


@dataclass
class Listing:
    path: str
    parent: str | None
    entries: list[Entry] = field(default_factory=list)
    photos_here: int = 0
    shortcuts: list[dict] = field(default_factory=list)
    library: str | None = None


def _comparable(path: str | Path) -> str:
    """A path in the one spelling comparisons can use. normcase is what makes
    C:\\Users and c:\\users the same folder on Windows and two different ones
    everywhere else."""
    return os.path.normcase(os.path.normpath(str(path)))


def library_state(path: str | Path, roots: list[str]) -> str | None:
    """How this folder relates to what has already been added.

    Adding the same photos twice is easy to do by accident and tedious to undo,
    and the picker gave no sign of it: every folder looked equally new. The
    three answers worth having are different warnings, not one.

    "root"     -- this exact folder was added.
    "inside"   -- something above it was, so its photos are already indexed.
    "contains" -- something below it was, so adding this overlaps that.
    """
    here = _comparable(path)
    keyed = [_comparable(r) for r in roots]
    if here in keyed:
        return "root"
    if any(here.startswith(root + os.sep) for root in keyed):
        return "inside"
    if any(root.startswith(here + os.sep) for root in keyed):
        return "contains"
    return None


def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in (repo.IMAGE_EXTS | repo.RAW_EXTS)


def _peek(path: Path, cap: int = 400) -> tuple[int, int]:
    """(images, subdirectories) directly inside `path`.

    Capped: counting every file in a 40k-photo directory just to render one row
    of a picker would make the picker unusable.
    """
    images = subdirs = 0
    try:
        with os.scandir(path) as it:
            for i, e in enumerate(it):
                if i > cap:
                    break
                try:
                    if e.is_dir(follow_symlinks=False):
                        subdirs += 1
                    elif _is_image(e.name):
                        images += 1
                except OSError:
                    continue
    except (PermissionError, OSError):
        return 0, 0
    return images, subdirs


def shortcuts() -> list[dict]:
    """Likely starting points: mounted drives and the usual photo folders."""
    out: list[dict] = []
    seen: set[str] = set()

    def add(label: str, path: Path) -> None:
        p = str(path)
        if p in seen or not path.is_dir():
            return
        seen.add(p)
        out.append({"label": label, "path": p})

    home = Path.home()
    for name in ("Pictures", "Photos", "Downloads"):
        add(name, home / name)
    add("Home", home)

    if sys.platform == "win32":
        import string
        for letter in string.ascii_uppercase:
            add(f"{letter}:", Path(f"{letter}:\\"))
        return out

    # Mounted volumes: /mnt/* covers WSL drive letters, /media and /run/media
    # cover most Linux desktop automounts.
    for base in (Path("/mnt"), Path("/media"), Path("/run/media"), Path("/Volumes")):
        if not base.is_dir():
            continue
        try:
            for child in sorted(base.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    add(f"{child.name.upper()}:" if len(child.name) == 1 else child.name, child)
        except (PermissionError, OSError):
            continue
    return out


def browse(raw: str | None, roots: list[str] | None = None) -> Listing:
    path = Path(raw).expanduser() if raw else Path.home()
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"cannot open {raw}") from exc
    if not path.is_dir():
        raise NotADirectoryError(f"not a folder: {path}")

    entries: list[Entry] = []
    photos_here = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name.startswith(".") or e.name in SKIP_DIRS:
                            continue
                        images, subdirs = _peek(Path(e.path))
                        entries.append(Entry(e.name, e.path, images, subdirs,
                                             library=library_state(e.path, roots or [])))
                    elif _is_image(e.name):
                        photos_here += 1
                except OSError:
                    entries.append(Entry(e.name, e.path, readable=False))
    except PermissionError as exc:
        raise PermissionError(f"no permission to read {path}") from exc

    entries.sort(key=lambda e: e.name.lower())
    return Listing(
        path=str(path),
        parent=str(path.parent) if path.parent != path else None,
        entries=entries,
        photos_here=photos_here,
        shortcuts=shortcuts(),
        library=library_state(path, roots or []),
    )
