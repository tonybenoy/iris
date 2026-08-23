"""Stable identification of the drive a photo lives on.

Roots are stored as (volume_uuid, path-relative-to-mount) rather than an absolute
path. External drives remount at different letters and mount points; keying on
the absolute path would make a replugged HDD look like thousands of deleted
photos and re-annotate everything when it came back.

Identification strategies, most reliable first:
  1. Native Windows        -> NTFS volume serial via GetVolumeInformationW
  2. Windows drives in WSL -> the same serial, via `cmd.exe /c vol`
  3. Linux filesystems     -> filesystem UUID via /dev/disk/by-uuid
  4. Anything else         -> a marker file written at the mount root

The serial is the same number whichever way it is read, so a library built under
WSL still resolves if you later run this natively on Windows, and vice versa.
"""
from __future__ import annotations

import os
import re
import string
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

MARKER = ".photo_anotater_volume"
_DRVFS_PATH = re.compile(r"path=([A-Za-z]):\\")


@dataclass(frozen=True)
class Mount:
    mountpoint: Path
    source: str
    fstype: str
    options: str


WINDOWS = sys.platform == "win32"


def _windows_drive_serial(letter: str) -> tuple[str, str] | None:
    """(serial, label) for a drive letter, read straight from the Win32 API."""
    import ctypes

    buf = ctypes.create_unicode_buffer(261)
    fs = ctypes.create_unicode_buffer(261)
    serial = ctypes.c_ulong(0)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(f"{letter}:\\"), buf, ctypes.sizeof(buf),
        ctypes.byref(serial), None, None, fs, ctypes.sizeof(fs))
    if not ok or not serial.value:
        return None
    # Format it exactly as `vol` prints it, so a library indexed under WSL and
    # one indexed natively agree on the same drive.
    raw = f"{serial.value:08X}"
    return f"{raw[:4]}-{raw[4:]}", (buf.value or f"{letter}:")


def _windows_mounts() -> list[Mount]:
    """Every drive letter that currently exists, as a Mount."""
    out = []
    bits = ctypes_bitmask()
    for i, letter in enumerate(string.ascii_uppercase):
        if not bits >> i & 1:
            continue
        root = f"{letter}:\\"
        try:
            if not os.path.isdir(root):
                continue
        except OSError:
            continue
        out.append(Mount(Path(root), f"{letter}:", "ntfs", f"path={letter}:\\"))
    return out


def ctypes_bitmask() -> int:
    import ctypes

    return int(ctypes.windll.kernel32.GetLogicalDrives())


def _mounts() -> list[Mount]:
    if WINDOWS:
        return _windows_mounts()
    out = []
    for line in Path("/proc/mounts").read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        src, mp, fstype, opts = parts[0], parts[1], parts[2], parts[3]
        # /proc/mounts octal-escapes spaces and backslashes in the mount point.
        mp = mp.encode().decode("unicode_escape")
        out.append(Mount(Path(mp), src, fstype, opts))
    return out


def find_mount(path: Path) -> Mount:
    """The mount whose mountpoint is the longest prefix of `path`."""
    path = path.resolve()
    best: Mount | None = None
    for m in _mounts():
        if (path == m.mountpoint or m.mountpoint in path.parents) and (
                best is None or len(m.mountpoint.parts) > len(best.mountpoint.parts)):
            best = m
    if best is None:
        raise RuntimeError(f"no mount point found for {path}")
    return best


def _windows_volume(mount: Mount) -> tuple[str, str] | None:
    """(uuid, label) for a Windows drive, native or exposed through WSL."""
    if WINDOWS:
        letter = str(mount.mountpoint)[0].upper()
        got = _windows_drive_serial(letter)
        return (f"win-{got[0]}", got[1]) if got else None
    if mount.fstype not in ("9p", "drvfs", "drv_fs"):
        return None
    m = _DRVFS_PATH.search(mount.options)
    if not m:
        return None
    letter = m.group(1).upper()
    try:
        out = subprocess.run(["cmd.exe", "/c", "vol", f"{letter}:"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    serial = re.search(r"Serial Number is ([0-9A-Fa-f]{4}-[0-9A-Fa-f]{4})", out)
    if not serial:
        return None
    label = re.search(r"Volume in drive \w is (.+)", out)
    name = label.group(1).strip() if label else f"{letter}:"
    return f"win-{serial.group(1).upper()}", name


def _linux_uuid(mount: Mount) -> tuple[str, str] | None:
    by_uuid = Path("/dev/disk/by-uuid")
    if not mount.source.startswith("/dev/") or not by_uuid.is_dir():
        return None
    target = Path(mount.source).resolve()
    for link in by_uuid.iterdir():
        try:
            if link.resolve() == target:
                return f"uuid-{link.name}", mount.mountpoint.name or "root"
        except OSError:
            continue
    return None


def _marker(mount: Mount) -> tuple[str, str]:
    """Last resort: a tiny id file at the mount root, created if absent."""
    path = mount.mountpoint / MARKER
    try:
        if path.exists():
            return f"mark-{path.read_text().strip()}", mount.mountpoint.name
        vol_id = str(uuid.uuid4())
        path.write_text(vol_id)
        return f"mark-{vol_id}", mount.mountpoint.name
    except OSError:
        # Read-only media: fall back to the mount point itself. Less stable,
        # but better than refusing to index the drive at all.
        return f"path-{mount.mountpoint}", mount.mountpoint.name


def identify(path: Path) -> tuple[str, str, Mount]:
    """Return (volume_uuid, label, mount) for the volume containing `path`."""
    mount = find_mount(path)
    for probe in (_windows_volume, _linux_uuid):
        got = probe(mount)
        if got:
            return got[0], got[1], mount
    vol_id, label = _marker(mount)
    return vol_id, label, mount


def relative_to_mount(path: Path, mount: Mount) -> str:
    rel = path.resolve().relative_to(mount.mountpoint)
    return str(rel) if str(rel) != "." else ""


def current_mountpoint(volume_uuid: str) -> Path | None:
    """Where this volume is mounted right now, or None if it is not attached."""
    for m in _mounts():
        try:
            for probe in (_windows_volume, _linux_uuid):
                got = probe(m)
                if got and got[0] == volume_uuid:
                    return m.mountpoint
            if volume_uuid.startswith("mark-"):
                marker = m.mountpoint / MARKER
                if marker.exists() and f"mark-{marker.read_text().strip()}" == volume_uuid:
                    return m.mountpoint
            elif volume_uuid.startswith("path-") and str(m.mountpoint) == volume_uuid[5:]:
                return m.mountpoint
        except OSError:
            continue
    return None
