"""EXIF/metadata extraction, normalised into the columns `photo` actually stores."""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

try:  # HEIC/HEIF is what modern phones shoot; without this they cannot be opened
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF = True
except ImportError:  # pragma: no cover
    HEIF = False

_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_GPSTAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}

# Phone cameras and messaging apps bake the timestamp into the filename far more
# reliably than they write EXIF -- WhatsApp strips EXIF entirely, for instance.
_FILENAME_DATE = [
    # Date and time, with separators optional and independent in each half:
    # IMG20260401195853, IMG_20260412_134132, "Screenshot 2026-08-21 132906",
    # "2026-08-08 20.43.42", 2026_08_08-204342 all parse.
    re.compile(r"(?P<y>20\d{2})[-_.]?(?P<mo>\d{2})[-_.]?(?P<d>\d{2})"
               r"[ _T-]+(?P<h>\d{2})[-:.]?(?P<mi>\d{2})[-:.]?(?P<s>\d{2})"),
    re.compile(r"(?P<y>20\d{2})(?P<mo>\d{2})(?P<d>\d{2})"
               r"(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})"),
    # Date only.
    re.compile(r"(?P<y>20\d{2})[-_.]?(?P<mo>\d{2})[-_.]?(?P<d>\d{2})"),
]


@dataclass
class PhotoMeta:
    width: int | None = None
    height: int | None = None
    mime: str | None = None
    taken_at: int | None = None
    taken_at_source: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    lens: str | None = None
    iso: int | None = None
    f_number: float | None = None
    exposure_s: float | None = None
    focal_len: float | None = None
    orientation: int | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    keywords: list[str] = field(default_factory=list)


def _rational(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _gps_degrees(coord, ref: str | None) -> float | None:
    try:
        d, m, s = (float(x) for x in coord)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    deg = d + m / 60 + s / 3600
    if ref and ref.upper() in ("S", "W"):
        deg = -deg
    return round(deg, 7)


def _parse_exif_datetime(raw: str) -> datetime | None:
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw.strip()[:19], fmt)
        except (ValueError, AttributeError):
            continue
    return None


def date_from_filename(name: str) -> datetime | None:
    for pattern in _FILENAME_DATE:
        m = pattern.search(name)
        if not m:
            continue
        g = m.groupdict()
        try:
            dt = datetime(
                int(g["y"]), int(g["mo"]), int(g["d"]),
                int(g.get("h") or 0), int(g.get("mi") or 0), int(g.get("s") or 0),
            )
        except ValueError:
            continue
        if 2000 <= dt.year <= datetime.now().year + 1:
            return dt
    return None


def extract(path: Path, img: Image.Image | None = None) -> PhotoMeta:
    meta = PhotoMeta()
    close = False
    if img is None:
        img = Image.open(path)
        close = True
    try:
        meta.width, meta.height = img.size
        meta.mime = Image.MIME.get(img.format or "", None)
        exif = img.getexif()
        if exif:
            _fill_from_exif(meta, exif)
    except Exception:
        pass
    finally:
        if close:
            img.close()

    if meta.taken_at is None:
        dt = date_from_filename(path.name)
        if dt:
            meta.taken_at, meta.taken_at_source = int(dt.timestamp()), "filename"
    if meta.taken_at is None:
        try:
            meta.taken_at = int(path.stat().st_mtime)
            meta.taken_at_source = "mtime"
        except OSError:
            pass
    # EXIF orientation 5-8 mean the image is stored rotated 90 degrees; the
    # stored width/height are swapped relative to how it should be displayed.
    if meta.orientation and meta.orientation >= 5 and meta.width and meta.height:
        meta.width, meta.height = meta.height, meta.width
    return meta


def _fill_from_exif(meta: PhotoMeta, exif) -> None:
    def tag(name):
        return exif.get(_TAGS.get(name, -1))

    meta.camera_make = (str(tag("Make")).strip() or None) if tag("Make") else None
    meta.camera_model = (str(tag("Model")).strip() or None) if tag("Model") else None
    meta.orientation = tag("Orientation")

    ifd = {}
    with contextlib.suppress(AttributeError, KeyError, ValueError):
        ifd = exif.get_ifd(ExifTags.IFD.Exif) or {}

    def sub(name):
        return ifd.get(_TAGS.get(name, -1))

    meta.lens = (str(sub("LensModel")).strip() or None) if sub("LensModel") else None
    meta.iso = sub("ISOSpeedRatings") if isinstance(sub("ISOSpeedRatings"), int) else None
    meta.f_number = _rational(sub("FNumber"))
    meta.exposure_s = _rational(sub("ExposureTime"))
    meta.focal_len = _rational(sub("FocalLength"))

    for key in ("DateTimeOriginal", "DateTimeDigitized"):
        raw = sub(key)
        if raw:
            dt = _parse_exif_datetime(str(raw))
            if dt:
                meta.taken_at, meta.taken_at_source = int(dt.timestamp()), "exif"
                break
    if meta.taken_at is None and tag("DateTime"):
        dt = _parse_exif_datetime(str(tag("DateTime")))
        if dt:
            meta.taken_at, meta.taken_at_source = int(dt.timestamp()), "exif"

    try:
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo) or {}
    except (AttributeError, KeyError, ValueError):
        gps = {}
    if gps:
        lat = _gps_degrees(gps.get(_GPSTAGS["GPSLatitude"]), gps.get(_GPSTAGS["GPSLatitudeRef"]))
        lon = _gps_degrees(gps.get(_GPSTAGS["GPSLongitude"]), gps.get(_GPSTAGS["GPSLongitudeRef"]))
        # 0,0 is in the Atlantic; it is always a zeroed sensor, never a real photo.
        if lat is not None and lon is not None and (abs(lat) > 1e-6 or abs(lon) > 1e-6):
            meta.gps_lat, meta.gps_lon = lat, lon
