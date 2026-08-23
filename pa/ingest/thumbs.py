"""Thumbnail cache, keyed by content hash.

This cache is what makes the library browsable with the external drives
unplugged: search results, the grid and the lightbox all read from here, and
only the full-resolution original needs the source drive attached.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

from pa.db.repo import RAW_EXTS

SIZES = {"grid": "grid_px", "view": "view_px"}


def thumb_path(root: Path, blake3_hex: str, kind: str, ext: str = "webp") -> Path:
    # Two levels of 2-hex-char sharding: 65k leaf dirs, so even a 500k-photo
    # library keeps a few files per directory instead of one enormous folder
    # that makes every filesystem operation crawl.
    return root / blake3_hex[:2] / blake3_hex[2:4] / f"{blake3_hex}_{kind}.{ext}"


def _load_raw(path: Path) -> Image.Image:
    """Decode a camera RAW file.

    Pillow cannot open any RAW format, so without this the scanner would happily
    index a folder of .CR2 or .NEF and then fail every one of them at the
    thumbnail stage. Most RAWs embed a full-size JPEG preview; using it is an
    order of magnitude faster than demosaicing and is what every other photo
    manager shows you anyway.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            return Image.open(io.BytesIO(thumb.data))
        if thumb is not None and thumb.format == rawpy.ThumbFormat.BITMAP:
            return Image.fromarray(thumb.data)
        # No usable preview: demosaic properly. Slow, but correct.
        return Image.fromarray(raw.postprocess(use_camera_wb=True, no_auto_bright=False))


def open_image(path: Path, max_px: int | None = None) -> Image.Image:
    """Open any supported photo as a PIL image, RAW included."""
    if path.suffix.lower() in RAW_EXTS:
        img = _load_raw(path)
    else:
        img = Image.open(path)
        if max_px:
            # draft() lets the JPEG decoder skip DCT coefficients and decode at
            # 1/2, 1/4 or 1/8 scale directly -- several times faster than
            # decoding full size and resizing, which matters over a USB drive.
            img.draft("RGB", (max_px * 2, max_px * 2))
    return img


def load_oriented(path: Path, max_px: int | None = None) -> Image.Image:
    """Open an image, apply EXIF rotation, optionally downscaling during decode."""
    img = open_image(path, max_px)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def generate(path: Path, blake3_hex: str, cache_root: Path, cfg) -> dict[str, Path]:
    """Write grid and view thumbnails; returns {kind: path}. Idempotent."""
    targets = {kind: thumb_path(cache_root, blake3_hex, kind, cfg.format.lower())
               for kind in SIZES}
    if all(p.exists() for p in targets.values()):
        return targets

    largest = max(getattr(cfg, attr) for attr in SIZES.values())
    with load_oriented(path, max_px=largest) as img:
        for kind, attr in SIZES.items():
            out = targets[kind]
            if out.exists():
                continue
            side = getattr(cfg, attr)
            thumb = img.copy()
            thumb.thumbnail((side, side), Image.LANCZOS)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".tmp")
            thumb.save(tmp, cfg.format, quality=cfg.quality, method=4)
            tmp.replace(out)  # atomic: a killed indexer never leaves a torn thumb
    return targets
