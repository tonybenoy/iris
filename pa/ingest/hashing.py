"""Content hashing: exact identity (BLAKE3) and near-duplicate detection (pHash)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from blake3 import blake3
from PIL import Image

CHUNK = 1 << 20


def content_hash(path: Path) -> str:
    """BLAKE3 of the whole file. This is the photo's identity: it survives
    renames, moves and copies to other drives, which is what lets one photo own
    many file rows and be annotated exactly once."""
    h = blake3(max_threads=blake3.AUTO)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0] *= np.sqrt(0.5)
    return m * np.sqrt(2.0 / n)


_DCT32 = _dct_matrix(32)


def perceptual_hash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash. Two photos within a few bits of each other are
    the same shot -- a resize, a re-compress, a lightly edited export."""
    small = img.convert("L").resize((32, 32), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    freq = _DCT32 @ pixels @ _DCT32.T
    block = freq[:8, :8].flatten()
    # Drop the DC term before taking the median: it carries overall brightness,
    # which would otherwise dominate and wash out the structural signal.
    median = np.median(block[1:])
    bits = block > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    # SQLite INTEGER is signed 64-bit; fold the top bit so it round-trips.
    return value - (1 << 64) if value >= (1 << 63) else value


def hamming(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFFFFFFFFFF) ^ (b & 0xFFFFFFFFFFFFFFFF)).bit_count()
