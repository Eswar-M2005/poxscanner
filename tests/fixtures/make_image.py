"""Write a tiny RGB PNG fixture without requiring a real dataset."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

DEFAULT_SIZE = (32, 32)
DEFAULT_COLOR = (180, 90, 70)


def tiny_rgb_png_bytes(
    width: int = DEFAULT_SIZE[0],
    height: int = DEFAULT_SIZE[1],
    color: tuple[int, int, int] = DEFAULT_COLOR,
) -> bytes:
    """Return a valid 8-bit RGB PNG as bytes."""
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    r, g, b = (int(c) & 0xFF for c in color)
    raw = bytearray()
    row = bytes((r, g, b)) * width
    for _ in range(height):
        raw.append(0)
        raw.extend(row)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        + _chunk(b"IEND", b"")
    )


def write_tiny_rgb_png(
    path: Path,
    width: int = DEFAULT_SIZE[0],
    height: int = DEFAULT_SIZE[1],
    color: tuple[int, int, int] = DEFAULT_COLOR,
) -> Path:
    """Write a tiny RGB PNG to ``path`` and return the path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(tiny_rgb_png_bytes(width=width, height=height, color=color))
    return dest
