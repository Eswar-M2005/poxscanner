"""Image validation: MIME allowlist, size, Pillow decode, RGB conversion."""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from app.core.errors import APIError

ALLOWED_MIMES = ("image/jpeg", "image/png", "image/webp")
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_IMAGE_DIMENSION = 4096

_MIME_TO_FORMATS = {
    "image/jpeg": {"JPEG"},
    "image/png": {"PNG"},
    "image/webp": {"WEBP"},
}


def normalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";")[0].strip().lower()


def validate_and_decode(
    data: bytes,
    content_type: str | None,
    max_bytes: int,
) -> Image.Image:
    """Decode upload bytes as RGB. Raises APIError on 400/413 failures."""
    if not data:
        raise APIError(400, "empty_file", "Uploaded file is empty.")

    if len(data) > max_bytes:
        raise APIError(
            413,
            "file_too_large",
            f"File exceeds maximum size of {max_bytes} bytes.",
        )

    mime = normalize_content_type(content_type)
    if mime not in ALLOWED_MIMES:
        raise APIError(
            400,
            "unsupported_media_type",
            f"Unsupported media type: {mime or 'unknown'}.",
        )

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            fmt = (img.format or "").upper()
            width, height = img.size
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise APIError(
            400,
            "invalid_image",
            "File could not be decoded as an image.",
        ) from None

    if fmt not in ALLOWED_PIL_FORMATS:
        raise APIError(
            400,
            "unsupported_media_type",
            f"Unsupported image format: {fmt or 'unknown'}.",
        )

    allowed_for_mime = _MIME_TO_FORMATS.get(mime)
    if allowed_for_mime is not None and fmt not in allowed_for_mime:
        raise APIError(
            400,
            "invalid_image",
            "File could not be decoded as an image.",
        )

    if width < 1 or height < 1:
        raise APIError(
            400,
            "invalid_image",
            "File could not be decoded as an image.",
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise APIError(
            400,
            "invalid_image",
            f"Image dimension exceeds {MAX_IMAGE_DIMENSION}px.",
        )

    return rgb
