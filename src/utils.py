#!/usr/bin/env python
"""Image utility functions for encoding, converting, and loading images."""

import base64
import io
import logging
import os
from pathlib import Path

from PIL import Image


__all__ = [
    "load_image",
    "encode_image_to_base64",
    "get_mime_type",
    "convert_image_to_png",
    "encode_image_for_api",
]


def load_image(img_path: Path) -> Image.Image:
    """Load an image from path.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        PIL Image object
        
    Raises:
        OSError: If image cannot be loaded
    """
    return Image.open(img_path)


def encode_image_to_base64(image_path) -> str | None:
    """Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file (str or Path)
        
    Returns:
        Base64 encoded string, or None on failure
    """
    path = Path(image_path) if not isinstance(image_path, Path) else image_path
    if not path.exists():
        logging.warning(f"参考图片不存在: {path}")
        return None

    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except (OSError, IOError) as e:
        logging.warning(f"无法读取参考图片 {path}: {e}")
        return None


def get_mime_type(file_path: str) -> str:
    """Get MIME type based on file extension.

    Args:
        file_path: Path to the image file

    Returns:
        MIME type string (e.g., "image/png", "image/jpeg")
    """
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")


def convert_image_to_png(image_path: str, ext: str) -> tuple[str | None, str]:
    """Convert BMP image to PNG format and return base64 encoding.

    Args:
        image_path: Path to the image file
        ext: File extension (.bmp)

    Returns:
        Tuple of (base64 encoded string, MIME type) or (None, "") on failure
    """
    try:
        if ext == ".bmp":
            # Use PIL to convert BMP -> PNG
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8"), "image/png"

        return None, ""

    except Exception as e:
        logging.error(f"转换图片格式失败 '{image_path}': {e}")
        return None, ""


def encode_image_for_api(image_path: str) -> tuple[str | None, str]:
    """Encode an image file to base64 with proper MIME type.

    Handles BMP by converting to PNG first.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (base64 encoded string, MIME type) or (None, "") on failure
    """
    ext = os.path.splitext(image_path)[1].lower()

    # BMP needs conversion to PNG
    if ext == ".bmp":
        return convert_image_to_png(image_path, ext)

    # Standard image formats
    image_base64 = encode_image_to_base64(image_path)
    if image_base64 is None:
        return None, ""

    mime_type = get_mime_type(image_path)
    return image_base64, mime_type
