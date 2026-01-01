#!/usr/bin/env python
"""OpenAI-compatible API client utilities for text and image generation."""

import base64
import io
import logging
from dataclasses import dataclass

from PIL import Image

import openai

__all__ = [
    "OpenAIConfig",
    "generate_text",
    "generate_image",
    "generate_image_via_chat",
    "encode_image_to_base64",
    "encode_image_for_api",
    "get_mime_type",
    "convert_image_to_png",
    "DEFAULT_BASE_URL",
]

DEFAULT_BASE_URL = "https://api.openai.com/v1"




@dataclass
class OpenAIConfig:
    """Configuration for creating OpenAI-compatible client."""
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str | None = None  # Default model name

    def create_client(self):
        """Create an OpenAI client based on the configuration.
        
        Returns:
            openai.OpenAI client instance
        """

        
        if not self.api_key:
            raise ValueError("API key is required for OpenAI-compatible mode")
        
        try:
            return openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenAI client: {e}") from e


def generate_text(
    client,
    model: str,
    prompt: str,
) -> str:
    """Generate text content using the specified model.

    Args:
        client: OpenAI client instance
        model: Model name to use for generation
        prompt: The prompt text to send

    Returns:
        Generated text content

    Raises:
        RuntimeError: If API call fails
    """

    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        logging.error(f"文本生成 API 调用失败 (OpenAI): {e}")
        raise RuntimeError(f"文本生成失败: {e}") from e

    if response is None:
        raise RuntimeError("文本生成 API 返回空响应")

    if not response.choices:
        raise RuntimeError("文本生成 API 返回空选择列表")

    choice = response.choices[0]
    
    # Check finish reason
    if choice.finish_reason and choice.finish_reason not in ('stop', 'length'):
        logging.warning(f"文本生成可能被截断或过滤: finish_reason={choice.finish_reason}")

    text = choice.message.content if choice.message and choice.message.content else ""
    return text


def generate_image(
    client,
    model: str,
    prompt: str,
    reference_images: list | None = None,
    size: str = "1024x1024",
) -> tuple[Image.Image | None, str | None]:
    """Generate an image using the specified model.

    Args:
        client: OpenAI client instance
        model: Model name to use for generation (e.g., "dall-e-3", "gpt-image-1")
        prompt: The prompt text describing the image
        reference_images: Optional list of paths to reference images (Path objects)
        size: Image size (default: "1024x1024", options: "1024x1024", "1792x1024", "1024x1792")

    Returns:
        Tuple of (generated PIL Image or None, revised prompt or None)

    Raises:
        RuntimeError: If API call fails
    
    Note:
        - For models like "dall-e-3": Uses images.generate API (no reference image support)
        - For models like "gpt-image-1": Uses images.edit API with reference images
        - Reference images are encoded as base64 and sent to the API
    """

    
    # If reference images provided, use edit API for compatible models
    if reference_images:
        return _generate_image_with_reference(client, model, prompt, reference_images, size)
    
    # Standard generation without reference images
    return _generate_image_standard(client, model, prompt, size)


def encode_image_to_base64(image_path) -> str | None:
    """Encode an image file to base64 string."""
    from pathlib import Path

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
    import os
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
    }.get(ext, "image/jpeg")


def convert_image_to_png(image_path: str, ext: str) -> tuple[str | None, str]:
    """Convert BMP or SVG image to PNG format and return base64 encoding.

    Args:
        image_path: Path to the image file
        ext: File extension (.bmp or .svg)

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

        elif ext == ".svg":
            # Use cairosvg to convert SVG -> PNG
            try:
                import cairosvg
            except ImportError:
                logging.error("需要安装 cairosvg 来处理 SVG 文件: pip install cairosvg")
                return None, ""

            png_data = cairosvg.svg2png(url=image_path)
            return base64.b64encode(png_data).decode("utf-8"), "image/png"

        return None, ""

    except Exception as e:
        logging.error(f"转换图片格式失败 '{image_path}': {e}")
        return None, ""


def encode_image_for_api(image_path: str) -> tuple[str | None, str]:
    """Encode an image file to base64 with proper MIME type.

    Handles BMP and SVG by converting to PNG first.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (base64 encoded string, MIME type) or (None, "") on failure
    """
    import os
    ext = os.path.splitext(image_path)[1].lower()

    # BMP and SVG need conversion to PNG
    if ext in (".bmp", ".svg"):
        return convert_image_to_png(image_path, ext)

    # Standard image formats
    image_base64 = encode_image_to_base64(image_path)
    if image_base64 is None:
        return None, ""

    mime_type = get_mime_type(image_path)
    return image_base64, mime_type


def generate_image_via_chat(
    client,
    model: str,
    prompt: str,
    reference_images: list[str] | None = None,
) -> tuple[bytes | None, str | None]:
    """Generate an image using chat completions API.

    This is used for models that return images embedded in chat responses
    (e.g., Gemini image generation models via OpenAI-compatible API).

    Args:
        client: OpenAI client instance
        model: Model name to use for generation
        prompt: The prompt text describing the image
        reference_images: Optional list of paths to reference images

    Returns:
        Tuple of (image bytes or None, text response or None)

    Raises:
        RuntimeError: If API call fails or no image found in response
    """
    # Build message content
    message_content = [{"type": "text", "text": prompt}]

    # Add reference images if provided
    if reference_images:
        for image_path in reference_images:
            image_base64, mime_type = encode_image_for_api(image_path)
            if image_base64 is None:
                logging.warning(f"无法读取参考图片: {image_path}")
                continue

            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            })

    # Call API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message_content}],
        )
    except Exception as e:
        logging.error(f"图像生成 API 调用失败: {e}")
        raise RuntimeError(f"图像生成失败: {e}") from e

    if response is None:
        raise RuntimeError("图像生成 API 返回空响应")

    if not response.choices:
        raise RuntimeError("图像生成 API 返回空选择列表")

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("图像生成 API 返回的消息内容为空")

    # Parse image from markdown format: ![image](data:image/xxx;base64,...)
    image_bytes = None
    for fmt in ["jpeg", "png", "gif", "webp"]:
        marker = f"![image](data:image/{fmt};base64,"
        if marker in content:
            try:
                image_data = content.split(marker)[1].rstrip(")")
                image_bytes = base64.b64decode(image_data)
                break
            except (IndexError, Exception) as e:
                logging.warning(f"解析 {fmt} 格式图片失败: {e}")
                continue

    if image_bytes is None:
        # Return text response if no image found
        return None, content

    return image_bytes, None


def _generate_image_standard(
    client,
    model: str,
    prompt: str,
    size: str,
) -> tuple[Image.Image | None, str | None]:
    """Generate image using standard images.generate API."""
    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            response_format="b64_json",
            n=1,
        )
    except Exception as e:
        logging.error(f"图像生成 API 调用失败 (OpenAI): {e}")
        raise RuntimeError(f"图像生成失败: {e}") from e

    return _parse_image_response(response)


def _generate_image_with_reference(
    client,
    model: str,
    prompt: str,
    reference_images: list,
    size: str,
) -> tuple[Image.Image | None, str | None]:
    """Generate image with reference images using images.edit API or chat completions."""
    from pathlib import Path
    
    # Prepare the first reference image for edit API
    # OpenAI's edit API accepts one image at a time
    ref_image_path = None
    for img_path in reference_images:
        path = Path(img_path) if not isinstance(img_path, Path) else img_path
        if path.exists():
            ref_image_path = path
            break
    
    if ref_image_path is None:
        logging.warning("没有找到有效的参考图片，回退到标准生成模式")
        return _generate_image_standard(client, model, prompt, size)
    
    # Try using images.edit API first (works with dall-e-2 and some compatible APIs)
    try:
        with open(ref_image_path, "rb") as image_file:
            response = client.images.edit(
                model=model,
                image=image_file,
                prompt=prompt,
                size=size,
                response_format="b64_json",
                n=1,
            )
        return _parse_image_response(response)
    except Exception as e:
        # If edit API fails (e.g., model doesn't support it), try chat completions with vision
        logging.debug(f"images.edit API 不可用，尝试使用 chat completions: {e}")
    
    # Fallback: Use chat completions with vision capability
    # This works with models like gpt-4o, gpt-4-vision, etc.
    try:
        # Build content with images
        content = []
        
        # Add reference images
        for img_path in reference_images:
            b64_data = encode_image_to_base64(img_path)
            if b64_data:
                # Detect image type
                path = Path(img_path) if not isinstance(img_path, Path) else img_path
                suffix = path.suffix.lower()
                media_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }.get(suffix, "image/png")
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64_data}"
                    }
                })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        
        if response and response.choices:
            result_text = response.choices[0].message.content
            # Chat completions returns text, not image
            # Return None for image, text for the response
            return None, result_text
        
        return None, None
        
    except Exception as e:
        logging.error(f"图像生成 API 调用失败 (OpenAI with reference): {e}")
        raise RuntimeError(f"图像生成失败: {e}") from e


def _parse_image_response(response) -> tuple[Image.Image | None, str | None]:
    """Parse image generation API response."""
    if response is None:
        raise RuntimeError("图像生成 API 返回空响应")

    if not response.data:
        raise RuntimeError("图像生成 API 返回空数据")

    image_data = response.data[0]
    
    result_image = None
    result_text = None
    
    # Decode base64 image
    if hasattr(image_data, 'b64_json') and image_data.b64_json:
        try:
            image_bytes = base64.b64decode(image_data.b64_json)
            result_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logging.error(f"解码图像失败: {e}")
            raise RuntimeError(f"解码图像失败: {e}") from e
    
    # Get revised prompt if available
    if hasattr(image_data, 'revised_prompt') and image_data.revised_prompt:
        result_text = image_data.revised_prompt
    
    return result_image, result_text
