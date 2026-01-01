#!/usr/bin/env python
"""Google GenAI (Gemini) client utilities for text and image generation."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

__all__ = [
    "GeminiConfig",
    "generate_text",
    "generate_image",
    "generate_image_via_chat",
    "DEFAULT_BASE_URL",
    "DEFAULT_LOCATION",
]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_LOCATION = "us-central1"

# Protobuf enum value for STOP finish reason
# Used when API returns integer instead of string
FINISH_REASON_STOP = 1


@dataclass
class GeminiConfig:
    """Configuration for creating Google GenAI (Gemini) client."""
    # API Key mode
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    # Vertex AI mode
    vertex: bool = False
    project: str | None = None
    location: str = DEFAULT_LOCATION
    credentials: str | None = None

    def create_client(self) -> genai.Client:
        """Create a GenAI client based on the configuration."""
        try:
            if self.vertex:
                # Set credentials environment variable if provided
                if self.credentials:
                    if not Path(self.credentials).exists():
                        raise FileNotFoundError(f"Credentials file not found: {self.credentials}")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials
                # Support custom base_url for Vertex AI (e.g., private endpoints)
                http_opts = None
                if self.base_url != DEFAULT_BASE_URL:
                    http_opts = types.HttpOptions(base_url=self.base_url)
                return genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                    http_options=http_opts,
                )
            else:
                if not self.api_key:
                    raise ValueError("API key is required for non-Vertex AI mode")
                return genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(base_url=self.base_url),
                )
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to create GenAI client: {e}") from e


def generate_text(
    client: genai.Client,
    model: str,
    prompt: str,
) -> str:
    """Generate text content using the specified model.

    Args:
        client: GenAI client instance
        model: Model name to use for generation
        prompt: The prompt text to send

    Returns:
        Generated text content

    Raises:
        RuntimeError: If API call fails or content is blocked
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
        )
    except Exception as e:
        logging.error(f"文本生成 API 调用失败: {e}")
        raise RuntimeError(f"文本生成失败: {e}") from e

    if response is None:
        raise RuntimeError("文本生成 API 返回空响应")

    # Check for blocked content or safety filters
    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
            raise RuntimeError(f"提示词被阻止: {response.prompt_feedback.block_reason}")

    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                # finish_reason can be string ('STOP', 'MAX_TOKENS') or int (FINISH_REASON_STOP in protobuf enum)
                if candidate.finish_reason not in ('STOP', 'MAX_TOKENS', FINISH_REASON_STOP):
                    logging.warning(f"文本生成可能被截断或过滤: finish_reason={candidate.finish_reason}")

    text = response.text if response.text else ""
    return text


def generate_image(
    client: genai.Client,
    model: str,
    prompt: str,
    reference_images: list[Path] | None = None,
    aspect_ratio: str = "3:4",
    resolution: str = "1K",
) -> tuple[Image.Image | None, str | None]:
    """Generate an image using the specified model.

    Args:
        client: GenAI client instance
        model: Model name to use for generation
        prompt: The prompt text describing the image
        reference_images: Optional list of paths to reference images
        aspect_ratio: Image aspect ratio (default: "3:4")
        resolution: Image resolution (default: "1K")

    Returns:
        Tuple of (generated PIL Image or None, text response or None)

    Raises:
        RuntimeError: If API call fails or content is blocked
    """
    contents = [prompt]
    opened_images = []  # Track opened images for cleanup

    # Add reference images if provided
    if reference_images:
        for img_path in reference_images:
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    opened_images.append(img)
                    contents.append(img)
                except (OSError, IOError) as e:
                    logging.warning(f"无法打开参考图片 {img_path}: {e}")
            else:
                logging.warning(f"参考图片不存在: {img_path}")

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio, image_size=resolution
                ),
            ),
        )
    except Exception as e:
        logging.error(f"图像生成 API 调用失败: {e}")
        raise RuntimeError(f"图像生成失败: {e}") from e
    finally:
        for img in opened_images:
            img.close()

    if response is None:
        raise RuntimeError("图像生成 API 返回空响应")

    # Check for safety/content filtering
    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
            raise RuntimeError(f"提示词被阻止: {response.prompt_feedback.block_reason}")

    # Check for candidates finish reason (similar to generate_text)
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                if candidate.finish_reason not in ('STOP', 'MAX_TOKENS', FINISH_REASON_STOP):
                    logging.warning(f"图像生成可能不完整或被过滤: finish_reason={candidate.finish_reason}")

    result_image = None
    result_text = None
    texts = []

    if response.parts is not None:
        for part in response.parts:
            if part.text is not None:
                texts.append(part.text)
            elif image := part.as_image():
                # Note: If the response contains multiple images, only the last one is retained.
                # This is intentional as we expect single image generation per request.
                result_image = image

        if texts:
            result_text = "\n".join(texts)

    return result_image, result_text


def generate_image_via_chat(
    client: genai.Client,
    model: str,
    prompt: str,
    reference_images: list[str] | None = None,
    aspect_ratio: str = "3:4",
    resolution: str = "1K",
) -> tuple[bytes | None, str | None]:
    """Generate an image using chat-style API and return raw bytes.

    This is a wrapper around generate_image that returns image bytes
    instead of PIL Image, matching the signature of openai_compat.generate_image_via_chat.

    Args:
        client: GenAI client instance
        model: Model name to use for generation
        prompt: The prompt text describing the image
        reference_images: Optional list of paths to reference images
        aspect_ratio: Image aspect ratio (default: "3:4")
        resolution: Image resolution (default: "1K")

    Returns:
        Tuple of (image bytes or None, text response or None)

    Raises:
        RuntimeError: If API call fails or content is blocked
    """
    import io

    # Convert string paths to Path objects
    ref_paths = None
    if reference_images:
        ref_paths = [Path(p) for p in reference_images]

    # Call the existing generate_image function
    result_image, result_text = generate_image(
        client=client,
        model=model,
        prompt=prompt,
        reference_images=ref_paths,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )

    # Convert image to bytes
    image_bytes = None
    if result_image is not None:
        # result_image is a google.genai.types.Image object
        # It has a save(path) method, but we need bytes
        # Try different approaches to extract the image data

        # Method 1: Check for _pil_image attribute (internal PIL image)
        if hasattr(result_image, '_pil_image') and result_image._pil_image is not None:
            buffer = io.BytesIO()
            result_image._pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            image_bytes = buffer.read()
        # Method 2: Check for raw data attribute
        elif hasattr(result_image, 'data') and result_image.data is not None:
            image_bytes = result_image.data
        elif hasattr(result_image, '_loaded_bytes') and result_image._loaded_bytes is not None:
            image_bytes = result_image._loaded_bytes
        # Method 3: Check for image_bytes attribute
        elif hasattr(result_image, 'image_bytes') and result_image.image_bytes is not None:
            image_bytes = result_image.image_bytes
        # Method 4: Save to temp file and read back
        else:
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                result_image.save(tmp_path)
                with open(tmp_path, "rb") as f:
                    image_bytes = f.read()
                os.unlink(tmp_path)
            except Exception as e:
                logging.warning(f"无法从响应中提取图片数据: {e}")

    return image_bytes, result_text
