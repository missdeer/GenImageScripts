#!/usr/bin/env python
"""LLM client utilities for text and image generation using Google GenAI."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_LOCATION = "us-central1"


@dataclass
class ClientConfig:
    """Configuration for creating GenAI client."""
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
        logging.error(f"Text generation API call failed: {e}")
        raise RuntimeError(f"Failed to generate text: {e}") from e

    if response is None:
        raise RuntimeError("API returned None response for text generation")

    # Check for blocked content or safety filters
    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
            raise RuntimeError(f"Prompt was blocked: {response.prompt_feedback.block_reason}")

    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                # finish_reason can be string ('STOP', 'MAX_TOKENS') or int (1 = STOP in protobuf enum)
                if candidate.finish_reason not in ('STOP', 'MAX_TOKENS', 1):
                    logging.warning(f"Text generation may be truncated or filtered: finish_reason={candidate.finish_reason}")

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
                    logging.warning(f"Failed to open reference image {img_path}: {e}")
            else:
                logging.warning(f"Reference image not found: {img_path}")

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
        logging.error(f"Image generation API call failed: {e}")
        raise RuntimeError(f"Failed to generate image: {e}") from e
    finally:
        for img in opened_images:
            img.close()

    if response is None:
        raise RuntimeError("API returned None response for image generation")

    # Check for safety/content filtering
    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
            raise RuntimeError(f"Prompt was blocked: {response.prompt_feedback.block_reason}")

    result_image = None
    result_text = None
    texts = []

    if response.parts is not None:
        for part in response.parts:
            if part.text is not None:
                texts.append(part.text)
            elif image := part.as_image():
                result_image = image

        if texts:
            result_text = "\n".join(texts)

    return result_image, result_text
