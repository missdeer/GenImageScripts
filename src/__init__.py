"""GenImageScripts LLM utilities package."""

from .gemini import (
    GeminiConfig,
    GeminiConfig as ClientConfig,
    DEFAULT_BASE_URL,
    DEFAULT_LOCATION,
    generate_text,
    generate_image,
)

from .openai_compat import (
    OpenAIConfig,
    DEFAULT_BASE_URL as OPENAI_DEFAULT_BASE_URL,
    OPENAI_AVAILABLE,
)
from .openai_compat import generate_text as openai_generate_text
from .openai_compat import generate_image as openai_generate_image

__all__ = [
    "ClientConfig",
    "generate_text",
    "generate_image",
    "DEFAULT_BASE_URL",
    "DEFAULT_LOCATION",
    "GeminiConfig",
    "OpenAIConfig",
    "OPENAI_DEFAULT_BASE_URL",
    "OPENAI_AVAILABLE",
    "openai_generate_text",
    "openai_generate_image",
]
