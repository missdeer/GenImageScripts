"""GenImageScripts LLM utilities package."""

from typing import Any, Callable

from .gemini import (
    GeminiConfig as _GeminiConfig,
    DEFAULT_BASE_URL as _GEMINI_DEFAULT_BASE_URL,
    DEFAULT_LOCATION as _GEMINI_DEFAULT_LOCATION,
    generate_text as _gemini_generate_text,
    generate_image as _gemini_generate_image,
)

from .openai_compat import (
    OpenAIConfig as _OpenAIConfig,
    DEFAULT_BASE_URL as _OPENAI_DEFAULT_BASE_URL,
    OPENAI_AVAILABLE,
    generate_text as _openai_generate_text,
    generate_image as _openai_generate_image,
)


class APIService:
    """Container for backend-specific configuration and functions."""

    def __init__(
        self,
        config_class: type,
        generate_text: Callable,
        generate_image: Callable,
        default_base_url: str,
        default_location: str | None = None,
    ):
        self.config_class = config_class
        self.generate_text = generate_text
        self.generate_image = generate_image
        self.default_base_url = default_base_url
        self.default_location = default_location


def get_api_service(backend: str) -> APIService:
    """Get backend-specific configuration and functions.

    Args:
        backend: Backend name, either "gemini" or "openai"

    Returns:
        APIService containing config class and generate functions

    Raises:
        ValueError: If backend is not supported or not available
    """
    if backend == "gemini":
        return APIService(
            config_class=_GeminiConfig,
            generate_text=_gemini_generate_text,
            generate_image=_gemini_generate_image,
            default_base_url=_GEMINI_DEFAULT_BASE_URL,
            default_location=_GEMINI_DEFAULT_LOCATION,
        )
    elif backend == "openai":
        if not OPENAI_AVAILABLE:
            raise ValueError(
                "OpenAI backend selected but 'openai' package is not installed. "
                "Install it with: pip install openai"
            )
        return APIService(
            config_class=_OpenAIConfig,
            generate_text=_openai_generate_text,
            generate_image=_openai_generate_image,
            default_base_url=_OPENAI_DEFAULT_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Must be 'gemini' or 'openai'")


__all__ = [
    "APIService",
    "get_api_service",
    "OPENAI_AVAILABLE",
]
