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
    generate_text as _openai_generate_text,
    generate_image as _openai_generate_image,
)


class APIService:
    """Container for API service-specific configuration and functions."""

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


def get_api_service(api_service_name: str) -> APIService:
    """Get API service-specific configuration and functions.

    Args:
        api_service_name: API service name, either "gemini" or "openai"

    Returns:
        APIService containing config class and generate functions

    Raises:
        ValueError: If api_service is not supported or not available
    """
    if api_service_name == "gemini":
        return APIService(
            config_class=_GeminiConfig,
            generate_text=_gemini_generate_text,
            generate_image=_gemini_generate_image,
            default_base_url=_GEMINI_DEFAULT_BASE_URL,
            default_location=_GEMINI_DEFAULT_LOCATION,
        )
    elif api_service_name == "openai":
        return APIService(
            config_class=_OpenAIConfig,
            generate_text=_openai_generate_text,
            generate_image=_openai_generate_image,
            default_base_url=_OPENAI_DEFAULT_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported api_service: {api_service_name}. Must be 'gemini' or 'openai'")


__all__ = [
    "APIService",
    "get_api_service",
]
