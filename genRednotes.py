#!/usr/bin/env python

import argparse
import json
import logging
import os
import re
import sys
import multiprocessing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

try:
    from src import get_api_service
except ImportError as e:
    print(f"Error: Failed to import src module: {e}")
    sys.exit(1)

DEFAULT_BACKEND = "gemini"  # "gemini" or "openai"
DEFAULT_ASPECT_RATIO = "3:4"  # "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
DEFAULT_RESOLUTION = "1K"  # "1K", "2K", "4K"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3-image-pro-preview"
DEFAULT_GEMINI_TEXT_MODEL = "gemini-3-pro-preview"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1"
DEFAULT_OPENAI_TEXT_MODEL = "gpt-4o"
DEFAULT_PARALLEL = 2



@dataclass
class PageInfo:
    """Represents a single page with its content and type."""
    index: int
    page_type: str  # e.g., "封面", "内容", "总结"
    content: str


def parse_bool(value: Any) -> bool:
    """Parse a value as boolean.
    
    Accepts bool, or string values like 'true', '1', 'yes' (case-insensitive).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def resolve_config(
    cli_value: Any,
    config: dict,
    config_key: str,
    env_key: str,
    default: Any = None,
) -> Any:
    """Resolve a configuration value with priority: CLI > config file > env var > default.
    
    Args:
        cli_value: Value from command line argument
        config: Configuration dictionary from config file
        config_key: Key to look up in config dictionary
        env_key: Environment variable name
        default: Default value if none of the above are set
    
    Returns:
        Resolved configuration value
    """
    if cli_value is not None:
        return cli_value
    if config_key in config:
        return config[config_key]
    env_value = os.environ.get(env_key)
    if env_value is not None:
        return env_value
    return default


def setup_logger(log_file: Path) -> None:
    """Configure logging to output to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _init_worker(log_file: Path) -> None:
    """Initialize worker process logging.
    
    On Windows, multiprocessing uses 'spawn' which doesn't inherit
    the parent's logging configuration. This initializer ensures
    each worker process has proper logging setup.
    """
    # Reset logging configuration for this process
    root = logging.getLogger()
    root.handlers.clear()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def generate_outline(
    client,
    text_model: str,
    outline_prompt: str,
    user_topic: str,
    generate_text_fn,
) -> str:
    """Generate outline using the text model."""
    final_prompt = outline_prompt.replace("{topic}", user_topic)

    logging.info("正在使用文本模型生成大纲...")
    outline_text = generate_text_fn(client, text_model, final_prompt)

    if not outline_text.strip():
        logging.warning("大纲生成结果为空")

    logging.info(f"大纲生成完成，共 {len(outline_text)} 字符")
    return outline_text


def parse_outline(outline_text: str) -> list[PageInfo]:
    """Parse the outline text into pages by splitting on <page> delimiter."""
    # Split by <page> delimiter
    raw_pages = outline_text.split("<page>")

    pages = []
    page_type_pattern = re.compile(r"^\s*\[([^\]]+)\]", re.MULTILINE)

    for raw_content in raw_pages:
        content = raw_content.strip()
        if not content:
            continue

        # Extract page type from brackets at the beginning
        match = page_type_pattern.search(content)
        if match:
            page_type = match.group(1)
        else:
            page_type = "内容"  # Default to content type

        pages.append(PageInfo(
            index=len(pages) + 1,
            page_type=page_type,
            content=content,
        ))

    logging.info(f"解析大纲完成，共 {len(pages)} 页")
    return pages


def _generate_page_image(
    page: PageInfo,
    client_config: Any,
    image_model: str,
    image_prompt_template: str,
    user_topic: str,
    full_outline: str,
    aspect_ratio: str,
    resolution: str,
    output_dir: Path,
    generate_image_fn,
    ref_images: list[Path] | None = None,
) -> str | None:
    """Generate image for a page (internal helper).

    Args:
        page: Page information
        client_config: Client configuration
        image_model: Model name for image generation
        image_prompt_template: Template for image prompt
        user_topic: User's topic
        full_outline: Full outline text
        aspect_ratio: Image aspect ratio
        resolution: Image resolution
        output_dir: Output directory
        generate_image_fn: Function to generate image
        ref_images: Optional list of reference image paths

    Returns:
        Path to generated image, or None if generation failed.
    """
    page_label = f"Page{page.index}"
    is_cover = page.page_type == "封面"
    log_prefix = "封面 " if is_cover else ""

    logging.info(f"开始生成{log_prefix}{page_label} [{page.page_type}] …")

    # Replace placeholders in image prompt
    final_prompt = image_prompt_template.replace("{page_content}", page.content)
    final_prompt = final_prompt.replace("{page_type}", page.page_type)
    final_prompt = final_prompt.replace("{topic}", user_topic)
    final_prompt = final_prompt.replace("{full_outline}", full_outline)

    # Save the final prompt to file
    try:
        (output_dir / f"page{page.index}.txt").write_text(final_prompt, encoding="utf-8")
    except OSError as e:
        logging.error(f"无法保存 {page_label} 的 prompt 文件: {e}")
        return None

    try:
        client = client_config.create_client()
    except Exception as e:
        logging.error(f"创建 API 客户端失败 ({page_label}): {e}")
        return None

    try:
        result_image, result_text = generate_image_fn(
            client, image_model, final_prompt,
            reference_images=ref_images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )
    except RuntimeError as e:
        logging.error(f"图像生成失败 ({page_label}): {e}")
        return None

    image_path = output_dir / f"{page_label}.png"

    # Save results
    if result_image:
        try:
            result_image.save(image_path)
        except (OSError, IOError) as e:
            logging.error(f"无法保存 {page_label} 图片: {e}")
            return None
    else:
        logging.error(f"{page_label} 没有生成图片")
        return None

    if result_text:
        try:
            (output_dir / f"{page_label}.txt").write_text(result_text, encoding="utf-8")
        except OSError as e:
            logging.warning(f"无法保存 {page_label} 文本响应: {e}")

    logging.info(f"完成{log_prefix}{page_label}")
    return str(image_path)


def generate_one_page(
    page: PageInfo,
    style_image_path: str | None,
    cover_image_path: str | None,
    client_config: Any,
    image_model: str,
    image_prompt_template: str,
    user_topic: str,
    full_outline: str,
    aspect_ratio: str,
    resolution: str,
    output_dir: Path,
    generate_image_fn,
) -> bool:
    """Generate image for a single page.

    Returns:
        True if generation succeeded, False otherwise.
    """
    try:
        # Build reference images list
        ref_images = []
        if style_image_path:
            ref_images.append(Path(style_image_path))
        if page.page_type != "封面" and cover_image_path:
            ref_images.append(Path(cover_image_path))

        result = _generate_page_image(
            page=page,
            client_config=client_config,
            image_model=image_model,
            image_prompt_template=image_prompt_template,
            user_topic=user_topic,
            full_outline=full_outline,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_dir=output_dir,
            generate_image_fn=generate_image_fn,
            ref_images=ref_images if ref_images else None,
        )
        return result is not None

    except KeyboardInterrupt:
        logging.warning(f"Page{page.index} 生成被用户中断")
        raise
    except Exception as e:
        logging.exception(f"生成 Page{page.index} 时发生未预期错误: {e}")
        return False


def generate_cover_page(
    page: PageInfo,
    style_image_path: str | None,
    client_config: Any,
    image_model: str,
    image_prompt_template: str,
    user_topic: str,
    full_outline: str,
    aspect_ratio: str,
    resolution: str,
    output_dir: Path,
    generate_image_fn,
) -> str | None:
    """Generate cover page image and return its path.

    Returns:
        Path to the generated cover image, or None if generation failed.
    """
    try:
        ref_images = [Path(style_image_path)] if style_image_path else None

        return _generate_page_image(
            page=page,
            client_config=client_config,
            image_model=image_model,
            image_prompt_template=image_prompt_template,
            user_topic=user_topic,
            full_outline=full_outline,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_dir=output_dir,
            generate_image_fn=generate_image_fn,
            ref_images=ref_images,
        )

    except KeyboardInterrupt:
        logging.warning("封面生成被用户中断")
        raise
    except Exception as e:
        logging.exception(f"生成封面 Page{page.index} 时发生未预期错误: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PPT/小红书 page images using Gemini or OpenAI-compatible API"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="The topic/requirement for generating content (env: GIS_TOPIC)"
    )
    parser.add_argument(
        "-b", "--backend",
        default=None,
        choices=["gemini", "openai"],
        help=f"Backend to use for API calls (env: GIS_BACKEND, default: {DEFAULT_BACKEND})"
    )
    parser.add_argument(
        "-k", "--api-key",
        default=None,
        help="API key (env: GEMINI_API_KEY for gemini, OPENAI_API_KEY for openai)"
    )
    parser.add_argument(
        "-u", "--base-url",
        default=None,
        help="API base URL (env: GEMINI_BASE_URL for gemini, OPENAI_BASE_URL for openai)"
    )
    parser.add_argument(
        "-r", "--ref-image",
        default=None,
        help="Path to the reference style image (env: GIS_REF_IMAGE)"
    )
    parser.add_argument(
        "-i", "--image-model",
        default=None,
        help="Image model name (env: GIS_IMAGE_MODEL, backend-specific defaults apply)"
    )
    parser.add_argument(
        "-t", "--text-model",
        default=None,
        help="Text model name (env: GIS_TEXT_MODEL, backend-specific defaults apply)"
    )
    parser.add_argument(
        "-o", "--outline-prompt",
        default=None,
        help="Path to a file containing the outline prompt template (env: GIS_OUTLINE_PROMPT)"
    )
    parser.add_argument(
        "-p", "--image-prompt",
        default=None,
        help="Path to a file containing the image prompt template (env: GIS_IMAGE_PROMPT)"
    )
    parser.add_argument(
        "-a", "--aspect-ratio",
        default=None,
        choices=["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"],
        help=f"Image aspect ratio (env: GIS_ASPECT_RATIO, default: {DEFAULT_ASPECT_RATIO})"
    )
    parser.add_argument(
        "-s", "--resolution",
        default=None,
        choices=["1K", "2K", "4K"],
        help=f"Image resolution (env: GIS_RESOLUTION, default: {DEFAULT_RESOLUTION})"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to a JSON configuration file (env: GIS_CONFIG). Priority: CLI > config file > env var"
    )
    parser.add_argument(
        "-j", "--parallel",
        type=int,
        default=None,
        help=f"Number of parallel processes for image generation (env: GIS_PARALLEL, default: {DEFAULT_PARALLEL})"
    )
    # Vertex AI arguments
    parser.add_argument(
        "--vertex",
        action="store_true",
        default=None,
        help="Use Vertex AI instead of API key authentication (env: GIS_VERTEX)"
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Google Cloud project ID for Vertex AI (env: GIS_PROJECT)"
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Google Cloud location for Vertex AI (env: GIS_LOCATION, default: us-central1)"
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="Path to service account JSON key file for Vertex AI (env: GOOGLE_APPLICATION_CREDENTIALS)"
    )
    parser.add_argument(
        "-d", "--output-directory",
        default=None,
        help="Directory to store output files (.txt and .png) (env: GIS_OUTPUT_DIRECTORY, default: current directory)"
    )

    args = parser.parse_args()

    # Load config file if provided (CLI or env var)
    config_path = args.config or os.environ.get("GIS_CONFIG")
    config = {}
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            try:
                config_content = config_file.read_text(encoding="utf-8")
            except OSError as e:
                parser.error(f"Failed to read config file {config_path}: {e}")
            except UnicodeDecodeError as e:
                parser.error(f"Config file {config_path} has invalid encoding (expected UTF-8): {e}")
            
            try:
                config = json.loads(config_content)
            except json.JSONDecodeError as e:
                parser.error(f"Config file {config_path} contains invalid JSON: {e}")
            
            if not isinstance(config, dict):
                parser.error(f"Config file {config_path} must contain a JSON object, got {type(config).__name__}")
        else:
            parser.error(f"Config file not found: {config_path}")

    # Helper function to resolve value with priority: CLI > config > env > default
    def resolve(cli_value, config_key: str, env_key: str, default=None):
        return resolve_config(cli_value, config, config_key, env_key, default)

    # Resolve backend first (it affects other defaults)
    backend = resolve(args.backend, "backend", "GIS_BACKEND", DEFAULT_BACKEND)
    if backend not in ("gemini", "openai"):
        parser.error(f"--backend must be 'gemini' or 'openai', got: {backend}")

    # Get backend-specific configuration and functions
    try:
        api_service = get_api_service(backend)
    except ValueError as e:
        parser.error(str(e))

    # Resolve all settings with priority: CLI > config > env > default
    topic = resolve(args.topic, "topic", "GIS_TOPIC")
    ref_image = resolve(args.ref_image, "ref_image", "GIS_REF_IMAGE")
    outline_prompt_path = resolve(args.outline_prompt, "outline_prompt", "GIS_OUTLINE_PROMPT")
    image_prompt_path = resolve(args.image_prompt, "image_prompt", "GIS_IMAGE_PROMPT")
    aspect_ratio = resolve(args.aspect_ratio, "aspect_ratio", "GIS_ASPECT_RATIO", DEFAULT_ASPECT_RATIO)
    resolution = resolve(args.resolution, "resolution", "GIS_RESOLUTION", DEFAULT_RESOLUTION)

    # Backend-specific settings
    if backend == "gemini":
        api_key = resolve(args.api_key, "api_key", "GEMINI_API_KEY")
        base_url = resolve(args.base_url, "base_url", "GEMINI_BASE_URL", api_service.default_base_url)
        image_model = resolve(args.image_model, "image_model", "GIS_IMAGE_MODEL", DEFAULT_GEMINI_IMAGE_MODEL)
        text_model = resolve(args.text_model, "text_model", "GIS_TEXT_MODEL", DEFAULT_GEMINI_TEXT_MODEL)
    else:  # openai
        api_key = resolve(args.api_key, "api_key", "OPENAI_API_KEY")
        base_url = resolve(args.base_url, "base_url", "OPENAI_BASE_URL", api_service.default_base_url)
        image_model = resolve(args.image_model, "image_model", "GIS_IMAGE_MODEL", DEFAULT_OPENAI_IMAGE_MODEL)
        text_model = resolve(args.text_model, "text_model", "GIS_TEXT_MODEL", DEFAULT_OPENAI_TEXT_MODEL)

    # Parse parallel with explicit error handling for invalid values
    parallel_str = resolve(args.parallel, "parallel", "GIS_PARALLEL", DEFAULT_PARALLEL)
    try:
        parallel = int(parallel_str) if isinstance(parallel_str, str) else parallel_str
    except ValueError:
        parser.error(f"--parallel must be a valid integer, got: {parallel_str}")
    if parallel < 1:
        parser.error(f"--parallel must be at least 1, got {parallel}")
    
    # Vertex AI settings (only for Gemini backend) - use parse_bool for consistent boolean handling
    vertex_raw = resolve(args.vertex, "vertex", "GIS_VERTEX", False)
    vertex = parse_bool(vertex_raw)
    project = resolve(args.project, "project", "GIS_PROJECT")
    location = resolve(args.location, "location", "GIS_LOCATION", api_service.default_location or "us-central1")
    credentials = resolve(args.credentials, "credentials", "GOOGLE_APPLICATION_CREDENTIALS")
    output_directory = resolve(args.output_directory, "output_directory", "GIS_OUTPUT_DIRECTORY", ".")

    # Validate required parameters
    if not topic:
        parser.error("topic is required (provide as argument, in config file, or via GIS_TOPIC env var)")
    
    if backend == "gemini":
        if vertex:
            # Vertex AI mode requires project
            if not project:
                parser.error("--project is required for Vertex AI mode (or set in config file or GIS_PROJECT env var)")
        else:
            # API key mode requires api_key
            if not api_key:
                parser.error("--api-key is required (or set in config file or GEMINI_API_KEY env var)")
    else:  # openai
        if not api_key:
            parser.error("--api-key is required for OpenAI backend (or set in config file or OPENAI_API_KEY env var)")

    # ref_image is optional - if not provided, cover generates freely,
    # and subsequent pages use cover as reference
    if not outline_prompt_path:
        parser.error("--outline-prompt is required (or set in config file or GIS_OUTLINE_PROMPT env var)")
    if not image_prompt_path:
        parser.error("--image-prompt is required (or set in config file or GIS_IMAGE_PROMPT env var)")

    # Create output directory if it doesn't exist
    output_dir = Path(output_directory)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        parser.error(f"Failed to create output directory {output_directory}: {e}")

    # Check output directory is writable
    if not os.access(output_dir, os.W_OK):
        parser.error(f"Output directory is not writable: {output_directory}")

    # Read prompt templates
    outline_prompt_file = Path(outline_prompt_path)
    if not outline_prompt_file.exists():
        parser.error(f"Outline prompt file not found: {outline_prompt_path}")
    try:
        outline_prompt = outline_prompt_file.read_text(encoding="utf-8")
    except OSError as e:
        parser.error(f"Failed to read outline prompt file {outline_prompt_path}: {e}")
    except UnicodeDecodeError as e:
        parser.error(f"Outline prompt file {outline_prompt_path} has invalid encoding: {e}")

    if not outline_prompt.strip():
        parser.error(f"Outline prompt file is empty: {outline_prompt_path}")

    image_prompt_file = Path(image_prompt_path)
    if not image_prompt_file.exists():
        parser.error(f"Image prompt file not found: {image_prompt_path}")
    try:
        image_prompt_template = image_prompt_file.read_text(encoding="utf-8")
    except OSError as e:
        parser.error(f"Failed to read image prompt file {image_prompt_path}: {e}")
    except UnicodeDecodeError as e:
        parser.error(f"Image prompt file {image_prompt_path} has invalid encoding: {e}")

    if not image_prompt_template.strip():
        parser.error(f"Image prompt file is empty: {image_prompt_path}")

    # Validate reference image if provided (check is_file before logger setup)
    ref_image_missing = False
    if ref_image:
        ref_image_path = Path(ref_image)
        if not ref_image_path.exists():
            ref_image_missing = True
        elif not ref_image_path.is_file():
            parser.error(f"Reference image path is not a file: {ref_image}")

    # Setup logging (log file goes to output directory)
    log_file = output_dir / "gen.log"
    setup_logger(log_file)

    # Log reference image warning after logger is configured
    if ref_image_missing:
        logging.warning(f"Reference image not found: {ref_image}")

    logging.info(f"主题: {topic}")
    logging.info(f"后端: {backend}")
    logging.info(f"文本模型: {text_model}")
    logging.info(f"图像模型: {image_model}")
    if backend == "gemini":
        if vertex:
            logging.info(f"使用 Vertex AI 模式 (project: {project}, location: {location})")
        else:
            logging.info(f"使用 API Key 模式 (base_url: {base_url})")
    else:  # openai
        logging.info(f"使用 OpenAI API (base_url: {base_url})")
    logging.info(f"输出目录: {output_dir.absolute()}")

    # Create client configuration based on backend
    if backend == "gemini":
        client_config = api_service.config_class(
            api_key=api_key,
            base_url=base_url,
            vertex=vertex,
            project=project,
            location=location,
            credentials=credentials,
        )
    else:  # openai
        client_config = api_service.config_class(
            api_key=api_key,
            base_url=base_url,
        )

    # Create API client for outline generation
    try:
        client = client_config.create_client()
    except FileNotFoundError as e:
        logging.error(f"凭据文件错误: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"配置错误: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logging.error(f"创建 API 客户端失败: {e}")
        sys.exit(1)

    # Step 1: Generate outline using text model
    try:
        full_outline = generate_outline(
            client, text_model, outline_prompt, topic, api_service.generate_text
        )
    except RuntimeError as e:
        logging.error(f"生成大纲失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("大纲生成被用户中断")
        sys.exit(130)  # Standard exit code for Ctrl+C

    if not full_outline.strip():
        logging.error("生成的大纲为空，无法继续")
        sys.exit(1)

    # Save the outline for reference
    try:
        (output_dir / "outline.txt").write_text(full_outline, encoding="utf-8")
        logging.info(f"大纲已保存到 {output_dir / 'outline.txt'}")
    except OSError as e:
        logging.error(f"无法保存大纲文件: {e}")
        sys.exit(1)

    # Step 2: Parse outline into pages
    pages = parse_outline(full_outline)
    if not pages:
        logging.error("大纲解析失败，没有找到有效页面")
        sys.exit(1)

    total = len(pages)
    logging.info(f"开始生成 {total} 页图片…")

    # Step 3: Generate cover page first (synchronously)
    cover_page = next((p for p in pages if p.page_type == "封面"), pages[0])
    try:
        cover_image_path = generate_cover_page(
            page=cover_page,
            style_image_path=ref_image,
            client_config=client_config,
            image_model=image_model,
            image_prompt_template=image_prompt_template,
            user_topic=topic,
            full_outline=full_outline,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_dir=output_dir,
            generate_image_fn=api_service.generate_image,
        )
    except KeyboardInterrupt:
        logging.warning("封面生成被用户中断，程序退出")
        sys.exit(130)

    if cover_image_path is None:
        logging.warning("封面生成失败，后续页面将不使用封面作为参考")

    # Step 4: Generate remaining pages in parallel
    remaining_pages = [p for p in pages if p.index != cover_page.index]
    failed_pages = []
    success_count = 1 if cover_image_path else 0  # Count cover only if it succeeded

    if remaining_pages:
        worker = partial(
            generate_one_page,
            style_image_path=ref_image,
            cover_image_path=cover_image_path,
            client_config=client_config,
            image_model=image_model,
            image_prompt_template=image_prompt_template,
            user_topic=topic,
            full_outline=full_outline,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_dir=output_dir,
            generate_image_fn=api_service.generate_image,
        )

        try:
            with multiprocessing.Pool(
                processes=parallel,
                initializer=_init_worker,
                initargs=(log_file,),
            ) as pool:
                try:
                    results = list(zip(remaining_pages, pool.imap(worker, remaining_pages)))
                    for page, success in results:
                        if success:
                            success_count += 1
                        else:
                            failed_pages.append(page.index)
                        logging.info(f"整体进度：{success_count}/{total}")
                except KeyboardInterrupt:
                    logging.warning("图像生成被用户中断，正在终止进程池...")
                    pool.terminate()
                    pool.join()
                    sys.exit(130)
        except OSError as e:
            logging.error(f"进程池创建失败: {e}")
            logging.info("回退到串行处理模式...")
            # Fallback to serial processing
            for page in remaining_pages:
                try:
                    success = worker(page)
                    if success:
                        success_count += 1
                    else:
                        failed_pages.append(page.index)
                    logging.info(f"整体进度：{success_count}/{total}")
                except KeyboardInterrupt:
                    logging.warning("图像生成被用户中断")
                    sys.exit(130)

    # Summary
    if failed_pages:
        logging.warning(f"以下页面生成失败: {failed_pages}")
        logging.info(f"成功生成 {success_count}/{total} 页 ({success_count / total:.0%})")
    else:
        logging.info(f"全部 {total} 页图片生成完成。")


if __name__ == "__main__":
    main()