#!/usr/bin/env python

import argparse
import json
import logging
import os
import re
import multiprocessing
from functools import partial
from pathlib import Path
from dataclasses import dataclass

from google import genai
from google.genai import types
from PIL import Image

DEFAULT_ASPECT_RATIO = "3:4"  # "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
DEFAULT_RESOLUTION = "1K"  # "1K", "2K", "4K"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_IMAGE_MODEL = "gemini-3-image-pro-preview"
DEFAULT_TEXT_MODEL = "gemini-3-pro-preview"
DEFAULT_PARALLEL = 2
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
        if self.vertex:
            # Set credentials environment variable if provided
            if self.credentials:
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
            return genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(base_url=self.base_url),
            )

@dataclass
class PageInfo:
    """Represents a single page with its content and type."""
    index: int
    page_type: str  # e.g., "封面", "内容", "总结"
    content: str


def setup_logger(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def generate_outline(
    client: genai.Client,
    text_model: str,
    outline_prompt: str,
    user_topic: str,
) -> str:
    """Generate outline using the text model."""
    # Replace {topic} placeholder in outline prompt
    final_prompt = outline_prompt.replace("{topic}", user_topic)

    logging.info("正在使用文本模型生成大纲...")
    response = client.models.generate_content(
        model=text_model,
        contents=[final_prompt],
    )

    outline_text = response.text if response.text else ""
    logging.info(f"大纲生成完成，共 {len(outline_text)} 字符")
    return outline_text


def parse_outline(outline_text: str) -> list[PageInfo]:
    """Parse the outline text into pages by splitting on <page> delimiter."""
    # Split by <page> delimiter
    raw_pages = outline_text.split("<page>")

    pages = []
    page_type_pattern = re.compile(r"^\s*\[([^\]]+)\]", re.MULTILINE)

    for idx, raw_content in enumerate(raw_pages):
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


def generate_one_page(
    page: PageInfo,
    style_image_path: str | None,
    cover_image_path: str | None,
    client_config: ClientConfig,
    image_model: str,
    image_prompt_template: str,
    user_topic: str,
    full_outline: str,
    aspect_ratio: str,
    resolution: str,
    output_dir: Path,
) -> None:
    """Generate image for a single page."""
    try:
        logging.info(f"开始生成 Page{page.index} [{page.page_type}] …")

        # Replace placeholders in image prompt
        final_prompt = image_prompt_template.replace("{page_content}", page.content)
        final_prompt = final_prompt.replace("{page_type}", page.page_type)
        final_prompt = final_prompt.replace("{topic}", user_topic)
        final_prompt = final_prompt.replace("{full_outline}", full_outline)

        # Save the final prompt to file
        (output_dir / f"page{page.index}.txt").write_text(final_prompt, encoding="utf-8")

        client = client_config.create_client()

        # Build contents with reference images
        contents = [final_prompt]

        # Add style reference image if provided
        if style_image_path and Path(style_image_path).exists():
            contents.append(Image.open(style_image_path))

        # Add cover image as reference for non-cover pages
        if page.page_type != "封面" and cover_image_path and Path(cover_image_path).exists():
            contents.append(Image.open(cover_image_path))

        response = client.models.generate_content(
            model=image_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio, image_size=resolution
                ),
            ),
        )

        text_path = output_dir / f"Page{page.index}.txt"

        if response.parts is not None:
            texts = []
            for part in response.parts:
                if part.text is not None:
                    texts.append(part.text)
                elif image := part.as_image():
                    image.save(output_dir / f"Page{page.index}.png")

            if texts:
                text_path.write_text("\n".join(texts), encoding="utf-8")

        logging.info(f"完成 Page{page.index}")
    except Exception as e:
        logging.exception(f"生成 Page{page.index} 失败: {e}")


def generate_cover_page(
    page: PageInfo,
    style_image_path: str | None,
    client_config: ClientConfig,
    image_model: str,
    image_prompt_template: str,
    user_topic: str,
    full_outline: str,
    aspect_ratio: str,
    resolution: str,
    output_dir: Path,
) -> str | None:
    """Generate cover page image and return its path."""
    try:
        logging.info(f"开始生成封面 Page{page.index} …")

        # Replace placeholders in image prompt
        final_prompt = image_prompt_template.replace("{page_content}", page.content)
        final_prompt = final_prompt.replace("{page_type}", page.page_type)
        final_prompt = final_prompt.replace("{topic}", user_topic)
        final_prompt = final_prompt.replace("{full_outline}", full_outline)

        # Save the final prompt to file
        (output_dir / f"page{page.index}.txt").write_text(final_prompt, encoding="utf-8")

        client = client_config.create_client()

        # Build contents with optional style reference image
        contents = [final_prompt]
        if style_image_path and Path(style_image_path).exists():
            contents.append(Image.open(style_image_path))

        response = client.models.generate_content(
            model=image_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio, image_size=resolution
                ),
            ),
        )

        cover_path = output_dir / f"Page{page.index}.png"
        text_path = output_dir / f"Page{page.index}.txt"

        if response.parts is not None:
            texts = []
            for part in response.parts:
                if part.text is not None:
                    texts.append(part.text)
                elif image := part.as_image():
                    image.save(cover_path)

            if texts:
                text_path.write_text("\n".join(texts), encoding="utf-8")

        logging.info(f"完成封面 Page{page.index}")
        return cover_path
    except Exception as e:
        logging.exception(f"生成封面 Page{page.index} 失败: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PPT/小红书 page images using Gemini/Nano Banana Pro or any other AI image generation API"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="The topic/requirement for generating content (env: GIS_TOPIC)"
    )
    parser.add_argument(
        "-k", "--api-key",
        default=None,
        help="Gemini API key (env: GEMINI_API_KEY)"
    )
    parser.add_argument(
        "-u", "--base-url",
        default=None,
        help="API base URL (env: GEMINI_BASE_URL)"
    )
    parser.add_argument(
        "-r", "--ref-image",
        default=None,
        help="Path to the reference style image (env: GIS_REF_IMAGE)"
    )
    parser.add_argument(
        "-i", "--image-model",
        default=None,
        help="Gemini image model name (env: GEMINI_IMAGE_MODEL)"
    )
    parser.add_argument(
        "-t", "--text-model",
        default=None,
        help="Gemini text model name (env: GEMINI_TEXT_MODEL)"
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
        help=f"Google Cloud location for Vertex AI (env: GIS_LOCATION, default: {DEFAULT_LOCATION})"
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
            config = json.loads(config_file.read_text(encoding="utf-8"))
        else:
            parser.error(f"Config file not found: {config_path}")

    # Helper function to resolve value with priority: CLI > config > env > default
    def resolve(cli_value, config_key: str, env_key: str, default=None):
        if cli_value is not None:
            return cli_value
        if config_key in config:
            return config[config_key]
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
        return default

    # Resolve all settings with priority: CLI > config > env > default
    topic = resolve(args.topic, "topic", "GIS_TOPIC")
    api_key = resolve(args.api_key, "api_key", "GEMINI_API_KEY")
    base_url = resolve(args.base_url, "base_url", "GEMINI_BASE_URL", DEFAULT_BASE_URL)
    ref_image = resolve(args.ref_image, "ref_image", "GIS_REF_IMAGE")
    image_model = resolve(args.image_model, "image_model", "GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    text_model = resolve(args.text_model, "text_model", "GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    outline_prompt_path = resolve(args.outline_prompt, "outline_prompt", "GIS_OUTLINE_PROMPT")
    image_prompt_path = resolve(args.image_prompt, "image_prompt", "GIS_IMAGE_PROMPT")
    aspect_ratio = resolve(args.aspect_ratio, "aspect_ratio", "GIS_ASPECT_RATIO", DEFAULT_ASPECT_RATIO)
    resolution = resolve(args.resolution, "resolution", "GIS_RESOLUTION", DEFAULT_RESOLUTION)
    parallel_str = resolve(args.parallel, "parallel", "GIS_PARALLEL", DEFAULT_PARALLEL)
    parallel = int(parallel_str) if isinstance(parallel_str, str) else parallel_str
    # Vertex AI settings
    vertex_env = os.environ.get("GIS_VERTEX", "").lower() in ("true", "1", "yes")
    vertex = args.vertex if args.vertex is not None else config.get("vertex", vertex_env)
    project = resolve(args.project, "project", "GIS_PROJECT")
    location = resolve(args.location, "location", "GIS_LOCATION", DEFAULT_LOCATION)
    credentials = resolve(args.credentials, "credentials", "GOOGLE_APPLICATION_CREDENTIALS")
    output_directory = resolve(args.output_directory, "output_directory", "GIS_OUTPUT_DIRECTORY", ".")

    # Validate required parameters
    if not topic:
        parser.error("topic is required (provide as argument, in config file, or via GIS_TOPIC env var)")
    if vertex:
        # Vertex AI mode requires project
        if not project:
            parser.error("--project is required for Vertex AI mode (or set in config file or GIS_PROJECT env var)")
    else:
        # API key mode requires api_key
        if not api_key:
            parser.error("--api-key is required (or set in config file or GEMINI_API_KEY env var)")
    # ref_image is optional - if not provided, cover generates freely,
    # and subsequent pages use cover as reference
    if not outline_prompt_path:
        parser.error("--outline-prompt is required (or set in config file or GIS_OUTLINE_PROMPT env var)")
    if not image_prompt_path:
        parser.error("--image-prompt is required (or set in config file or GIS_IMAGE_PROMPT env var)")

    # Create output directory if it doesn't exist
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read prompt templates
    outline_prompt = Path(outline_prompt_path).read_text(encoding="utf-8")
    image_prompt_template = Path(image_prompt_path).read_text(encoding="utf-8")

    # Setup logging (log file goes to output directory)
    log_file = output_dir / "gen.log"
    setup_logger(log_file)

    logging.info(f"主题: {topic}")
    logging.info(f"文本模型: {text_model}")
    logging.info(f"图像模型: {image_model}")
    if vertex:
        logging.info(f"使用 Vertex AI 模式 (project: {project}, location: {location})")
    else:
        logging.info(f"使用 API Key 模式 (base_url: {base_url})")
    logging.info(f"输出目录: {output_dir.absolute()}")

    # Create client configuration
    client_config = ClientConfig(
        api_key=api_key,
        base_url=base_url,
        vertex=vertex,
        project=project,
        location=location,
        credentials=credentials,
    )

    # Create Gemini client for outline generation
    client = client_config.create_client()

    # Step 1: Generate outline using text model
    full_outline = generate_outline(client, text_model, outline_prompt, topic)

    # Save the outline for reference
    (output_dir / "outline.txt").write_text(full_outline, encoding="utf-8")
    logging.info(f"大纲已保存到 {output_dir / 'outline.txt'}")

    # Step 2: Parse outline into pages
    pages = parse_outline(full_outline)
    if not pages:
        logging.error("大纲解析失败，没有找到有效页面")
        return

    total = len(pages)
    logging.info(f"开始生成 {total} 页图片…")

    # Step 3: Generate cover page first (synchronously)
    cover_page = next((p for p in pages if p.page_type == "封面"), pages[0])
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
    )

    # Step 4: Generate remaining pages in parallel
    remaining_pages = [p for p in pages if p.index != cover_page.index]

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
        )

        with multiprocessing.Pool(processes=parallel) as pool:
            for idx, _ in enumerate(pool.imap_unordered(worker, remaining_pages), start=2):
                logging.info(f"整体进度：{idx}/{total} ({idx / total:.0%})")

    logging.info("全部页面生成完成。")


if __name__ == "__main__":
    main()