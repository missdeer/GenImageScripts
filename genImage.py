#!/usr/bin/env python3

import argparse
import json
import sys
import os

from src.openai_compat import OpenAIConfig, generate_image_via_chat as openai_generate_image_via_chat
from src.gemini import GeminiConfig, generate_image_via_chat as gemini_generate_image_via_chat

# 支持的 API 服务
API_SERVICES = ["openai", "gemini", "vertexai"]


def load_config(config_path: str) -> dict:
    """从 JSON 文件加载配置"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件 JSON 格式错误: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"错误: 读取配置文件失败: {e}", file=sys.stderr)
        sys.exit(1)


# 可用的模型列表
AVAILABLE_MODELS = [
    "gemini-3-pro-image-preview (CLIProxyAPI gemini/vertexai 模式)", # 预览版
    "gemini-3-pro-image (AntiGravity-Manager openai 模式)",        # 默认 1:1 比例
    "gemini-3-pro-image-3x4 (AntiGravity-Manager openai 模式)",    # 3:4 小红书风格
    "gemini-3-pro-image-4x3 (AntiGravity-Manager openai 模式)",    # 4:3 标准横图
    "gemini-3-pro-image-9x16 (AntiGravity-Manager openai 模式)",   # 9:16 手机壁纸
    "gemini-3-pro-image-16x9 (AntiGravity-Manager openai 模式)",   # 16:9 横屏
    "gemini-3-pro-image-4k (AntiGravity-Manager openai 模式)",     # 4K 超清图 (1:1)
    "gemini-3-pro-image-16x9-4k (AntiGravity-Manager openai 模式)",# 16:9 4K 超清图
    "gemini-3-pro-image-9x16-4k (AntiGravity-Manager openai 模式)", # 9:16 4K 超清图
    "gemini-3-pro-image-3x4-4k (AntiGravity-Manager openai 模式)", # 3:4 4K 超清图
    "gemini-3-pro-image-4x3-4k (AntiGravity-Manager openai 模式)"  # 4:3 4K 超清图
]

# 默认值
DEFAULTS = {
    "api_service": "gemini",
    "model": "gemini-3-pro-image-preview",
    "base_url": "http://127.0.0.1:8045/v1",
    "api_key": "sk-c526bb53270242339bd07504d11607a4",
    "output": "xhs1.jpg",
    "location": "us-central1",
    "aspect_ratio": "1:1",
    "resolution": "1K",
    "prompt": '''为以下文字生成小红书风格3:4宽高比的图片，要求将以下文本完整显示在图中，不要修改，字体颜色为黑色，字体大一点，不要使图片留出很多空白。添加浅色的科技感的背景，搜索你的知识库或网络并添加Claude Code和Gemini CLI和Codex CLI三者的官方logo在背景上：
"不用 MCP ，不用 SKILLs ，多 agent 协作让 Claude Code 调用 Gemini CLI 和 Codex CLI 只需要在 CLAUDE.md 里加两句话…"'''
}

# 解析命令行参数
parser = argparse.ArgumentParser(
    description="调用 OpenAI/Gemini/Vertex AI API 生成图片",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    "-c", "--config",
    metavar="FILE",
    help="从 JSON 配置文件读取配置（命令行参数优先级更高）"
)
parser.add_argument(
    "-s", "--api-service",
    choices=API_SERVICES,
    help="API 服务类型（默认: gemini）\n可选值: openai, gemini, vertexai"
)
parser.add_argument(
    "-m", "--model",
    help="模型名称（默认: gemini-3-pro-image-preview）\n可选值:\n" +
         "\n".join(f"  {m}" for m in AVAILABLE_MODELS)
)
# 提示词互斥组（-p/--prompt 和 --prompt-file 不能同时使用）
prompt_group = parser.add_mutually_exclusive_group()
prompt_group.add_argument(
    "-p", "--prompt",
    help="图片生成的提示词"
)
prompt_group.add_argument(
    "-f",  "--prompt-file",
    help="从文本文件读取提示词（不能与 -p/--prompt 同时使用）"
)
parser.add_argument(
   "-u", "--base-url",
    help="API 基础 URL（默认: http://127.0.0.1:8045/v1）"
)
parser.add_argument(
   "-k", "--api-key",
    help="API 密钥"
)
parser.add_argument(
   "-o", "--output",
    help="输出图片文件名（默认: xhs1.jpg）"
)
# Vertex AI 专用参数
parser.add_argument(
    "-j", "--project",
    help="Google Cloud 项目 ID（vertexai 模式必需）"
)
parser.add_argument(
    "-l", "--location",
    help="Vertex AI 区域（默认: us-central1）"
)
parser.add_argument(
    "-x", "--credentials",
    help="Google Cloud 凭证文件路径"
)
# Gemini 图片生成参数
parser.add_argument(
    "-t", "--aspect-ratio",
    help="图片宽高比（gemini/vertexai 模式，默认: 1:1）\n可选值: 21:9, 16:9, 3:2, 4:3, 5:3, 5:4, 1:1, 4:5, 3:5, 3:4, 2:3, 9:16, 9:21"
)
parser.add_argument(
    "-r", "--resolution",
    choices=["1K", "2K", "4K"],
    help="图片分辨率（gemini/vertexai 模式，默认: 1K）\n可选值: 1K, 2K, 4K"
)
parser.add_argument(
    "images",
    nargs="*",
    metavar="IMAGE",
    help="要上传的本地图片文件路径，可指定多个（可选）"
)
args = parser.parse_args()


def get_config_value(key: str, cli_value, config: dict) -> str:
    """
    获取配置值，优先级：命令行参数 > 配置文件 > 默认值
    """
    if cli_value is not None:
        return cli_value
    if config and key in config:
        return config[key]
    return DEFAULTS.get(key)


def main():
    # 加载配置文件（如果指定）
    config = {}
    if args.config:
        config = load_config(args.config)

    # 解析 API 服务类型
    api_service = get_config_value("api_service", args.api_service, config)
    if api_service not in API_SERVICES:
        print(f"错误: 无效的 API 服务类型: {api_service}", file=sys.stderr)
        print(f"可选值: {', '.join(API_SERVICES)}", file=sys.stderr)
        sys.exit(1)

    # 解析配置值（命令行参数优先级高于配置文件）
    model = get_config_value("model", args.model, config)
    base_url = get_config_value("base_url", args.base_url, config)
    api_key = get_config_value("api_key", args.api_key, config)
    output = get_config_value("output", args.output, config)
    aspect_ratio = get_config_value("aspect_ratio", args.aspect_ratio, config)
    resolution = get_config_value("resolution", args.resolution, config)

    # Vertex AI 专用参数
    project = args.project or config.get("project")
    location = get_config_value("location", args.location, config)
    credentials = args.credentials or config.get("credentials")

    # 验证 Vertex AI 必需参数
    if api_service == "vertexai" and not project:
        print("错误: vertexai 模式需要指定 --project 参数", file=sys.stderr)
        sys.exit(1)

    # 获取 prompt_file（命令行参数优先级高于配置文件）
    prompt_file = args.prompt_file
    if prompt_file is None and config and "prompt_file" in config:
        # 只有当命令行没有指定 -p/--prompt 时才使用配置文件中的 prompt_file
        if args.prompt is None:
            prompt_file = config.get("prompt_file")

    # 验证 --prompt-file 文件是否存在
    if prompt_file:
        if not os.path.isfile(prompt_file):
            print(f"错误: 提示词文件不存在: {prompt_file}", file=sys.stderr)
            sys.exit(1)

    # 获取图片列表（命令行参数追加到配置文件中的图片列表）
    images = list(args.images) if args.images else []
    if config and "images" in config and isinstance(config["images"], list):
        # 配置文件中的图片放在前面，命令行的放在后面
        images = config["images"] + images

    # 验证所有输入图片文件是否存在
    for image_path in images:
        if not os.path.isfile(image_path):
            print(f"错误: 图片文件不存在: {image_path}", file=sys.stderr)
            sys.exit(1)

    # 验证输出目录是否存在
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.isdir(output_dir):
        print(f"错误: 输出目录不存在: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # 根据 API 服务类型创建客户端
    try:
        if api_service == "openai":
            client_config = OpenAIConfig(
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            generate_image_fn = openai_generate_image_via_chat
        else:
            # gemini 或 vertexai
            client_config = GeminiConfig(
                api_key=api_key if api_service == "gemini" else None,
                base_url=base_url,
                vertex=api_service == "vertexai",
                project=project,
                location=location,
                credentials=credentials,
            )
            generate_image_fn = gemini_generate_image_via_chat

        client = client_config.create_client()
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 创建 API 客户端失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 确定提示词：优先使用命令行参数，其次配置文件，最后使用默认值
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except IOError as e:
            print(f"错误: 读取提示词文件失败: {e}", file=sys.stderr)
            sys.exit(1)
        except UnicodeDecodeError as e:
            print(f"错误: 提示词文件编码错误（需要 UTF-8）: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.prompt:
        prompt_text = args.prompt
    else:
        prompt_text = get_config_value("prompt", None, config)

    if not prompt_text or not prompt_text.strip():
        print("错误: 提示词不能为空", file=sys.stderr)
        sys.exit(1)

    # 调用 API 生成图片
    try:
        if api_service == "openai":
            image_bytes, text_response = generate_image_fn(
                client=client,
                model=model,
                prompt=prompt_text,
                reference_images=images if images else None,
            )
        else:
            # gemini 或 vertexai
            image_bytes, text_response = generate_image_fn(
                client=client,
                model=model,
                prompt=prompt_text,
                reference_images=images if images else None,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: API 调用失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 检查是否生成了图片
    if image_bytes is None:
        if text_response:
            print(f"错误: 响应中未找到图片数据。响应内容: {text_response[:500]}...", file=sys.stderr)
        else:
            print("错误: 响应中未找到图片数据", file=sys.stderr)
        sys.exit(1)

    # 保存图片
    try:
        with open(output, 'wb') as f:
            f.write(image_bytes)
        print(f"图片已保存到: {output}")
    except IOError as e:
        print(f"错误: 保存图片失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
