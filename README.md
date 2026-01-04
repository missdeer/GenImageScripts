# PPT/小红书 图片生成工具

使用 Gemini/Nano Banana Pro 或其他 AI 图像生成 API 生成 PPT/小红书风格的多页图片。

## 使用方法

```bash
python genRednoets.py "主题" [选项...]
```

## 参数说明

所有参数都支持三种输入方式，优先级为：**命令行参数 > 配置文件 > 环境变量 > 默认值**

| 参数 | 简写 | 环境变量 | 默认值 | 说明 |
|------|------|----------|--------|------|
| `topic` | - | `GRN_TOPIC` | 必填 | 生成内容的主题（位置参数） |
| `--api-key` | `-k` | `GEMINI_API_KEY` | 必填* | Gemini API 密钥 |
| `--base-url` | `-u` | `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | API 基础 URL |
| `--ref-image` | `-r` | `GRN_REF_IMAGE` | - | 参考风格图片路径（可选，不提供时封面自由生成，后续页面以封面为参考） |
| `--image-model` | `-i` | `GEMINI_IMAGE_MODEL` | `gemini-3-image-pro-preview` | 图像生成模型名称 |
| `--text-model` | `-t` | `GEMINI_TEXT_MODEL` | `gemini-3-pro-preview` | 文本生成模型名称 |
| `--outline-prompt` | `-o` | `GRN_OUTLINE_PROMPT` | 必填 | 大纲提示词模板文件路径 |
| `--image-prompt` | `-p` | `GRN_IMAGE_PROMPT` | 必填 | 图片提示词模板文件路径 |
| `--aspect-ratio` | `-a` | `GRN_ASPECT_RATIO` | `3:4` | 图片比例 |
| `--resolution` | `-s` | `GRN_RESOLUTION` | `1K` | 图片分辨率 |
| `--config` | `-c` | `GRN_CONFIG` | - | JSON 配置文件路径 |
| `--parallel` | `-j` | `GRN_PARALLEL` | `2` | 并行生成图片的进程数 |
| `--output-directory` | `-d` | `GRN_OUTPUT_DIRECTORY` | `.` | 输出文件目录（存放 .txt 和 .png 文件） |

### Vertex AI 参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| `--vertex` | `GRN_VERTEX` | `false` | 启用 Vertex AI 模式 |
| `--project` | `GRN_PROJECT` | 必填* | Google Cloud 项目 ID |
| `--location` | `GRN_LOCATION` | `us-central1` | Google Cloud 区域 |
| `--credentials` | `GOOGLE_APPLICATION_CREDENTIALS` | - | 服务账号 JSON 密钥文件路径 |

### 两种认证模式对比

| 参数 | API Key 模式 | Vertex AI 模式 |
|------|-------------|----------------|
| `api_key` | ✅ 必需 | ❌ 不需要 |
| `base_url` | ✅ 可选（有默认值） | ✅ 可选（支持私有端点） |
| `project` | ❌ 不需要 | ✅ 必需 |
| `location` | ❌ 不需要 | ✅ 可选（默认 us-central1） |
| `credentials` | ❌ 不需要 | ✅ 需要* |

> **注意**: 
> - Vertex AI 模式下，SDK 默认根据 `project` 和 `location` 自动构建 API 端点
> - 如需使用私有端点，可通过 `base_url` 自定义
> - `credentials` 在 GCP 虚拟机等环境下可自动获取，但普通用户通常需要提供服务账号 JSON 文件

### 可选值

- **aspect-ratio**: `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`
- **resolution**: `1K`, `2K`, `4K`


## 配置文件示例

创建 `config.json` 文件：

```json
{
    "topic": "AI 入门教程",
    "api_key": "your-api-key",
    "base_url": "https://generativelanguage.googleapis.com",
    "ref_image": "style.png",
    "image_model": "gemini-3-image-pro-preview",
    "text_model": "gemini-3-pro-preview",
    "outline_prompt": "outline_prompt.txt",
    "image_prompt": "image_prompt.txt",
    "aspect_ratio": "3:4",
    "resolution": "1K",
    "parallel": 2,

    "vertex": false,
    "project": "your-gcp-project-id",
    "location": "us-central1",
    "credentials": "path/to/service-account.json",
    "output_directory": "./output"
}
```

然后运行：

```bash
python genRednoets.py -c config.json
```

## 使用示例

### 命令行方式

```bash
python genRednoets.py "Python入门教程" -k YOUR_API_KEY -r style.png -o outline.txt -p image.txt -d ./output
```

### 环境变量方式

```bash
# Windows
set GEMINI_API_KEY=your-api-key
set GRN_REF_IMAGE=style.png
set GRN_OUTLINE_PROMPT=outline.txt
set GRN_IMAGE_PROMPT=image.txt
python genRednoets.py "Python入门教程"

# Linux/macOS
export GEMINI_API_KEY=your-api-key
export GRN_REF_IMAGE=style.png
export GRN_OUTLINE_PROMPT=outline.txt
export GRN_IMAGE_PROMPT=image.txt
python genRednoets.py "Python入门教程"
```

### 混合方式

```bash
# 环境变量设置通用配置
set GEMINI_API_KEY=your-api-key
set GRN_REF_IMAGE=style.png

# 命令行覆盖特定参数
python genRednoets.py "主题" -o outline.txt -p image.txt -a 16:9
```

## 输出文件

所有输出文件默认保存在当前目录，可通过 `--output-directory` 指定其他目录：

- `outline.txt` - 生成的大纲文本
- `Page1.png`, `Page2.png`, ... - 生成的各页图片
- `page1.txt`, `page2.txt`, ... - 每页使用的最终提示词
- `gen.log` - 运行日志

## 依赖

- Python >= 3.10
