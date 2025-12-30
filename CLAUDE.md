# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python tool for generating PPT/小红书 (Xiaohongshu) style multi-page images using Google's Gemini AI API. It generates content outlines from a topic, then creates styled images for each page.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with config file
python genRednotes.py -c config.json

# Run with CLI arguments
python genRednotes.py "主题" -k YOUR_API_KEY -o outline_prompt.txt -p image_prompt.txt

# Run with Vertex AI
python genRednotes.py "主题" --vertex --project YOUR_PROJECT_ID -o outline_prompt.txt -p image_prompt.txt
```

## Architecture

**Single-file tool** (`genRednotes.py`) with this workflow:

1. **Outline Generation**: Uses text model (e.g., `gemini-3-pro-preview`) to generate structured outline from topic
2. **Outline Parsing**: Splits outline by `<page>` delimiter, extracts page types (`[封面]`, `[内容]`, `[总结]`)
3. **Cover Generation**: Generates cover page first (synchronously) - used as style reference for subsequent pages
4. **Parallel Page Generation**: Generates remaining pages using multiprocessing pool, referencing both optional style image and generated cover

**Key data structures:**
- `ClientConfig`: Dataclass for API client configuration (supports both API key and Vertex AI modes)
- `PageInfo`: Dataclass representing a page with index, type, and content

**Configuration priority**: CLI args > config file (JSON) > environment variables > defaults

## Prompt Templates

- `outline_prompt.txt`: Template for generating outlines. Uses `{topic}` placeholder.
- `image_prompt.txt`: Template for image generation. Uses `{page_content}`, `{page_type}`, `{topic}`, `{full_outline}` placeholders.

## Output Files

Generated in output directory (default: current dir):
- `outline.txt` - Generated outline
- `Page1.png`, `Page2.png`, ... - Generated images
- `page1.txt`, `page2.txt`, ... - Final prompts used for each page
- `gen.log` - Execution log
