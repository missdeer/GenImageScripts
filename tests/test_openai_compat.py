#!/usr/bin/env python
"""Basic tests for openai_compat.py module."""

import base64
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from PIL import Image

# Mock openai module
import sys
mock_openai = MagicMock()
sys.modules['openai'] = mock_openai

from src.openai_compat import (
    OpenAIConfig, 
    generate_text, 
    generate_image,
    DEFAULT_BASE_URL,
    OPENAI_AVAILABLE,
)


class TestOpenAIConfig(unittest.TestCase):
    """Test cases for OpenAIConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = OpenAIConfig()
        self.assertIsNone(config.api_key)
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertIsNone(config.model)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OpenAIConfig(
            api_key="sk-test-key",
            base_url="https://custom.openai.com/v1",
            model="gpt-4",
        )
        self.assertEqual(config.api_key, "sk-test-key")
        self.assertEqual(config.base_url, "https://custom.openai.com/v1")
        self.assertEqual(config.model, "gpt-4")
    
    def test_create_client_requires_api_key(self):
        """Test that create_client raises error without API key."""
        config = OpenAIConfig()
        with self.assertRaises(ValueError) as ctx:
            config.create_client()
        self.assertIn("API key is required", str(ctx.exception))


class TestGenerateText(unittest.TestCase):
    """Test cases for generate_text function."""
    
    def test_generate_text_returns_content(self):
        """Test that generate_text returns generated text."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Hello! How can I help you?"
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        result = generate_text(mock_client, "gpt-4", "Hello!")
        
        self.assertEqual(result, "Hello! How can I help you?")
        mock_client.chat.completions.create.assert_called_once()
    
    def test_generate_text_handles_empty_content(self):
        """Test handling of empty message content."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = None
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        result = generate_text(mock_client, "gpt-4", "Test")
        
        self.assertEqual(result, "")
    
    def test_generate_text_raises_on_empty_choices(self):
        """Test that empty choices raises RuntimeError."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as ctx:
            generate_text(mock_client, "gpt-4", "Test")
        self.assertIn("空选择列表", str(ctx.exception))
    
    def test_generate_text_raises_on_none_response(self):
        """Test that None response raises RuntimeError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = None
        
        with self.assertRaises(RuntimeError) as ctx:
            generate_text(mock_client, "gpt-4", "Test")
        self.assertIn("空响应", str(ctx.exception))


class TestGenerateImage(unittest.TestCase):
    """Test cases for generate_image function."""
    
    def _create_test_image_b64(self) -> str:
        """Create a small test image and return base64 encoded data."""
        img = Image.new('RGB', (10, 10), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_generate_image_without_reference(self):
        """Test image generation without reference images."""
        mock_client = MagicMock()
        
        mock_image_data = MagicMock()
        mock_image_data.b64_json = self._create_test_image_b64()
        mock_image_data.revised_prompt = "A beautiful sunset"
        
        mock_response = MagicMock()
        mock_response.data = [mock_image_data]
        mock_client.images.generate.return_value = mock_response
        
        result_image, result_text = generate_image(
            mock_client, "dall-e-3", "A sunset"
        )
        
        self.assertIsInstance(result_image, Image.Image)
        self.assertEqual(result_text, "A beautiful sunset")
        mock_client.images.generate.assert_called_once()
    
    def test_generate_image_with_custom_size(self):
        """Test image generation with custom size."""
        mock_client = MagicMock()
        
        mock_image_data = MagicMock()
        mock_image_data.b64_json = self._create_test_image_b64()
        mock_image_data.revised_prompt = None
        
        mock_response = MagicMock()
        mock_response.data = [mock_image_data]
        mock_client.images.generate.return_value = mock_response
        
        result_image, result_text = generate_image(
            mock_client, "dall-e-3", "A landscape",
            size="1792x1024"
        )
        
        self.assertIsInstance(result_image, Image.Image)
        self.assertIsNone(result_text)
        
        # Verify size was passed
        call_kwargs = mock_client.images.generate.call_args[1]
        self.assertEqual(call_kwargs['size'], "1792x1024")
    
    def test_generate_image_raises_on_empty_data(self):
        """Test that empty data raises RuntimeError."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = []
        mock_client.images.generate.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as ctx:
            generate_image(mock_client, "dall-e-3", "Test")
        self.assertIn("空数据", str(ctx.exception))
    
    def test_generate_image_raises_on_none_response(self):
        """Test that None response raises RuntimeError."""
        mock_client = MagicMock()
        mock_client.images.generate.return_value = None
        
        with self.assertRaises(RuntimeError) as ctx:
            generate_image(mock_client, "dall-e-3", "Test")
        self.assertIn("空响应", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
