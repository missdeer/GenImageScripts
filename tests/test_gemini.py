#!/usr/bin/env python
"""Basic tests for gemini.py module."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock the google.genai module before importing gemini
import sys
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

from src.gemini import GeminiConfig, generate_text, generate_image, DEFAULT_BASE_URL, DEFAULT_LOCATION


class TestGeminiConfig(unittest.TestCase):
    """Test cases for GeminiConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GeminiConfig()
        self.assertIsNone(config.api_key)
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertFalse(config.vertex)
        self.assertIsNone(config.project)
        self.assertEqual(config.location, DEFAULT_LOCATION)
        self.assertIsNone(config.credentials)
    
    def test_api_key_mode_config(self):
        """Test API key mode configuration."""
        config = GeminiConfig(
            api_key="test-api-key",
            base_url="https://custom.api.com",
        )
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.base_url, "https://custom.api.com")
        self.assertFalse(config.vertex)
    
    def test_vertex_mode_config(self):
        """Test Vertex AI mode configuration."""
        config = GeminiConfig(
            vertex=True,
            project="test-project",
            location="us-east1",
        )
        self.assertTrue(config.vertex)
        self.assertEqual(config.project, "test-project")
        self.assertEqual(config.location, "us-east1")
    
    def test_create_client_requires_api_key(self):
        """Test that create_client raises error without API key in non-Vertex mode."""
        config = GeminiConfig()
        with self.assertRaises(ValueError) as ctx:
            config.create_client()
        self.assertIn("API key is required", str(ctx.exception))


class TestGenerateText(unittest.TestCase):
    """Test cases for generate_text function."""
    
    def test_generate_text_returns_string(self):
        """Test that generate_text returns generated text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated response text"
        mock_response.prompt_feedback = None
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response
        
        result = generate_text(mock_client, "gemini-pro", "Hello, world!")
        
        self.assertEqual(result, "Generated response text")
        mock_client.models.generate_content.assert_called_once()
    
    def test_generate_text_handles_empty_response(self):
        """Test handling of empty response text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.prompt_feedback = None
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response
        
        result = generate_text(mock_client, "gemini-pro", "Test prompt")
        
        self.assertEqual(result, "")
    
    def test_generate_text_raises_on_none_response(self):
        """Test that None response raises RuntimeError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = None
        
        with self.assertRaises(RuntimeError) as ctx:
            generate_text(mock_client, "gemini-pro", "Test")
        self.assertIn("空响应", str(ctx.exception))


class TestGenerateImage(unittest.TestCase):
    """Test cases for generate_image function."""
    
    def test_generate_image_without_reference(self):
        """Test image generation without reference images."""
        mock_client = MagicMock()
        mock_image = MagicMock()
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.as_image.return_value = mock_image
        
        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = []
        mock_response.parts = [mock_part]
        mock_client.models.generate_content.return_value = mock_response
        
        result_image, result_text = generate_image(
            mock_client, "gemini-2.0-flash-exp", "A cat"
        )
        
        self.assertEqual(result_image, mock_image)
        self.assertIsNone(result_text)
    
    def test_generate_image_with_text_response(self):
        """Test image generation that returns both image and text."""
        mock_client = MagicMock()
        mock_image = MagicMock()
        
        mock_text_part = MagicMock()
        mock_text_part.text = "Image description"
        mock_text_part.as_image.return_value = None
        
        mock_image_part = MagicMock()
        mock_image_part.text = None
        mock_image_part.as_image.return_value = mock_image
        
        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = []
        mock_response.parts = [mock_text_part, mock_image_part]
        mock_client.models.generate_content.return_value = mock_response
        
        result_image, result_text = generate_image(
            mock_client, "gemini-2.0-flash-exp", "A cat"
        )
        
        self.assertEqual(result_image, mock_image)
        self.assertEqual(result_text, "Image description")
    
    def test_generate_image_raises_on_none_response(self):
        """Test that None response raises RuntimeError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = None
        
        with self.assertRaises(RuntimeError) as ctx:
            generate_image(mock_client, "gemini-2.0-flash-exp", "Test")
        self.assertIn("空响应", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
