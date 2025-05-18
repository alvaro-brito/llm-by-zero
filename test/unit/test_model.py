import unittest
import json
import requests
from unittest.mock import patch, MagicMock

class TestModel(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:8000/api/v1"
        
    @patch('requests.post')
    def test_model_creation(self, mock_post):
        # Mock response for model creation
        mock_response = MagicMock()
        mock_response.json.return_value = {"model_id": "test_model_123"}
        mock_post.return_value = mock_response
        
        # Test model creation
        response = requests.post(f"{self.base_url}/models/create")
        self.assertEqual(response.json()["model_id"], "test_model_123")
    
    @patch('requests.get')
    def test_model_status(self, mock_get):
        # Mock response for model status
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ready"}
        mock_get.return_value = mock_response
        
        # Test model status
        response = requests.get(f"{self.base_url}/models/test_model_123/status")
        self.assertEqual(response.json()["status"], "ready")
    
    @patch('requests.post')
    def test_model_inference(self, mock_post):
        # Mock response for inference
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "generated_text": "Test response",
            "confidence": 0.95
        }
        mock_post.return_value = mock_response
        
        # Test inference
        test_input = {"text": "Test input"}
        response = requests.post(
            f"{self.base_url}/models/test_model_123/inference",
            json=test_input
        )
        self.assertIn("generated_text", response.json())
        self.assertIn("confidence", response.json())

if __name__ == '__main__':
    unittest.main() 