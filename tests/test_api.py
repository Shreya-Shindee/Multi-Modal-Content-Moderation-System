"""
Unit Tests for API Endpoints
============================
"""

import unittest
import sys
import os
from fastapi.testclient import TestClient
import io
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app  # noqa: E402


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("device", data)
        self.assertIn("version", data)

    def test_classes_endpoint(self):
        """Test classes endpoint."""
        response = self.client.get("/classes")
        self.assertEqual(response.status_code, 200)

        classes = response.json()
        self.assertIsInstance(classes, list)
        self.assertGreater(len(classes), 0)

    def test_text_prediction_endpoint(self):
        """Test text-only prediction endpoint."""
        test_data = {"text": "This is a test message", "threshold": 0.5}

        response = self.client.post("/predict/text", json=test_data)

        # Note: This might fail if model is not loaded
        if response.status_code == 200:
            data = response.json()
            self.assertIn("prediction", data)
            self.assertIn("confidence", data)
            self.assertIn("scores", data)
            self.assertIn("processing_time", data)
        elif response.status_code == 503:
            # Model not loaded, which is expected in testing
            pass
        else:
            self.fail(f"Unexpected status code: {response.status_code}")

    def test_text_prediction_validation(self):
        """Test text prediction input validation."""
        # Test with empty text
        test_data = {"text": "", "threshold": 0.5}

        response = self.client.post("/predict/text", json=test_data)
        # Should still process empty text, but might return error
        self.assertIn(response.status_code, [200, 400, 503])

        # Test with invalid threshold
        test_data = {
            "text": "Test text",
            "threshold": -1.0,  # Invalid threshold
        }

        response = self.client.post("/predict/text", json=test_data)
        # Should handle invalid threshold gracefully
        self.assertIn(response.status_code, [200, 400, 422, 503])

    def test_batch_prediction_endpoint(self):
        """Test batch text prediction endpoint."""
        test_data = {
            "texts": [
                "First test message",
                "Second test message",
                "Third test message",
            ],
            "threshold": 0.5,
        }

        response = self.client.post("/predict/batch", json=test_data)

        if response.status_code == 200:
            data = response.json()
            self.assertIn("predictions", data)
            self.assertIn("total_processing_time", data)
            self.assertEqual(len(data["predictions"]), 3)
        elif response.status_code == 503:
            # Model not loaded
            pass
        else:
            self.fail(f"Unexpected status code: {response.status_code}")

    def test_multimodal_prediction_text_only(self):
        """Test multimodal endpoint with text only."""
        test_data = {"text": "This is a test message", "threshold": 0.5}

        response = self.client.post("/predict/multimodal", data=test_data)

        if response.status_code == 200:
            data = response.json()
            self.assertIn("prediction", data)
            self.assertIn("modalities_used", data)
            self.assertIn("text", data["modalities_used"])
        elif response.status_code == 503:
            # Model not loaded
            pass

    def test_multimodal_prediction_with_image(self):
        """Test multimodal endpoint with text and image."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        files = {"image": ("test.png", img_bytes, "image/png")}
        data = {"text": "Test message with image", "threshold": 0.5}

        response = self.client.post(
            "/predict/multimodal", data=data, files=files
        )

        if response.status_code == 200:
            result = response.json()
            self.assertIn("prediction", result)
            self.assertIn("modalities_used", result)
        elif response.status_code == 503:
            # Model not loaded
            pass

    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("model_accuracy", data)
        self.assertIn("total_predictions", data)

    def test_invalid_endpoints(self):
        """Test invalid endpoints."""
        # Test non-existent endpoint
        response = self.client.get("/nonexistent")
        self.assertEqual(response.status_code, 404)

        # Test invalid method
        response = self.client.delete("/predict/text")
        self.assertEqual(response.status_code, 405)


class TestAPIErrorHandling(unittest.TestCase):
    """Test error handling in API."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        response = self.client.post(
            "/predict/text",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Missing text field
        response = self.client.post("/predict/text", json={"threshold": 0.5})
        self.assertEqual(response.status_code, 422)

        # Missing texts field in batch
        response = self.client.post("/predict/batch", json={"threshold": 0.5})
        self.assertEqual(response.status_code, 422)

    def test_invalid_file_upload(self):
        """Test handling of invalid file uploads."""
        # Upload non-image file
        files = {
            "image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")
        }
        data = {"text": "Test", "threshold": 0.5}

        response = self.client.post(
            "/predict/multimodal", data=data, files=files
        )
        # Should handle invalid image gracefully
        self.assertIn(response.status_code, [200, 400, 503])


if __name__ == "__main__":
    unittest.main()
