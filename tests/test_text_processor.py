"""
Unit Tests for Text Processing
==============================
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_processor import TextProcessor  # noqa: E402
import torch  # noqa: E402


class TestTextProcessor(unittest.TestCase):
    """Test cases for TextProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = TextProcessor()
        self.sample_texts = [
            "This is a normal text message.",
            "Check out this link: https://example.com @user #hashtag",
            "Text with special chars!@#$%^&*()",
            "",  # Empty text
            "   ",  # Whitespace only
        ]

    def test_clean_text(self):
        """Test text cleaning functionality."""
        test_cases = [
            ("Hello world!", "hello world!"),
            ("Check https://example.com", "check"),
            ("@user mentioned #hashtag", "mentioned"),
            ("Multiple    spaces", "multiple spaces"),
            ("", ""),
            ("   ", ""),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.processor.clean_text(input_text)
                self.assertEqual(result, expected)

    def test_tokenize_batch(self):
        """Test batch tokenization."""
        tokens = self.processor.tokenize_batch(self.sample_texts[:3])

        # Check if tokens are returned
        self.assertIn("input_ids", tokens)
        self.assertIn("attention_mask", tokens)

        # Check tensor shapes
        self.assertEqual(tokens["input_ids"].shape[0], 3)  # Batch size
        self.assertEqual(tokens["attention_mask"].shape[0], 3)

        # Check if tensors are of correct type
        self.assertIsInstance(tokens["input_ids"], torch.Tensor)
        self.assertIsInstance(tokens["attention_mask"], torch.Tensor)

    def test_create_embeddings(self):
        """Test embedding creation."""
        if self.processor.model is not None:
            embeddings = self.processor.create_embeddings(
                self.sample_texts[:2]
            )

            # Check embedding shape
            self.assertEqual(embeddings.shape[0], 2)  # Batch size
            self.assertEqual(embeddings.shape[1], 768)  # BERT hidden size

            # Check if embeddings are different for different texts
            self.assertFalse(torch.equal(embeddings[0], embeddings[1]))

    def test_preprocess_dataframe(self):
        """Test DataFrame preprocessing."""
        import pandas as pd

        # Create test DataFrame
        test_df = pd.DataFrame(
            {
                "text": self.sample_texts,
                "label": ["safe"] * len(self.sample_texts),
            }
        )

        processed_df = self.processor.preprocess_dataframe(test_df)

        # Check if cleaned text column is added
        self.assertIn("cleaned_text", processed_df.columns)
        self.assertIn("text_length", processed_df.columns)
        self.assertIn("word_count", processed_df.columns)

        # Check if empty texts are removed
        self.assertTrue(
            all(len(text) > 0 for text in processed_df["cleaned_text"])
        )


class TestTextProcessorIntegration(unittest.TestCase):
    """Integration tests for TextProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = TextProcessor()

    def test_end_to_end_processing(self):
        """Test complete text processing pipeline."""
        sample_text = (
            "This is a test message with URL https://test.com and @mention"
        )

        # Clean text
        cleaned = self.processor.clean_text(sample_text)

        # Tokenize
        tokens = self.processor.tokenize_batch([cleaned])

        # Create embeddings
        if self.processor.model is not None:
            embeddings = self.processor.create_embeddings([cleaned])

            # Verify the pipeline worked
            self.assertGreater(len(cleaned), 0)
            self.assertIn("input_ids", tokens)
            self.assertEqual(embeddings.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
