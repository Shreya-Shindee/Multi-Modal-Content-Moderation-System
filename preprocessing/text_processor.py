"""
Text Processing Module
=====================

Handles text preprocessing for the content moderation system.
"""

import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text preprocessing and feature extraction."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the text processor.

        Args:
            model_name: Name of the pretrained model to use
        """
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self.tokenizer = None
            self.model = None

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.

        Args:
            text: Raw text to clean

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove user mentions and hashtags
        text = re.sub(r"@\w+|#\w+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s!?.,]", "", text)

        # Convert to lowercase
        text = text.lower().strip()

        return text

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text (alias for clean_text).

        Args:
            text: Input text string

        Returns:
            Cleaned and preprocessed text
        """
        return self.clean_text(text)

    def tokenize_batch(
        self, texts: List[str], max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of texts to tokenize
            max_length: Maximum sequence length

        Returns:
            Dict containing tokenized inputs
        """
        if self.tokenizer is None:
            logger.error("Tokenizer not initialized")
            return {}

        try:
            # Clean texts first
            cleaned_texts = [self.clean_text(text) for text in texts]

            # Tokenize
            encoded = self.tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

        except Exception as e:
            logger.error(f"Error tokenizing texts: {e}")
            return {}

    def create_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Generate BERT embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            torch.Tensor: Text embeddings
        """
        if self.model is None:
            logger.error("Model not initialized")
            return torch.empty(0)

        try:
            # Tokenize texts
            inputs = self.tokenize_batch(texts)

            if not inputs:
                return torch.empty(0)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]

            return embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return torch.empty(0)

    def preprocess_dataframe(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Preprocess a DataFrame containing text data.

        Args:
            df: DataFrame with text data
            text_column: Name of the text column

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        try:
            logger.info(f"Preprocessing {len(df)} text samples...")

            # Clean text column
            df["cleaned_text"] = df[text_column].apply(self.clean_text)

            # Remove empty texts after cleaning
            df = df[df["cleaned_text"].str.len() > 0]

            # Add text statistics
            df["text_length"] = df["cleaned_text"].str.len()
            df["word_count"] = df["cleaned_text"].str.split().str.len()

            logger.info(
                f"Preprocessing completed: {len(df)} samples remaining"
            )
            return df

        except Exception as e:
            logger.error(f"Error preprocessing DataFrame: {e}")
            return df


def main():
    """Test the text processor."""
    processor = TextProcessor()

    # Test with sample texts
    sample_texts = [
        "This is a normal message.",
        "Check out this link: https://example.com @user #hashtag",
        "Some text with special chars!@#$%^&*()",
    ]

    # Test cleaning
    cleaned = [processor.clean_text(text) for text in sample_texts]
    print("Cleaned texts:", cleaned)

    # Test tokenization
    tokens = processor.tokenize_batch(sample_texts)
    print("Tokenization shape:", {k: v.shape for k, v in tokens.items()})

    # Test embeddings
    embeddings = processor.create_embeddings(sample_texts)
    print("Embeddings shape:", embeddings.shape)


if __name__ == "__main__":
    main()
