"""
Data Collection Module
=====================

This module handles collection of text and image data from various sources
for training the multi-modal content moderation system.

Classes:
- TextCollector: Collects text data from various APIs and datasets
- ImageCollector: Collects and preprocesses image data
- DataAugmentor: Applies data augmentation techniques
- DataValidator: Validates and cleans collected data
"""

import os
import pandas as pd
from typing import List, Dict
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCollector:
    """
    Collects text data from multiple sources including HuggingFace datasets.
    """

    def __init__(self):
        """Initialize the text collector."""
        self.datasets = []
        self.collected_data = []

    def collect_hate_speech_datasets(self) -> pd.DataFrame:
        """
        Collect hate speech datasets from HuggingFace.

        Returns:
            pd.DataFrame: Combined dataset with text and labels
        """
        try:
            logger.info("Loading hate speech datasets...")

            # Load multiple hate speech datasets
            datasets_to_load = [
                "hate_speech18",
                "hatexplain",
                "ucberkeley-dlab/measuring-hate-speech",
            ]

            combined_data = []

            for dataset_name in datasets_to_load:
                try:
                    logger.info(f"Loading {dataset_name}...")
                    dataset = load_dataset(dataset_name, split="train")

                    # Convert to pandas DataFrame safely
                    df: pd.DataFrame
                    if hasattr(dataset, 'to_pandas'):
                        df = dataset.to_pandas()  # type: ignore
                    else:
                        # Convert dataset to list of dicts first
                        data_list = []
                        for item in dataset:
                            data_list.append(item)
                        df = pd.DataFrame(data_list)
                    df["source"] = dataset_name
                    combined_data.append(df)

                except Exception as e:
                    logger.warning(f"Could not load {dataset_name}: {e}")
                    continue

            if combined_data:
                result_df = pd.concat(combined_data, ignore_index=True)
                logger.info(f"Collected {len(result_df)} text samples")
                return result_df
            else:
                logger.warning("No datasets were successfully loaded")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting datasets: {e}")
            return pd.DataFrame()

    def collect_reddit_data(
        self, subreddits: List[str], limit: int = 1000
    ) -> pd.DataFrame:
        """
        Collect Reddit data using PRAW (requires API credentials).

        Args:
            subreddits: List of subreddit names to collect from
            limit: Maximum number of posts to collect

        Returns:
            pd.DataFrame: Reddit posts and comments
        """
        try:
            # This would require PRAW setup with Reddit API credentials
            # For now, return empty DataFrame as placeholder
            logger.info("Reddit data collection would require API setup")
            return pd.DataFrame(columns=["text", "label", "source"])

        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            return pd.DataFrame()

    def create_synthetic_data(
        self, base_data: pd.DataFrame, augmentation_factor: int = 2
    ) -> pd.DataFrame:
        """
        Create synthetic data using text augmentation techniques.

        Args:
            base_data: Original dataset to augment
            augmentation_factor: How many synthetic samples per original

        Returns:
            pd.DataFrame: Augmented dataset
        """
        try:
            logger.info("Creating synthetic data...")

            # Simple text augmentation techniques
            augmented_data = []

            for _, row in base_data.iterrows():
                text = row.get("text", "")
                if isinstance(text, str) and text.strip():
                    # Original text
                    augmented_data.append(row.to_dict())

                    # Simple augmentations
                    for i in range(augmentation_factor):
                        augmented_row = row.to_dict().copy()
                        # Random word replacement/synonyms (simplified)
                        words = text.split()
                        if len(words) > 3:
                            # Simple word shuffling as augmentation
                            import random

                            random.shuffle(words)
                            augmented_row["text"] = " ".join(words)
                            augmented_row["is_synthetic"] = True
                            augmented_data.append(augmented_row)

            result_df = pd.DataFrame(augmented_data)
            logger.info(f"Created {len(result_df)} augmented samples")
            return result_df

        except Exception as e:
            logger.error(f"Error creating synthetic data: {e}")
            return base_data


class ImageCollector:
    """Collects and preprocesses image data from various sources."""

    def __init__(self, data_dir: str = "data/images"):
        """
        Initialize the image collector.

        Args:
            data_dir: Directory to store collected images
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def collect_hateful_memes_dataset(self) -> Dict[str, List]:
        """
        Collect the Hateful Memes dataset from Facebook AI.

        Returns:
            Dict containing image paths and labels
        """
        try:
            logger.info("Loading Hateful Memes dataset...")

            # This would download the actual dataset
            # For now, create placeholder structure
            dataset_info = {
                "image_paths": [],
                "labels": [],
                "text": [],
                "multimodal_labels": [],
            }

            logger.info(
                "Hateful Memes dataset collection requires manual download"
            )
            return dataset_info

        except Exception as e:
            logger.error(f"Error collecting Hateful Memes: {e}")
            return {
                "image_paths": [],
                "labels": [],
                "text": [],
                "multimodal_labels": [],
            }

    def collect_safe_images(self, num_images: int = 5000) -> List[str]:
        """
        Collect safe images from Unsplash API.

        Args:
            num_images: Number of safe images to collect

        Returns:
            List of image file paths
        """
        try:
            logger.info(f"Collecting {num_images} safe images...")

            # This would use Unsplash API with proper credentials
            # For now, return placeholder
            image_paths = []

            logger.info("Unsplash API collection requires API key setup")
            return image_paths

        except Exception as e:
            logger.error(f"Error collecting safe images: {e}")
            return []


class DataValidator:
    """Validates and cleans collected data."""

    @staticmethod
    def validate_text_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean text data.

        Args:
            df: DataFrame with text data

        Returns:
            pd.DataFrame: Cleaned and validated data
        """
        try:
            logger.info("Validating text data...")

            # Remove empty or null text
            df = df.dropna(subset=["text"])
            df = df[df["text"].str.strip() != ""]

            # Remove duplicates
            df = df.drop_duplicates(subset=["text"])

            # Basic text length filtering
            df = df[df["text"].str.len() > 10]  # At least 10 characters
            df = df[df["text"].str.len() < 5000]  # Less than 5000 characters

            logger.info(f"Validated data: {len(df)} samples remaining")
            return df

        except Exception as e:
            logger.error(f"Error validating text data: {e}")
            return df

    @staticmethod
    def validate_image_data(image_paths: List[str]) -> List[str]:
        """
        Validate image files.

        Args:
            image_paths: List of image file paths

        Returns:
            List of valid image paths
        """
        try:
            from PIL import Image

            valid_paths = []

            for path in image_paths:
                try:
                    with Image.open(path) as img:
                        # Basic validation
                        if (
                            img.size[0] > 32 and img.size[1] > 32
                        ):  # Minimum size
                            valid_paths.append(path)
                except Exception:
                    continue

            logger.info(f"Validated {len(valid_paths)} valid images")
            return valid_paths

        except Exception as e:
            logger.error(f"Error validating images: {e}")
            return image_paths


def main():
    """Main function to run data collection."""
    logger.info("Starting data collection process...")

    # Initialize collectors
    text_collector = TextCollector()
    image_collector = ImageCollector()

    # Collect text data
    hate_speech_data = text_collector.collect_hate_speech_datasets()

    if not hate_speech_data.empty:
        # Validate and clean data
        cleaned_data = DataValidator.validate_text_data(hate_speech_data)

        # Create augmented data
        augmented_data = text_collector.create_synthetic_data(cleaned_data)

        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        augmented_data.to_csv("data/processed/text_data.csv", index=False)
        logger.info(
            f"Saved {len(augmented_data)} text samples to "
            f"data/processed/text_data.csv"
        )

    # Collect image data
    _ = image_collector.collect_hateful_memes_dataset()
    _ = image_collector.collect_safe_images()

    logger.info("Data collection process completed!")


if __name__ == "__main__":
    main()
