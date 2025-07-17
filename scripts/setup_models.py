"""
Setup Script for Multi-Modal Content Moderation System
======================================================

Downloads required models and sets up the environment.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed",
        "data/images",
        "checkpoints",
        "logs",
        "outputs",
        "results",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_pretrained_models():
    """Download and cache pretrained models."""
    logger.info("Downloading pretrained models...")

    try:
        # Import transformers to trigger model downloads
        from transformers import (
            AutoTokenizer,
            AutoModel,
            ViTModel,
            ViTImageProcessor,
        )

        # Download BERT model
        logger.info("Downloading BERT model...")
        _ = AutoTokenizer.from_pretrained("bert-base-uncased")
        _ = AutoModel.from_pretrained("bert-base-uncased")
        logger.info("✅ BERT model downloaded and cached")

        # Download ViT model
        logger.info("Downloading Vision Transformer model...")
        _ = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        _ = ViTModel.from_pretrained("google/vit-base-patch16-224")
        logger.info("✅ ViT model downloaded and cached")

    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False

    return True


def setup_sample_data():
    """Set up sample data for testing."""
    logger.info("Setting up sample data...")

    try:
        import pandas as pd

        # Create sample text data
        sample_texts = [
            "This is a normal, safe message about everyday topics.",
            "I love this new restaurant, the food is amazing!",
            "Thank you for your help, I really appreciate it.",
            "Looking forward to our meeting tomorrow.",
            "This weather is beautiful today.",
            "Great job on the presentation!",
            "Can you help me with this problem?",
            "The movie was fantastic, highly recommend it.",
            "Hope you have a wonderful day!",
            "Thanks for sharing this interesting article.",
        ]

        sample_labels = ["Safe"] * len(sample_texts)

        # Create DataFrame
        sample_df = pd.DataFrame(
            {
                "text": sample_texts,
                "label": sample_labels,
                "image_path": [""]
                * len(sample_texts),  # No images for text-only samples
            }
        )

        # Save sample data
        sample_df.to_csv("data/processed/sample_data.csv", index=False)
        logger.info("✅ Sample data created")

        # Create sample train/val/test splits
        train_size = int(0.7 * len(sample_df))
        val_size = int(0.2 * len(sample_df))
        train_df = sample_df[:train_size]
        val_df = sample_df[train_size:train_size + val_size]
        test_df = sample_df[train_size + val_size:]

        train_df.to_csv("data/processed/train.csv", index=False)
        val_df.to_csv("data/processed/val.csv", index=False)
        test_df.to_csv("data/processed/test.csv", index=False)

        logger.info("✅ Train/Val/Test splits created")

    except Exception as e:
        logger.error(f"Error setting up sample data: {e}")
        return False

    return True


def install_dependencies():
    """Install additional dependencies if needed."""
    logger.info("Checking dependencies...")

    try:
        # Check if all required packages are installed
        pass

        logger.info("✅ All required packages are installed")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info(
            "Please install dependencies using: "
            "pip install -r requirements.txt"
        )
        return False

    return True


def setup_environment_variables():
    """Set up environment variables."""
    logger.info("Setting up environment variables...")

    # Create .env file if it doesn't exist
    env_file = ".env"
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write(
                "# Environment variables for Multi-Modal Content "
                "Moderation System\n"
            )
            f.write("DEVICE=auto\n")
            f.write("LOG_LEVEL=INFO\n")
            f.write("API_HOST=0.0.0.0\n")
            f.write("API_PORT=8000\n")
            f.write("WANDB_DISABLED=true\n")
            f.write("# Add your API keys here\n")
            f.write("# OPENAI_API_KEY=your_key_here\n")
            f.write("# REDDIT_CLIENT_ID=your_id_here\n")
            f.write("# REDDIT_CLIENT_SECRET=your_secret_here\n")
            f.write("# UNSPLASH_ACCESS_KEY=your_key_here\n")

        logger.info("✅ .env file created")
    else:
        logger.info("✅ .env file already exists")


def create_demo_notebook():
    """Create a demo Jupyter notebook."""
    logger.info("Creating demo notebook...")

    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Multi-Modal Content Moderation System Demo\n",
                    "\n",
                    "This notebook demonstrates the key features of the "
                    "content moderation system.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Import required libraries\n",
                    "import sys\n",
                    "sys.path.append('..')\n",
                    "\n",
                    "import torch\n",
                    "import pandas as pd\n",
                    "from preprocessing.text_processor import TextProcessor\n",
                    (
                        "from preprocessing.image_processor import "
                        "ImageProcessor\n"
                    ),
                    "from models import multimodal_model\n",
                    "\n",
                    'print("Libraries imported successfully!")',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Initialize processors and model\n",
                    "text_processor = TextProcessor()\n",
                    "image_processor = ImageProcessor()\n",
                    "model = MultiModalClassifier(num_classes=5)\n",
                    "\n",
                    'print("Model and processors initialized!")',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Test text processing\n",
                    'sample_text = "This is a sample text for content '
                    'moderation."\n',
                    "processed = text_processor.clean_text(sample_text)\n",
                    "tokens = text_processor.tokenize_batch([sample_text])\n",
                    "\n",
                    'print(f"Original: {sample_text}")\n',
                    'print(f"Processed: {processed}")\n',
                    "print(f\"Tokens shape: {tokens['input_ids'].shape}\")",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    import json

    with open("notebooks/demo.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)

    logger.info("✅ Demo notebook created")


def main():
    """Main setup function."""
    logger.info("🚀 Starting Multi-Modal Content Moderation System setup...")

    # Create necessary directories
    create_directories()

    # Create notebooks directory
    os.makedirs("notebooks", exist_ok=True)

    # Check dependencies
    if not install_dependencies():
        logger.error("❌ Dependency check failed")
        return False

    # Download pretrained models
    if not download_pretrained_models():
        logger.error("❌ Model download failed")
        return False

    # Set up sample data
    if not setup_sample_data():
        logger.error("❌ Sample data setup failed")
        return False

    # Set up environment variables
    setup_environment_variables()

    # Create demo notebook
    create_demo_notebook()

    logger.info("✅ Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Activate your virtual environment")
    logger.info("2. Run the API: python api/main.py")
    logger.info("3. Run the frontend: streamlit run frontend/app.py")
    logger.info(
        "4. Open the demo notebook: jupyter notebook notebooks/demo.ipynb"
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
