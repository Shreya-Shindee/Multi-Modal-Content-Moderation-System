#!/usr/bin/env python3
"""
Setup Script for Multi-Modal Content Moderation System
======================================================

This script sets up the environment and downloads required models.
"""

import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_requirements():
    """Install required Python packages."""
    logger.info("Installing required packages...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        logger.info("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    logger.info("Creating necessary directories...")

    directories = [
        "data/raw",
        "data/processed",
        "data/images",
        "checkpoints",
        "logs",
        "outputs",
        "results"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")


def download_models():
    """Download and cache required models."""
    logger.info("Downloading and caching models...")

    try:
        # Import here to avoid issues if packages aren't installed yet
        from transformers import (
            AutoTokenizer, AutoModel,
            ViTImageProcessor, ViTModel
        )

        # Download BERT model
        logger.info("Downloading BERT model...")
        AutoTokenizer.from_pretrained("bert-base-uncased")
        AutoModel.from_pretrained("bert-base-uncased")
        logger.info("✅ BERT model downloaded")

        # Download ViT model
        logger.info("Downloading Vision Transformer model...")
        ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        ViTModel.from_pretrained("google/vit-base-patch16-224")
        logger.info("✅ ViT model downloaded")

    except Exception as e:
        logger.warning(f"⚠️ Could not download models: {e}")
        logger.info("Models will be downloaded when first used.")


def setup_environment():
    """Set up environment variables and configurations."""
    logger.info("Setting up environment...")

    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Multi-Modal Content Moderation Environment
# Add your API keys and configuration here

# Weights & Biases (optional)
# WANDB_API_KEY=your_wandb_key_here

# Model Configuration
MODEL_CACHE_DIR=./model_cache
DATA_DIR=./data
CHECKPOINTS_DIR=./checkpoints

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
"""
        env_file.write_text(env_content)
        logger.info("✅ Created .env file")


def verify_setup():
    """Verify that the setup was successful."""
    logger.info("Verifying setup...")

    try:
        # Test imports
        import torch
        import transformers
        import pandas as pd  # noqa: F401
        import numpy as np  # noqa: F401

        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ Transformers version: {transformers.__version__}")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")

        # Test custom modules
        from preprocessing.text_processor import TextProcessor
        from preprocessing.image_processor import ImageProcessor  # noqa
        from models.multimodal_model import MultiModalClassifier  # noqa

        logger.info("✅ All custom modules import successfully")

        # Quick functionality test
        text_processor = TextProcessor()
        sample_tokens = text_processor.tokenize_batch(["Test text"])
        logger.info(
            f"✅ Text processing works: {sample_tokens['input_ids'].shape}"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Setup verification failed: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("🚀 Starting Multi-Modal Content Moderation System Setup")
    logger.info("=" * 60)

    # Step 1: Install requirements
    install_requirements()

    # Step 2: Create directories
    create_directories()

    # Step 3: Setup environment
    setup_environment()

    # Step 4: Download models
    download_models()

    # Step 5: Verify setup
    if verify_setup():
        logger.info("=" * 60)
        logger.info("🎉 Setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info(
            "1. Run demo notebook: jupyter notebook notebooks/demo.ipynb"
        )
        logger.info("2. Start API server: python api/main.py")
        logger.info("3. Run Streamlit demo: streamlit run frontend/app.py")
        logger.info("=" * 60)
    else:
        logger.error("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
