"""
Image Processing Module
======================

Handles image preprocessing and feature extraction.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing and feature extraction."""

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initialize the image processor.

        Args:
            model_name: Name of the pretrained vision model to use
        """
        self.model_name = model_name

        try:
            # Initialize Vision Transformer
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name)
            logger.info(f"Loaded ViT model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading ViT model: {e}")
            self.processor = None
            self.model = None

        # Standard image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_image(
        self, image_input: Union[str, Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image_input: Either image path or PIL Image object

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image if path is provided
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            else:
                image = image_input.convert("RGB")

            # Apply transformations
            processed_image = self.transform(image)

            # Ensure we return a tensor
            if not isinstance(processed_image, torch.Tensor):
                processed_image = torch.tensor(processed_image)

            return processed_image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return torch.zeros(3, 224, 224)

    def preprocess_batch(
        self, image_inputs: List[Union[str, Image.Image]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            image_inputs: List of image paths or PIL Image objects

        Returns:
            torch.Tensor: Batch of preprocessed images
        """
        try:
            processed_images = []

            for image_input in image_inputs:
                processed_img = self.preprocess_image(image_input)
                processed_images.append(processed_img)

            # Stack into batch
            batch_tensor = torch.stack(processed_images)
            return batch_tensor

        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            return torch.zeros(len(image_inputs), 3, 224, 224)

    def extract_features(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract features using Vision Transformer.

        Args:
            image_batch: Batch of preprocessed images

        Returns:
            torch.Tensor: Feature vectors
        """
        if self.model is None:
            logger.error("ViT model not initialized")
            return torch.empty(0)

        try:
            with torch.no_grad():
                # ViT expects pixel values
                if self.processor:
                    # Convert tensor back to PIL format for processor
                    images = []
                    for img_tensor in image_batch:
                        # Denormalize
                        img_array = img_tensor.permute(1, 2, 0).numpy()
                        std_array = np.array([0.229, 0.224, 0.225])
                        mean_array = np.array([0.485, 0.456, 0.406])
                        img_array = img_array * std_array + mean_array
                        img_array = np.clip(img_array * 255, 0, 255)
                        img_array = img_array.astype(np.uint8)
                        images.append(Image.fromarray(img_array))

                    # Process with ViT processor
                    inputs = self.processor(images, return_tensors="pt")
                    outputs = self.model(**inputs)

                    # Use [CLS] token embedding
                    features = outputs.last_hidden_state[:, 0, :]
                else:
                    # Fallback: direct forward pass
                    outputs = self.model(pixel_values=image_batch)
                    features = outputs.last_hidden_state[:, 0, :]

                return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return torch.empty(0)

    def extract_features_resnet(
        self, image_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Alternative feature extraction using ResNet.

        Args:
            image_batch: Batch of preprocessed images

        Returns:
            torch.Tensor: Feature vectors
        """
        try:
            # Load pretrained ResNet
            resnet = resnet50(pretrained=True)
            resnet.eval()

            # Remove final classification layer
            children = list(resnet.children())[:-1]
            feature_extractor = torch.nn.Sequential(*children)

            with torch.no_grad():
                features = feature_extractor(image_batch)
                features = features.squeeze()

                # Handle single image case
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)

                return features

        except Exception as e:
            logger.error(f"Error extracting ResNet features: {e}")
            return torch.empty(0)

    def validate_images(self, image_paths: List[str]) -> List[str]:
        """
        Validate that images can be loaded and processed.

        Args:
            image_paths: List of image file paths

        Returns:
            List of valid image paths
        """
        valid_paths = []

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    # Convert to RGB to ensure compatibility
                    img = img.convert("RGB")

                    # Check minimum size
                    if img.size[0] >= 32 and img.size[1] >= 32:
                        valid_paths.append(path)

            except Exception as e:
                logger.warning(f"Invalid image {path}: {e}")
                continue

        logger.info(f"Validated {len(valid_paths)}/{len(image_paths)} images")
        return valid_paths


def main():
    """Test the image processor."""
    processor = ImageProcessor()

    # Create a dummy image for testing
    dummy_image = Image.new("RGB", (224, 224), color="red")

    # Test preprocessing
    processed = processor.preprocess_image(dummy_image)
    print("Processed image shape:", processed.shape)

    # Test batch preprocessing
    batch = processor.preprocess_batch([dummy_image, dummy_image])
    print("Batch shape:", batch.shape)

    # Test feature extraction
    features = processor.extract_features(batch)
    print("ViT features shape:", features.shape)

    # Test ResNet features
    resnet_features = processor.extract_features_resnet(batch)
    print("ResNet features shape:", resnet_features.shape)


if __name__ == "__main__":
    main()
