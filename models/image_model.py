"""
Image Classification Model
=========================

Vision Transformer-based image classifier for content moderation.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision.models import resnet50, efficientnet_b0
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ImageClassifier(nn.Module):
    """Vision Transformer-based image classifier."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 5,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        """
        Initialize the image classifier.

        Args:
            model_name: Name of the pretrained ViT model
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone parameters
        """
        super(ImageClassifier, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        try:
            # Load Vision Transformer
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel.from_pretrained(model_name, config=config)

            # Freeze backbone if specified
            if freeze_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False

            # Classification head
            self.dropout = nn.Dropout(dropout_rate)

            # Multi-layer classification head for better performance
            hidden_size = config.hidden_size
            self.intermediate = nn.Linear(hidden_size, hidden_size // 2)
            self.activation = nn.ReLU()
            self.layer_norm = nn.LayerNorm(hidden_size // 2)
            self.final_classifier = nn.Linear(hidden_size // 2, num_classes)

            # Simple classifier for comparison
            self.simple_classifier = nn.Linear(hidden_size, num_classes)

            logger.info(f"Initialized ImageClassifier with {model_name}")

        except Exception as e:
            logger.error(f"Error initializing ImageClassifier: {e}")
            raise

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            pixel_values: Input image tensor

        Returns:
            Dict containing logits and features
        """
        try:
            # ViT forward pass
            outputs = self.vit(pixel_values=pixel_values)

            # Use [CLS] token representation
            pooled_output = outputs.last_hidden_state[:, 0, :]

            # Apply dropout
            features = self.dropout(pooled_output)

            # Multi-layer classification
            intermediate = self.activation(self.intermediate(features))
            intermediate = self.layer_norm(intermediate)
            intermediate = self.dropout(intermediate)
            logits = self.final_classifier(intermediate)

            # Simple classification
            simple_logits = self.simple_classifier(features)

            return {
                "logits": logits,
                "simple_logits": simple_logits,
                "features": pooled_output,
                "intermediate_features": intermediate,
            }

        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            batch_size = pixel_values.size(0)
            return {
                "logits": torch.zeros(batch_size, self.num_classes),
                "simple_logits": torch.zeros(batch_size, self.num_classes),
                "features": torch.zeros(batch_size, 768),
                "intermediate_features": torch.zeros(batch_size, 384),
            }

    def get_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get image embeddings without classification.

        Args:
            pixel_values: Input image tensor

        Returns:
            torch.Tensor: Image embeddings
        """
        try:
            with torch.no_grad():
                outputs = self.vit(pixel_values=pixel_values)
                embeddings = outputs.last_hidden_state[:, 0, :]
                return embeddings

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return torch.zeros(pixel_values.size(0), 768)


class ResNetImageClassifier(nn.Module):
    """ResNet-based image classifier as alternative."""

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize ResNet-based classifier.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
        """
        super(ResNetImageClassifier, self).__init__()

        # Load ResNet50
        self.backbone = resnet50(pretrained=pretrained)

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Custom classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet."""
        # Extract features
        features = self.backbone(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        # Classify
        features = self.dropout(features)
        logits = self.classifier(features)

        return logits


class EfficientNetImageClassifier(nn.Module):
    """EfficientNet-based image classifier."""

    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.3):
        """
        Initialize EfficientNet-based classifier.

        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(EfficientNetImageClassifier, self).__init__()

        # Load EfficientNet
        self.backbone = efficientnet_b0(pretrained=True)

        # Replace classifier
        # EfficientNet's classifier is usually Sequential(Dropout, Linear)
        # Find the Linear layer and get its in_features
        if isinstance(self.backbone.classifier, nn.Sequential):
            for layer in self.backbone.classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
            else:
                raise ValueError(
                    "No Linear layer found in EfficientNet classifier."
                )

        elif isinstance(self.backbone.classifier, nn.Linear):
            in_features = self.backbone.classifier.in_features
        else:
            raise ValueError("Unknown classifier type in EfficientNet.")

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EfficientNet."""
        return self.backbone(x)


class AttentionImageClassifier(nn.Module):
    """Image classifier with spatial attention mechanism."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 5,
        num_attention_heads: int = 8,
    ):
        """
        Initialize attention-based image classifier.

        Args:
            model_name: Name of the pretrained ViT model
            num_classes: Number of output classes
            num_attention_heads: Number of attention heads
        """
        super(AttentionImageClassifier, self).__init__()

        self.vit = ViTModel.from_pretrained(model_name)
        config = self.vit.config

        # Spatial attention mechanism
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        # Classification layers
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass with spatial attention."""
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state

        # Apply spatial attention
        attended_output, attention_weights = self.spatial_attention(
            sequence_output, sequence_output, sequence_output
        )

        # Use [CLS] token
        cls_output = attended_output[:, 0, :]

        # Classification
        cls_output = self.layer_norm(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits


def create_image_model(model_type: str = "vit", **kwargs) -> nn.Module:
    """
    Factory function to create image models.

    Args:
        model_type: Type of model ('vit',
            'resnet',
                'efficientnet',
                'attention')
        **kwargs: Additional arguments for model initialization

    Returns:
        nn.Module: Image classification model
    """
    if model_type == "vit":
        return ImageClassifier(**kwargs)
    elif model_type == "resnet":
        return ResNetImageClassifier(**kwargs)
    elif model_type == "efficientnet":
        return EfficientNetImageClassifier(**kwargs)
    elif model_type == "attention":
        return AttentionImageClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Test the image models."""
    # Test ViT model
    model = ImageClassifier(num_classes=5)

    # Create dummy input
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    outputs = model(pixel_values)
    print("ViT model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test ResNet model
    resnet_model = ResNetImageClassifier(num_classes=5)
    resnet_outputs = resnet_model(pixel_values)
    print(f"ResNet model output shape: {resnet_outputs.shape}")

    # Test EfficientNet model
    efficientnet_model = EfficientNetImageClassifier(num_classes=5)
    efficientnet_outputs = efficientnet_model(pixel_values)
    print(f"EfficientNet model output shape: {efficientnet_outputs.shape}")


if __name__ == "__main__":
    main()
