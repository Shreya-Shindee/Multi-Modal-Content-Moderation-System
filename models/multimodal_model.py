"""
Multi-Modal Fusion Model
========================

Combines text and image models for multi-modal content classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .text_model import TextClassifier
from .image_model import ImageClassifier
import logging

logger = logging.getLogger(__name__)


class MultiModalClassifier(nn.Module):
    """Multi-modal classifier combining text and image analysis."""

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 5,
        fusion_strategy: str = "attention",
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the multi-modal classifier.

        Args:
            text_model_name: Name of the text model
            image_model_name: Name of the image model
            num_classes: Number of output classes
            fusion_strategy: How to fuse modalities ('concat',
                'attention',
                    'cross_attention')
            dropout_rate: Dropout rate for fusion layers
        """
        super(MultiModalClassifier, self).__init__()

        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy

        # Initialize individual models
        self.text_model = TextClassifier(
            model_name=text_model_name,
            num_classes=num_classes,
            dropout=dropout_rate,
        )

        self.image_model = ImageClassifier(
            model_name=image_model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        # Get feature dimensions
        text_config = self.text_model.bert.config
        image_config = self.image_model.vit.config

        self.text_dim = text_config.hidden_size
        self.image_dim = image_config.hidden_size

        # Fusion layers based on strategy
        self._build_fusion_layers()

        logger.info(
            f"Initialized MultiModalClassifier with {fusion_strategy} fusion"
        )

    def _build_fusion_layers(self):
        """Build fusion layers based on the chosen strategy."""

        if self.fusion_strategy == "concat":
            # Simple concatenation fusion
            self.fusion_dim = self.text_dim + self.image_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.fusion_dim // 2, self.num_classes),
            )

        elif self.fusion_strategy == "attention":
            # Attention-based fusion
            self.fusion_dim = 512

            # Project both modalities to same dimension
            self.text_projection = nn.Linear(self.text_dim, self.fusion_dim)
            self.image_projection = nn.Linear(self.image_dim, self.fusion_dim)

            # Multi-head attention for fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim, num_heads=8, batch_first=True
            )

            # Final classifier
            self.fusion_classifier = nn.Sequential(
                nn.LayerNorm(self.fusion_dim),
                nn.Dropout(0.3),
                nn.Linear(self.fusion_dim, self.fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.fusion_dim // 2, self.num_classes),
            )

        elif self.fusion_strategy == "cross_attention":
            # Cross-modal attention
            self.fusion_dim = 512

            self.text_projection = nn.Linear(self.text_dim, self.fusion_dim)
            self.image_projection = nn.Linear(self.image_dim, self.fusion_dim)

            # Cross-attention layers
            self.text_to_image_attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim, num_heads=8, batch_first=True
            )

            self.image_to_text_attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim, num_heads=8, batch_first=True
            )

            # Fusion and classification
            self.layer_norm = nn.LayerNorm(self.fusion_dim)
            self.fusion_classifier = nn.Sequential(
                nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.fusion_dim, self.num_classes),
            )

        else:
            raise ValueError(
                f"Unknown fusion strategy: {self.fusion_strategy}"
            )

    def forward(
        self,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass through the multi-modal model.

        Args:
            text_input_ids: Text token IDs
            text_attention_mask: Text attention mask
            image_pixel_values: Image pixel values

        Returns:
            Dict containing predictions and features
        """
        batch_size = None
        text_features = None
        image_features = None
        text_logits = None
        image_logits = None

        # Process text if available
        if text_input_ids is not None and text_attention_mask is not None:
            batch_size = text_input_ids.size(0)
            text_outputs = self.text_model(text_input_ids, text_attention_mask)
            text_features = text_outputs["features"]
            text_logits = text_outputs["logits"]

        # Process image if available
        if image_pixel_values is not None:
            if batch_size is None:
                batch_size = image_pixel_values.size(0)
            image_outputs = self.image_model(image_pixel_values)
            image_features = image_outputs["features"]
            image_logits = image_outputs["logits"]

        # Fusion
        if text_features is not None and image_features is not None:
            # Both modalities available - perform fusion
            fused_logits = self._fuse_features(text_features, image_features)

        elif text_features is not None:
            # Only text available
            fused_logits = text_logits

        elif image_features is not None:
            # Only image available
            fused_logits = image_logits

        else:
            # No input provided
            fused_logits = torch.zeros(1, self.num_classes)

        return {
            "logits": fused_logits,
            "text_logits": text_logits,
            "image_logits": image_logits,
            "text_features": text_features,
            "image_features": image_features,
        }

    def _fuse_features(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse text and image features based on the fusion strategy.

        Args:
            text_features: Text feature representations
            image_features: Image feature representations

        Returns:
            torch.Tensor: Fused classification logits
        """
        logits = None  # Initialize logits

        if self.fusion_strategy == "concat":
            # Simple concatenation
            fused_features = torch.cat([text_features, image_features], dim=1)
            logits = self.fusion_layer(fused_features)

        elif self.fusion_strategy == "attention":
            # Project to same dimension
            text_proj = self.text_projection(text_features)
            image_proj = self.image_projection(image_features)

            # Stack for attention
            features = torch.stack([text_proj, image_proj], dim=1)

            # Apply attention
            attended_features, _ = self.attention(features, features, features)

            # Aggregate attended features (mean pooling)
            aggregated = attended_features.mean(dim=1)

            # Classify
            logits = self.fusion_classifier(aggregated)

        elif self.fusion_strategy == "cross_attention":
            # Project to same dimension
            text_proj = self.text_projection(text_features).unsqueeze(1)
            image_proj = self.image_projection(image_features).unsqueeze(1)

            # Cross-attention
            text_attended, _ = self.text_to_image_attention(
                text_proj, image_proj, image_proj
            )
            image_attended, _ = self.image_to_text_attention(
                image_proj, text_proj, text_proj
            )

            # Concatenate attended features
            text_attended = text_attended.squeeze(1)
            image_attended = image_attended.squeeze(1)

            # Layer norm
            text_attended = self.layer_norm(text_attended)
            image_attended = self.layer_norm(image_attended)

            # Fuse and classify
            fused = torch.cat([text_attended, image_attended], dim=1)
            logits = self.fusion_classifier(fused)

        else:
            # Fallback for unknown fusion strategy
            logits = torch.zeros(text_features.size(0), self.num_classes)

        return logits

    def predict_text_only(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict using only text input."""
        result = self.forward(
            text_input_ids=input_ids, text_attention_mask=attention_mask
        )["logits"]
        if result is None:
            return torch.zeros(input_ids.size(0), self.num_classes)
        return result

    def predict_image_only(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Predict using only image input."""
        result = self.forward(image_pixel_values=pixel_values)["logits"]
        if result is None:
            return torch.zeros(pixel_values.size(0), self.num_classes)
        return result


class EnsembleMultiModalClassifier(nn.Module):
    """Ensemble of multiple fusion strategies."""

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 5,
    ):
        """
        Initialize ensemble multi-modal classifier.

        Args:
            text_model_name: Name of the text model
            image_model_name: Name of the image model
            num_classes: Number of output classes
        """
        super(EnsembleMultiModalClassifier, self).__init__()

        # Create multiple fusion models
        self.models = nn.ModuleList(
            [
                MultiModalClassifier(
                    text_model_name, image_model_name, num_classes, "concat"
                ),
                MultiModalClassifier(
                    text_model_name, image_model_name, num_classes, "attention"
                ),
                MultiModalClassifier(
                    text_model_name,
                    image_model_name,
                    num_classes,
                    "cross_attention",
                ),
            ]
        )

        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)))

    def forward(self, **kwargs) -> torch.Tensor:
        """Forward pass through ensemble."""
        outputs = []

        for model in self.models:
            output = model(**kwargs)["logits"]
            outputs.append(output)

        # Weighted ensemble
        stacked_outputs = torch.stack(outputs, dim=0)
        weights = F.softmax(self.ensemble_weights, dim=0)

        ensemble_output = torch.sum(
            stacked_outputs * weights.view(-1, 1, 1), dim=0
        )

        return ensemble_output


def create_multimodal_model(
    model_type: str = "standard", **kwargs
) -> nn.Module:
    """
    Factory function to create multi-modal models.

    Args:
        model_type: Type of model ('standard', 'ensemble')
        **kwargs: Additional arguments

    Returns:
        nn.Module: Multi-modal classification model
    """
    if model_type == "standard":
        return MultiModalClassifier(**kwargs)
    elif model_type == "ensemble":
        return EnsembleMultiModalClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Test the multi-modal models."""
    # Test standard multi-modal model
    model = MultiModalClassifier(fusion_strategy="attention", num_classes=5)

    # Create dummy inputs
    batch_size = 2
    seq_length = 128
    text_input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    text_attention_mask = torch.ones(batch_size, seq_length)
    image_pixel_values = torch.randn(batch_size, 3, 224, 224)

    # Forward pass with both modalities
    outputs = model(
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        image_pixel_values=image_pixel_values,
    )

    print("Multi-modal outputs:")
    for key, value in outputs.items():
        if value is not None:
            print(f"  {key}: {value.shape}")

    # Test text-only
    text_only_outputs = model(
        text_input_ids=text_input_ids, text_attention_mask=text_attention_mask
    )
    print(f"Text-only logits shape: {text_only_outputs['logits'].shape}")

    # Test image-only
    image_only_outputs = model(image_pixel_values=image_pixel_values)
    print(f"Image-only logits shape: {image_only_outputs['logits'].shape}")


if __name__ == "__main__":
    main()
