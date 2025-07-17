"""
Text Classification Model
========================

BERT-based text classifier for content moderation.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    """BERT-based text classifier for content moderation."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 5,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        """
        Initialize the text classifier.

        Args:
            model_name: Pretrained BERT model name
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT parameters
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load BERT model and tokenizer
        try:
            self.bert = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded BERT model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise

        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # Initialize weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)

        Returns:
            Dict containing logits and features
        """
        try:
            # BERT forward pass
            bert_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if token_type_ids is not None:
                bert_inputs["token_type_ids"] = token_type_ids

            outputs = self.bert(**bert_inputs)

            # Use [CLS] token representation
            pooled_output = outputs.last_hidden_state[:, 0, :]

            # Apply dropout
            pooled_output = self.dropout(pooled_output)

            # Classification
            logits = self.classifier(pooled_output)

            return {"logits": logits, "features": pooled_output}

        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Return zeros as fallback
            batch_size = input_ids.size(0)
            return {
                "logits": torch.zeros(batch_size, self.num_classes),
                "features": torch.zeros(batch_size, 768),
            }

    def get_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get text embeddings without classification.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            torch.Tensor: Text embeddings
        """
        try:
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                embeddings = outputs.last_hidden_state[:, 0, :]
                return embeddings

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return torch.zeros(input_ids.size(0), 768)


class AttentionTextClassifier(nn.Module):
    """Text classifier with attention mechanism."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 5,
        num_attention_heads: int = 8,
    ):
        """
        Initialize attention-based text classifier.

        Args:
            model_name: Name of the pretrained BERT model
            num_classes: Number of output classes
            num_attention_heads: Number of attention heads
        """
        super(AttentionTextClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        config = self.bert.config

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        # Classification layers
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Apply attention
        attended_output, _ = self.attention(
            sequence_output,
            sequence_output,
            sequence_output,
            key_padding_mask=~attention_mask.bool(),
        )

        # Use [CLS] token
        cls_output = attended_output[:, 0, :]

        # Classification
        cls_output = self.layer_norm(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits


def create_text_model(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create text models.

    Args:
        model_type: Type of model ('standard', 'attention')
        **kwargs: Additional arguments for model initialization

    Returns:
        nn.Module: Text classification model
    """
    if model_type == "standard":
        return TextClassifier(**kwargs)
    elif model_type == "attention":
        return AttentionTextClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Test the text models."""
    # Test standard model
    model = TextClassifier(num_classes=5)

    # Create dummy inputs
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Forward pass
    outputs = model(input_ids, attention_mask)
    print("Standard model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test attention model
    attention_model = AttentionTextClassifier(num_classes=5)
    attention_outputs = attention_model(input_ids, attention_mask)
    print(f"Attention model output shape: {attention_outputs.shape}")


if __name__ == "__main__":
    main()
