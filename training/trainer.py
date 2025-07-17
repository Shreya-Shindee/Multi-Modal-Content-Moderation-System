"""
Training Pipeline
================

Comprehensive training pipeline for multi-modal content moderation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup  # type: ignore
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import pandas as pd
from typing import Dict, List, Tuple
import logging
import time
import os
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


class ContentModerationDataset(Dataset):
    """Dataset for multi-modal content moderation."""

    def __init__(
        self,
        data: pd.DataFrame,
        text_processor,
        image_processor,
        text_column: str = "text",
        image_column: str = "image_path",
        label_column: str = "label",
        max_length: int = 512,
    ):
        """
        Initialize the dataset.

        Args:
            data: DataFrame containing the data
            text_processor: Text preprocessing instance
            image_processor: Image preprocessing instance
            text_column: Name of text column
            image_column: Name of image path column
            label_column: Name of label column
            max_length: Maximum text sequence length
        """
        self.data = data
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.text_column = text_column
        self.image_column = image_column
        self.label_column = label_column
        self.max_length = max_length

        # Create label mapping if labels are strings
        if self.data[label_column].dtype == "object":
            self.label_map = {
                label: idx
                for idx, label in enumerate(self.data[label_column].unique())
            }
            self.data["label_id"] = self.data[label_column].map(self.label_map)
        else:
            self.label_map = None
            self.data["label_id"] = self.data[label_column]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        row = self.data.iloc[idx]

        # Process text
        text = str(row.get(self.text_column, ""))
        text_tokens = self.text_processor.tokenize_batch(
            [text], self.max_length
        )

        # Process image
        image_path = row.get(self.image_column, "")
        try:
            if image_path and os.path.exists(image_path):
                image_tensor = self.image_processor.preprocess_image(
                    image_path
                )
            else:
                # Create dummy image if no image available
                image_tensor = torch.zeros(3, 224, 224)
        except Exception:
            image_tensor = torch.zeros(3, 224, 224)

        # Get label
        label = torch.tensor(row["label_id"], dtype=torch.long)

        return {
            "text_input_ids": text_tokens["input_ids"].squeeze(0),
            "text_attention_mask": text_tokens["attention_mask"].squeeze(0),
            "image_pixel_values": image_tensor,
            "labels": label,
        }


class MultiModalTrainer:
    """Trainer for multi-modal content moderation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            use_wandb: Whether to use wandb for logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        total_steps = len(train_loader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training
        self.scaler = GradScaler()

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_accuracy = 0.0

        logger.info("Trainer initialized successfully")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    image_pixel_values=batch["image_pixel_values"],
                )

                logits = outputs["logits"]
                loss = self.criterion(logits, batch["labels"])

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item()
            total_samples += batch["labels"].size(0)

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": total_loss / (batch_idx + 1),
                    "lr": self.scheduler.get_last_lr()[0],
                }
            )

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )

        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1,
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    image_pixel_values=batch["image_pixel_values"],
                )

                logits = outputs["logits"]
                loss = self.criterion(logits, batch["labels"])

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        metrics = {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "confusion_matrix": cm,
        }

        return metrics

    def train(
        self, num_epochs: int, save_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints

        Returns:
            Dict containing training history
        """
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["epoch"] = epoch
            epoch_metrics["training_time"] = time.time() - start_time

            # Log metrics
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)

            # Print metrics
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.4f}"
            )
            logger.info(
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["val_accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["val_accuracy"]
                best_model_path = os.path.join(save_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_val_accuracy": self.best_val_accuracy,
                        "metrics": epoch_metrics,
                    },
                    best_model_path,
                )
                logger.info(
                    f"New best model saved with accuracy: "
                    f"{self.best_val_accuracy:.4f}"
                )

            # Save checkpoint
            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "metrics": epoch_metrics,
                },
                checkpoint_path,
            )

            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log(epoch_metrics)

        logger.info(
            f"Training completed. Best validation accuracy: "
            f"{self.best_val_accuracy:.4f}"
        )

        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def create_data_loaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    text_processor,
    image_processor,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.

    Args:
        train_data: Training data DataFrame
        val_data: Validation data DataFrame
        text_processor: Text processor instance
        image_processor: Image processor instance
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of train and validation data loaders
    """
    train_dataset = ContentModerationDataset(
        train_data, text_processor, image_processor
    )

    val_dataset = ContentModerationDataset(
        val_data, text_processor, image_processor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    """Example training script."""
    # This would be implemented with actual data
    logger.info("Training script template created")

    # Example usage:
    # 1. Load and preprocess data
    # 2. Create data loaders
    # 3. Initialize model
    # 4. Create trainer
    # 5. Train model

    print("Training module created successfully!")


if __name__ == "__main__":
    main()
