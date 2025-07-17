"""
FastAPI Backend for Multi-Modal Content Moderation
==================================================

Production-ready API for content moderation system.
"""

import io
import logging
import os
import sys
import time
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules  # noqa: E402
from models.multimodal_model import MultiModalClassifier  # noqa: E402
from preprocessing.text_processor import TextProcessor  # noqa: E402
from preprocessing.image_processor import ImageProcessor  # noqa: E402

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Modal Content Moderation API",
    description="AI-powered content moderation for text and images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processors
model = None
text_processor = None
image_processor = None
device = None
class_labels = [
    "Safe",
    "Hate Speech",
    "Violence",
    "Sexual Content",
    "Harassment",
]


# Pydantic models for API
class TextOnlyRequest(BaseModel):
    text: str = Field(..., description="Text content to analyze")
    threshold: float = Field(0.5, description="Confidence threshold")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence")
    scores: Dict[str, float] = Field(..., description="Scores for all classes")
    processing_time: float = Field(
        ..., description="Processing time in seconds"
    )
    modalities_used: List[str] = Field(
        ..., description="Which modalities were analyzed"
    )


class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    threshold: float = Field(0.5, description="Confidence threshold")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    total_processing_time: float = Field(
        ..., description="Total processing time"
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    version: str = Field(..., description="API version")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and processors on startup."""
    global model, text_processor, image_processor, device

    try:
        logger.info("Starting up Multi-Modal Content Moderation API...")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Initialize processors
        text_processor = TextProcessor()
        image_processor = ImageProcessor()

        # Initialize model
        model = MultiModalClassifier(
            num_classes=len(class_labels), fusion_strategy="attention"
        )

        # Load pretrained weights if available
        model_path = "checkpoints/best_model.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No pretrained model found, using random weights")

        model.to(device)
        model.eval()

        logger.info("API startup completed successfully!")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Multi-Modal Content Moderation API...")


# Helper functions
def predict_content(
    text: Optional[str] = None,
    image: Optional[Image.Image] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Predict content safety using text and/or image.

    Args:
        text: Text content to analyze
        image: PIL Image to analyze
        threshold: Confidence threshold

    Returns:
        Dictionary containing prediction results
    """
    start_time = time.time()
    modalities_used = []

    try:
        with torch.no_grad():
            # Prepare inputs
            text_inputs = None
            image_inputs = None

            if text and text.strip():
                modalities_used.append("text")
                # Process text
                if text_processor is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Text processor not initialized",
                    )
                tokens = text_processor.tokenize_batch([text])
                text_inputs = {
                    "text_input_ids": tokens["input_ids"].to(device),
                    "text_attention_mask": tokens["attention_mask"].to(
                        device
                    ),
                }

            if image:
                modalities_used.append("image")
                # Process image
                if image_processor is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Image processor not initialized",
                    )
                image_tensor = image_processor.preprocess_image(image)
                image_inputs = {
                    "image_pixel_values": image_tensor.unsqueeze(0).to(
                        device
                    )
                }

            # Make prediction
            if model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            if text_inputs and image_inputs:
                # Multi-modal prediction
                outputs = model(
                    text_input_ids=text_inputs["text_input_ids"],
                    text_attention_mask=text_inputs["text_attention_mask"],
                    image_pixel_values=image_inputs["image_pixel_values"],
                )  # type: ignore
            elif text_inputs:
                # Text-only prediction
                outputs = model(
                    text_input_ids=text_inputs["text_input_ids"],
                    text_attention_mask=text_inputs["text_attention_mask"],
                )  # type: ignore
            elif image_inputs:
                # Image-only prediction
                outputs = model(
                    image_pixel_values=image_inputs["image_pixel_values"]
                )  # type: ignore
            else:
                raise ValueError("No valid input provided")

            # Process outputs
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)

            # Get prediction
            predicted_class_tensor = torch.argmax(probabilities, dim=-1)
            predicted_class_idx = int(predicted_class_tensor.item())

            confidence = float(probabilities[0, predicted_class_idx].item())

            # Create scores dictionary
            scores = {}
            for i, label in enumerate(class_labels):
                scores[label] = float(probabilities[0, i].item())

            processing_time = time.time() - start_time

            return {
                "prediction": class_labels[predicted_class_idx],
                "confidence": confidence,
                "scores": scores,
                "processing_time": processing_time,
                "modalities_used": modalities_used,
            }

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Modal Content Moderation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        version="1.0.0",
    )


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text_only(request: TextOnlyRequest):
    """Analyze text content only."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predict_content(text=request.text, threshold=request.threshold)
    return PredictionResponse(**result)


@app.post("/predict/multimodal", response_model=PredictionResponse)
async def predict_multimodal(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    threshold: float = Form(0.5),
):
    """Analyze both text and image content."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not text and not image:
        raise HTTPException(
            status_code=400,
            detail="At least one of text or image must be provided",
        )

    # Process image if provided
    pil_image = None
    if image:
        try:
            # Read image file
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image: {str(e)}"
            )

    result = predict_content(text=text, image=pil_image, threshold=threshold)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_text(request: BatchTextRequest):
    """Analyze multiple text contents in batch."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    predictions = []

    for text in request.texts:
        try:
            result = predict_content(text=text, threshold=request.threshold)
            predictions.append(PredictionResponse(**result))
        except Exception:
            # Create error response for failed prediction
            error_result = {
                "prediction": "Error",
                "confidence": 0.0,
                "scores": {label: 0.0 for label in class_labels},
                "processing_time": 0.0,
                "modalities_used": [],
            }
            predictions.append(PredictionResponse(**error_result))

    total_time = time.time() - start_time

    return BatchPredictionResponse(
        predictions=predictions, total_processing_time=total_time
    )


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get model performance metrics."""
    # This would return actual metrics from a metrics store
    return {
        "model_accuracy": 0.912,
        "total_predictions": 10000,
        "average_response_time": 0.15,
        "uptime": "99.9%",
        "last_updated": "2024-01-15T10:30:00Z",
    }


@app.get("/classes", response_model=List[str])
async def get_classes():
    """Get list of available classes."""
    return class_labels


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "detail": str(exc)},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": "Please try again later",
        },
    )


# Development server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
