"""
Quick Demo Script
================

Test the multi-modal content moderation system components.
"""

import sys
import os
import torch
from PIL import Image
from preprocessing.text_processor import TextProcessor
from preprocessing.image_processor import ImageProcessor
from models.text_model import TextClassifier
from models.image_model import ImageClassifier
from models.multimodal_model import MultiModalClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_text_processing():
    """Test text processing pipeline."""
    print("🔤 Testing Text Processing...")

    try:
        processor = TextProcessor()

        # Test text samples
        test_texts = [
            "This is a normal, safe message about daily life.",
            "I really hate when people do inconsiderate things.",
            "Beautiful sunset at the beach today! 🌅",
            "Stop bothering me with your spam messages.",
        ]

        print("✅ Text processor initialized successfully")

        for i, text in enumerate(test_texts):
            processed = processor.preprocess_text(text)
            print(
                f"   Sample {i+1}: '{text[:50]}...' -> '{processed[:50]}...'"
            )

        return True

    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        return False


def test_image_processing():
    """Test image processing pipeline."""
    print("\n🖼️ Testing Image Processing...")

    try:
        processor = ImageProcessor()

        # Create dummy test images
        test_images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (300, 300), color="blue"),
            Image.new("RGB", (150, 200), color="green"),
        ]

        print("✅ Image processor initialized successfully")

        for i, img in enumerate(test_images):
            processed = processor.preprocess_image(img)
            print(f"   Image {i+1}: {img.size} -> {processed.shape}")

        # Test batch processing
        from typing import cast, Union, List
        batch_images = cast(List[Union[str, Image.Image]], test_images)
        batch = processor.preprocess_batch(batch_images)
        print(
            f"   Batch processing: {len(test_images)} images -> {batch.shape}"
        )

        return True

    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False


def test_models():
    """Test model architectures."""
    print("\n🤖 Testing Model Architectures...")

    try:
        # Test text model
        print("   Testing text model...")
        text_model = TextClassifier(num_classes=5)

        # Create dummy input
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 30000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        with torch.no_grad():
            text_output = text_model(input_ids, attention_mask)

        print(f"      Text model output shape: {text_output['logits'].shape}")

        # Test image model
        print("   Testing image model...")
        image_model = ImageClassifier(num_classes=5)

        # Create dummy image input
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            image_output = image_model(pixel_values)

        print(
            f"      Image model output shape: {image_output['logits'].shape}"
        )

        # Test multimodal model
        print("   Testing multimodal model...")
        multimodal_model = MultiModalClassifier(num_classes=5)

        with torch.no_grad():
            multimodal_output = multimodal_model(
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                image_pixel_values=pixel_values,
            )

        print(
            f"      Multimodal output shape: "
            f"{multimodal_output['logits'].shape}"
        )
        print("✅ All models initialized and tested successfully")

        return True

    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False


def test_end_to_end():
    """Test end-to-end pipeline."""
    print("\n🔄 Testing End-to-End Pipeline...")

    try:
        # Initialize components
        text_processor = TextProcessor()
        image_processor = ImageProcessor()
        model = MultiModalClassifier(num_classes=5)

        # Test data
        test_text = "This is a test message for content moderation."
        test_image = Image.new("RGB", (224, 224), color="blue")

        # Process inputs
        processed_text = text_processor.preprocess_text(test_text)
        if text_processor.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        encoding = text_processor.tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        processed_image = image_processor.preprocess_image(test_image)
        image_batch = processed_image.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = model(
                text_input_ids=encoding["input_ids"],
                text_attention_mask=encoding["attention_mask"],
                image_pixel_values=image_batch,
            )

        # Get predictions
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence = probabilities[0, predicted_class[0]].item()

        class_names = [
            "Safe",
            "Hate Speech",
            "Violence",
            "Sexual Content",
            "Harassment",
        ]
        prediction = class_names[int(predicted_class[0].item())]

        print(f"   Input text: '{test_text}'")
        print(f"   Input image: {test_image.size} RGB image")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   All probabilities: {probabilities[0].tolist()}")

        print("✅ End-to-end pipeline completed successfully")
        return True

    except Exception as e:
        print(f"❌ End-to-end pipeline failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Multi-Modal Content Moderation System - Demo Test")
    print("=" * 60)

    # Run tests
    tests = [
        test_text_processing,
        test_image_processing,
        test_models,
        test_end_to_end,
    ]

    results = []
    for test in tests:
        results.append(test())

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   ✅ Passed: {sum(results)}/{len(results)}")
    print(f"   ❌ Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Start the API server: uvicorn api.main:app --reload")
        print("2. Launch the frontend: streamlit run frontend/app.py")
        print("3. Access the web interface at: http://localhost:8501")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")

    return all(results)


if __name__ == "__main__":
    main()
