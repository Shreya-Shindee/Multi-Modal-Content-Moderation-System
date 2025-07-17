"""
Simple Working Demo
==================

A simplified demo that shows the core functionality working.
"""

import torch
from PIL import Image
from preprocessing.image_processor import ImageProcessor


def test_image_processing():
    """Test image processing which we know works."""
    print("🖼️ Testing Image Processing...")
    try:
        # Initialize processor
        processor = ImageProcessor()
        print("✅ Image processor initialized")

        # Create test images
        test_images = [
            Image.new("RGB", (300, 400), color="red"),
            Image.new("RGB", (500, 300), color="blue"),
            Image.new("RGB", (224, 224), color="green"),
        ]

        # Process single images
        for i, img in enumerate(test_images):
            processed = processor.preprocess_image(img)
            print(f"   Image {i+1}: {img.size} -> {processed.shape}")

        # Test batch processing
        from typing import cast, Union, List
        batch_images = cast(List[Union[str, Image.Image]], test_images)
        batch = processor.preprocess_batch(batch_images)
        print(f"   Batch: {len(test_images)} images -> {batch.shape}")

        # Test feature extraction
        features = processor.extract_features(batch)
        print(f"   Features extracted: {features.shape}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_basic_pytorch():
    """Test basic PyTorch functionality."""
    print("\n🔥 Testing PyTorch Setup...")

    try:
        # Check PyTorch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")

        # Create a simple tensor
        x = torch.randn(2, 3, 224, 224)
        print(f"   Created tensor: {x.shape}")

        # Simple neural network
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 224 * 224, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 5),
        )

        with torch.no_grad():
            output = model(x)

        print(f"   Model output: {output.shape}")
        print("✅ PyTorch working correctly")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Run working demo."""
    print("🚀 Multi-Modal Content Moderation - Simple Demo")
    print("=" * 50)

    # Test what we know works
    results = [test_basic_pytorch(), test_image_processing()]

    print("\n" + "=" * 50)
    print("📊 Results:")
    print(f"   ✅ Working: {sum(results)}/{len(results)}")

    if all(results):
        print("\n🎉 Core components are working!")
        print("\nThe system is ready for:")
        print("• Image preprocessing and feature extraction")
        print("• PyTorch model operations")
        print("• Basic neural network functionality")

        print("\nNext steps:")
        print("1. Fix model parameter inconsistencies")
        print("2. Start API server: uvicorn api.main:app --reload")
        print("3. Launch frontend: streamlit run frontend/app.py")
    else:
        print("\n⚠️ Some core components need attention")


if __name__ == "__main__":
    main()
