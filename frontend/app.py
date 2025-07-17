"""
Streamlit Frontend for Multi-Modal Content Moderation
====================================================

Interactive web interface for content moderation system.
"""

import streamlit as st
import requests
from PIL import Image
from typing import Dict, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Modal Content Moderation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except BaseException:
        return False


def predict_text(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Send text prediction request to API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/text",
            json={"text": text, "threshold": threshold},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}


def predict_multimodal(text: str = None, image_file=None,
                       threshold: float = 0.5) -> Dict[str, Any]:
    """Send multimodal prediction request to API."""
    try:
        files = {}
        data = {"threshold": threshold}

        if text:
            data["text"] = text

        if image_file:
            files["image"] = ("image.jpg", image_file, "image/jpeg")

        response = requests.post(
            f"{API_BASE_URL}/predict/multimodal",
            data=data,
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}


def display_prediction_results(results: Dict[str, Any]):
    """Display prediction results in a formatted way."""
    if "error" in results:
        st.error(f"❌ {results['error']}")
        return

    # Main prediction
    prediction = results.get("prediction", "Unknown")
    confidence = results.get("confidence", 0.0)

    # Color code based on prediction
    if prediction == "Safe":
        st.success(
            f"✅ **Prediction: {prediction}** "
            f"(Confidence: {confidence:.2%})")
    elif prediction in ["Hate Speech", "Violence", "Sexual Content",
                        "Harassment"]:
        st.error(
            f"⚠️ **Prediction: {prediction}** "
            f"(Confidence: {confidence:.2%})")
    else:
        st.warning(
            f"❓ **Prediction: {prediction}** (Confidence: {confidence:.2%})")

    # Detailed scores
    scores = results.get("scores", {})
    if scores:
        st.subheader("📊 Detailed Scores")

        # Create columns for better layout
        cols = st.columns(len(scores))
        for i, (category, score) in enumerate(scores.items()):
            with cols[i]:
                # Color code the metric
                if category == "Safe":
                    st.metric(
                        label=category,
                        value=f"{score:.2%}",
                        delta=None
                    )
                else:
                    st.metric(
                        label=category,
                        value=f"{score:.2%}",
                        delta=None
                    )

    # Processing info
    processing_time = results.get("processing_time", 0)
    modalities = results.get("modalities_used", [])

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"⏱️ Processing Time: {processing_time:.3f}s")
    with col2:
        st.info(f"🔍 Modalities: {', '.join(modalities)}")


def main():
    """Main Streamlit app."""
    # Title and description
    st.title("🛡️ Multi-Modal Content Moderation System")
    st.markdown("""
    This system analyzes text and images to detect harmful content including:
    - Hate Speech
    - Violence
    - Sexual Content
    - Harassment

    Upload content below to get started!
    """)

    # Check API status
    api_status = check_api_health()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Status
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.markdown("Make sure the API server is running:")
            st.code("python api/main.py")

        # Threshold setting
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence level for predictions"
        )

        st.markdown("---")
        st.markdown("**API Endpoints:**")
        st.markdown(f"- Health: {API_BASE_URL}/health")
        st.markdown(f"- Docs: {API_BASE_URL}/docs")

    if not api_status:
        st.warning("⚠️ Please start the API server to use the moderation "
                   "system.")
        st.stop()

    # Main content area
    tab1, tab2, tab3 = st.tabs(
        ["📝 Text Analysis", "🖼️ Image Analysis", "🔄 Multi-Modal Analysis"])

    # Text Analysis Tab
    with tab1:
        st.header("📝 Text Content Moderation")

        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here...",
            height=150
        )

        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    results = predict_text(text_input, threshold)
                    display_prediction_results(results)
            else:
                st.warning("Please enter some text to analyze.")

    # Image Analysis Tab
    with tab2:
        st.header("🖼️ Image Content Moderation")

        uploaded_image = st.file_uploader(
            "Upload an image:",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )

        if uploaded_image:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Reset file pointer
                    uploaded_image.seek(0)
                    results = predict_multimodal(
                        text=None,
                        image_file=uploaded_image,
                        threshold=threshold
                    )
                    display_prediction_results(results)

    # Multi-Modal Analysis Tab
    with tab3:
        st.header("🔄 Multi-Modal Content Moderation")
        st.markdown(
            "Analyze both text and image together for comprehensive "
            "moderation.")

        # Text input
        multi_text = st.text_area(
            "Text content (optional):",
            placeholder="Enter text that accompanies the image...",
            height=100
        )

        # Image input
        multi_image = st.file_uploader(
            "Upload image (optional):",
            type=["jpg", "jpeg", "png", "bmp"],
            key="multimodal_image"
        )

        if multi_image:
            image = Image.open(multi_image)
            st.image(image, caption="Uploaded Image", width=300)

        if st.button("Analyze Content", type="primary"):
            if multi_text.strip() or multi_image:
                with st.spinner("Analyzing multi-modal content..."):
                    # Reset file pointer if image exists
                    if multi_image:
                        multi_image.seek(0)

                    results = predict_multimodal(
                        text=multi_text if multi_text.strip() else None,
                        image_file=multi_image,
                        threshold=threshold
                    )
                    display_prediction_results(results)
            else:
                st.warning(
                    "Please provide either text, image, or both to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🛡️ Multi-Modal Content Moderation System |
        Built with Streamlit & FastAPI</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
