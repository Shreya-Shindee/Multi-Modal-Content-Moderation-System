"""
Streamlit Frontend for Multi-Modal Content Moderation
====================================================

Enhanced interactive web interface for content moderation system.
"""

import streamlit as st
import requests
from PIL import Image
from typing import Dict, Any
import time
import plotly.express as px
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Modal Content Moderation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .safe-prediction {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .unsafe-prediction {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .warning-prediction {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .stats-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #262730;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
    """Send text prediction request to API with loading indicator."""
    with st.spinner('🔍 Analyzing text content...'):
        # Brief delay for UX
        time.sleep(0.3)

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
    """Send multimodal prediction request to API with loading indicator."""
    with st.spinner('🔍 Analyzing multimodal content...'):
        # Brief delay for UX
        time.sleep(0.5)

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
    """Display prediction results in an enhanced, visually appealing way."""
    if "error" in results:
        st.error(f"❌ {results['error']}")
        return

    # Main prediction with enhanced styling
    prediction = results.get("prediction", "Unknown")
    confidence = results.get("confidence", 0.0)

    # Enhanced prediction display with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if prediction == "Safe":
            st.markdown(f"""
            <div class="safe-prediction">
                <h2>✅ CONTENT IS SAFE</h2>
                <h3>Confidence: {confidence:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
        elif prediction in ["Hate Speech", "Violence", "Sexual Content",
                            "Harassment"]:
            st.markdown(f"""
            <div class="unsafe-prediction">
                <h2>⚠️ HARMFUL CONTENT DETECTED</h2>
                <h3>Category: {prediction}</h3>
                <h3>Confidence: {confidence:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-prediction">
                <h2>❓ UNCERTAIN CLASSIFICATION</h2>
                <h3>Category: {prediction}</h3>
                <h3>Confidence: {confidence:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Enhanced detailed scores visualization
    scores = results.get("scores", {})
    if scores:
        st.markdown("### 📊 Detailed Analysis")

        # Create a more visually appealing scores display
        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart for scores
            df_scores = pd.DataFrame(
                list(scores.items()),
                columns=['Category', 'Score']
            )
            df_scores['Score_Percent'] = df_scores['Score'] * 100

            # Color map for categories
            color_map = {
                'Safe': '#28a745',
                'Hate Speech': '#dc3545',
                'Violence': '#fd7e14',
                'Sexual Content': '#e83e8c',
                'Harassment': '#6f42c1'
            }

            fig = px.bar(
                df_scores,
                x='Score_Percent',
                y='Category',
                orientation='h',
                title='Confidence Scores by Category',
                color='Category',
                color_discrete_map=color_map,
                text='Score_Percent'
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='inside'
            )
            fig.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="Confidence (%)",
                yaxis_title="",
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Metrics cards
            st.markdown("#### Key Metrics")
            for category, score in scores.items():
                # Color coding for metrics
                if category == "Safe":
                    delta_color = "normal" if score > 0.5 else "inverse"
                else:
                    delta_color = "inverse" if score > 0.5 else "normal"

                st.metric(
                    label=category,
                    value=f"{score:.1%}",
                    delta=f"{score-0.5:.1%}" if score != 0.5 else None,
                    delta_color=delta_color
                )

    # Processing statistics in an attractive container
    processing_time = results.get("processing_time", 0)
    modalities = results.get("modalities_used", [])

    st.markdown(f"""
    <div class="stats-container">
        <h4>📈 Processing Statistics</h4>
        <div style="display: flex; justify-content: space-around;
                   margin-top: 1rem;">
            <div style="text-align: center;">
                <h3 style="color: #1f77b4; margin: 0;">
                    {processing_time:.3f}s
                </h3>
                <p style="margin: 0; color: #666;">Processing Time</p>
            </div>
            <div style="text-align: center;">
                <h3 style="color: #1f77b4; margin: 0;">{len(modalities)}</h3>
                <p style="margin: 0; color: #666;">Modalities Used</p>
            </div>
            <div style="text-align: center;">
                <h3 style="color: #1f77b4; margin: 0;">
                    {', '.join(modalities)}
                </h3>
                <p style="margin: 0; color: #666;">Analysis Types</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk assessment gauge
    if prediction != "Safe":
        risk_level = confidence * 100
        if risk_level > 80:
            risk_color = "#dc3545"
            risk_text = "HIGH RISK"
        elif risk_level > 60:
            risk_color = "#fd7e14"
            risk_text = "MEDIUM RISK"
        else:
            risk_color = "#ffc107"
            risk_text = "LOW RISK"

        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <h4>🎯 Risk Assessment</h4>
            <div style="background: {risk_color}; color: white; padding: 1rem;
                        border-radius: 10px; font-weight: bold;
                        font-size: 1.2rem;">
                {risk_text} ({risk_level:.0f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)


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
        st.markdown("**System Status:**")
        st.markdown(f"- API Health: {API_BASE_URL}/health")

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
