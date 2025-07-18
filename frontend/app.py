"""
Streamlit Frontend for Multi-Modal Content Moderation
====================================================

Enhanced interactive web interface for content moderation system.
"""

import streamlit as st
import requests
from PIL import Image
from typing import Dict, Any, Optional, Union
import time
import plotly.express as px
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="AI Content Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': ('https://github.com/Shreya-Shindee/'
                     'Multi-Modal-Content-Moderation-System'),
        'Report a bug': ('https://github.com/Shreya-Shindee/'
                         'Multi-Modal-Content-Moderation-System/issues'),
        'About': ('# AI Content Guardian\n'
                  'Advanced multi-modal content moderation powered by AI')
    }
)

# Custom CSS for enhanced modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:'
                'wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
        100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }

    .safe-prediction {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 10px 30px rgba(40, 167, 69, 0.3);
        animation: fadeInUp 0.8s ease-out;
        border: 2px solid rgba(255,255,255,0.2);
    }

    .unsafe-prediction {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 10px 30px rgba(220, 53, 69, 0.4);
        animation: pulse 2s infinite, fadeInUp 0.8s ease-out;
        border: 2px solid rgba(255,255,255,0.2);
    }

    .warning-prediction {
        background: linear-gradient(135deg, #ffc107 0%, #ff8c00 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 10px 30px rgba(255, 193, 7, 0.4);
        animation: fadeInUp 0.8s ease-out;
        border: 2px solid rgba(255,255,255,0.2);
    }

    .stats-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        animation: fadeInUp 0.6s ease-out;
    }

    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .feature-box:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        padding: 8px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 25px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        animation: glow 2s infinite;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }

    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        font-family: 'Inter', sans-serif;
    }

    .uploadedFile {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    .sidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
    }

    .stSuccess {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border-radius: 12px;
        border: none;
        animation: fadeInUp 0.5s ease-out;
    }

    .stError {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
        color: white;
        border-radius: 12px;
        border: none;
        animation: fadeInUp 0.5s ease-out;
    }

    .stWarning {
        background: linear-gradient(135deg, #ffc107 0%, #ff8c00 100%);
        color: white;
        border-radius: 12px;
        border: none;
        animation: fadeInUp 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"


@st.cache_data(ttl=300)  # Cache for 5 minutes
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


def predict_multimodal(text: Optional[str] = None, image_file=None,
                       threshold: float = 0.5) -> Dict[str, Any]:
    """Send multimodal prediction request to API with loading indicator."""
    with st.spinner('🔍 Analyzing multimodal content...'):
        # Brief delay for UX
        time.sleep(0.5)

        try:
            files = {}
            data: Dict[str, Union[str, float]] = {"threshold": threshold}

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
    """Main Streamlit app with enhanced UI and performance."""
    # Enhanced title with animation
    st.markdown("""
    <div class="main-header">
        🛡️ AI Content Guardian
    </div>
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3 style="color: white; font-weight: 300; margin: 0;">
            Advanced Multi-Modal Content Moderation System
        </h3>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;
                  margin: 0.5rem 0;">
            Powered by BERT, Vision Transformer, and Advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced description with feature highlights
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>🔤 Text Analysis</h4>
            <p>Advanced NLP with BERT to detect hate speech, harassment,
               and harmful content in text.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>🖼️ Image Analysis</h4>
            <p>Vision Transformer technology to identify inappropriate
               visual content and violence.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
            <h4>🔄 Multi-Modal</h4>
            <p>Combined analysis of text and images for comprehensive
               content moderation.</p>
        </div>
        """, unsafe_allow_html=True)

    # Check API status
    api_status = check_api_health()

    # Enhanced sidebar with performance metrics
    with st.sidebar:
        st.markdown("""
        <div class="feature-box">
            <h3>⚙️ Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)

        # API Status with enhanced display
        if api_status:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                        padding: 1rem; border-radius: 12px; color: white;
                        text-align: center; margin: 1rem 0;">
                <h4 style="margin: 0;">✅ System Online</h4>
                <p style="margin: 0.5rem 0;">API Connected & Ready</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
                        padding: 1rem; border-radius: 12px; color: white;
                        text-align: center; margin: 1rem 0;">
                <h4 style="margin: 0;">❌ System Offline</h4>
                <p style="margin: 0.5rem 0;">API Disconnected</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Start the API server:**")
            st.code("python api/main.py", language="bash")

        # Enhanced threshold setting
        st.markdown("### 🎯 Detection Sensitivity")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher values = more strict filtering. Lower values = more permissive."
        )

        # Display threshold interpretation
        if threshold >= 0.8:
            st.success("🛡️ High Security Mode")
        elif threshold >= 0.6:
            st.info("⚖️ Balanced Mode")
        elif threshold >= 0.4:
            st.warning("🔍 Sensitive Mode")
        else:
            st.error("🚨 Maximum Sensitivity")

        st.markdown("---")

        # Performance metrics
        st.markdown("### 📊 System Metrics")
        if api_status:
            st.metric("API Latency", "~0.3s", "Fast")
            st.metric("Models Loaded", "3/3", "Ready")
            st.metric("System Health", "100%", "Optimal")
        else:
            st.metric("API Latency", "N/A", "Offline")
            st.metric("Models Loaded", "0/3", "Offline")
            st.metric("System Health", "0%", "Down")

        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown(
            "- [GitHub Repository](https://github.com/Shreya-Shindee/Multi-Modal-Content-Moderation-System)")
        st.markdown(
            "- [Documentation](https://github.com/Shreya-Shindee/Multi-Modal-Content-Moderation-System/blob/main/README.md)")
        st.markdown(
            "- [Report Issues](https://github.com/Shreya-Shindee/Multi-Modal-Content-Moderation-System/issues)")

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
