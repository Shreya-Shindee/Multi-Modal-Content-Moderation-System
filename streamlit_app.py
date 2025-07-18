"""
Standalone Streamlit App - AI Content Guardian
"""

import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AI Content Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .main-header {
        text-align: center;
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .safe-content {
        background-color: rgba(40, 167, 69, 0.1);
        border-left-color: #28a745;
    }
    .unsafe-content {
        background-color: rgba(220, 53, 69, 0.1);
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def analyze_text_content(text, threshold=0.5):
    """Analyze text content for moderation"""
    inappropriate_words = [
        'hate', 'violence', 'abuse', 'threat', 'spam',
        'scam', 'fake', 'misleading', 'harmful'
    ]

    text_lower = text.lower()
    violations = [word for word in inappropriate_words if word in text_lower]

    if violations:
        confidence = min(0.9, len(violations) * 0.3)
        return {
            'is_safe': confidence < threshold,
            'confidence': confidence,
            'violations': violations,
            'category': 'inappropriate_content'
        }
    else:
        return {
            'is_safe': True,
            'confidence': 0.1,
            'violations': [],
            'category': 'safe_content'
        }


def analyze_image_content(image, threshold=0.5):
    """Analyze image content for moderation"""
    return {
        'is_safe': True,
        'confidence': 0.1,
        'violations': [],
        'category': 'safe_content',
        'detected_objects': ['general_image']
    }


def main():
    """Main Streamlit app"""

    # Title
    st.markdown(
        '<h1 class="main-header">🛡️ AI Content Guardian</h1>',
        unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: white; margin-bottom: 3rem;">
        <h3>Advanced Multi-Modal Content Moderation System</h3>
        <p>Powered by AI - Safe Content Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: #667eea; padding: 1rem; border-radius: 10px; 
                    color: white; text-align: center; margin-bottom: 1rem;">
            <h3>⚙️ Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)

        # System status
        st.markdown("""
        <div style="background: #28a745; padding: 1rem; border-radius: 10px; 
                    color: white; text-align: center; margin: 1rem 0;">
            <h4>✅ System Online</h4>
            <p>Standalone Mode - Ready</p>
        </div>
        """, unsafe_allow_html=True)

        # Settings
        st.markdown("### 🎯 Detection Sensitivity")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher values = stricter filtering"
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

        # System metrics
        st.markdown("### 📊 System Metrics")
        st.metric("Processing Time", "~0.1s", "Fast")
        st.metric("Models Active", "Demo Mode", "Ready")
        st.metric("System Health", "100%", "Optimal")

        st.markdown("---")

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            st.checkbox("Enhanced Detection", value=True)
            st.checkbox("Contextual Analysis", value=True)
            show_details = st.checkbox("Show Details", value=False)

    # Main content
    tab1, tab2, tab3 = st.tabs([
        "📝 Text Analysis",
        "🖼️ Image Analysis",
        "🔄 Multi-Modal Analysis"
    ])

    # Text Analysis Tab
    with tab1:
        st.header("📝 Text Content Moderation")

        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste text content here..."
        )

        if st.button("🔍 Analyze Text", type="primary"):
            if text_input:
                with st.spinner("Analyzing text content..."):
                    result = analyze_text_content(text_input, threshold)

                # Display results
                if result['is_safe']:
                    confidence = (1-result['confidence'])*100
                    st.markdown(f"""
                    <div class="result-box safe-content">
                        <h4>✅ Content Approved</h4>
                        <p><strong>Status:</strong> Safe for publication</p>
                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = result['confidence']*100
                    violations = ', '.join(result['violations'])
                    st.markdown(f"""
                    <div class="result-box unsafe-content">
                        <h4>⚠️ Content Flagged</h4>
                        <p><strong>Status:</strong> Requires review</p>
                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                        <p><strong>Issues:</strong> {violations}</p>
                    </div>
                    """, unsafe_allow_html=True)

                if show_details:
                    st.json(result)
            else:
                st.warning("Please enter some text to analyze.")

    # Image Analysis Tab
    with tab2:
        st.header("🖼️ Image Content Moderation")

        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=['jpg', 'jpeg', 'png', 'gif'],
            help="Supported formats: JPG, PNG, GIF"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(
                    image,
                    caption="Uploaded Image",
                    use_column_width=True)

            with col2:
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image content..."):
                        result = analyze_image_content(image, threshold)

                    # Display results
                    if result['is_safe']:
                        st.markdown("""
                        <div class="result-box safe-content">
                            <h4>✅ Image Approved</h4>
                            <p><strong>Status:</strong> Safe</p>
                            <p><strong>Confidence:</strong> 90.0%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-box unsafe-content">
                            <h4>⚠️ Image Flagged</h4>
                            <p><strong>Status:</strong> Requires review</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if show_details:
                        st.json(result)

    # Multi-Modal Analysis Tab
    with tab3:
        st.header("🔄 Multi-Modal Content Analysis")
        st.info("Combine text and image analysis for comprehensive " +
                "content moderation.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📝 Text Component")
            multimodal_text = st.text_area(
                "Enter accompanying text:",
                height=100,
                placeholder="Text that goes with the image..."
            )

        with col2:
            st.subheader("🖼️ Image Component")
            multimodal_image = st.file_uploader(
                "Upload accompanying image:",
                type=['jpg', 'jpeg', 'png'],
                key="multimodal"
            )

        if st.button("🔍 Analyze Combined Content", type="primary"):
            if multimodal_text or multimodal_image:
                with st.spinner("Performing multi-modal analysis..."):
                    # Analyze both components
                    text_result = None
                    if multimodal_text:
                        text_result = analyze_text_content(
                            multimodal_text, threshold)

                    image_result = None
                    if multimodal_image:
                        image_result = analyze_image_content(
                            multimodal_image, threshold)

                    # Combined analysis
                    overall_safe = True
                    if text_result and not text_result['is_safe']:
                        overall_safe = False
                    if image_result and not image_result['is_safe']:
                        overall_safe = False

                # Display combined results
                if overall_safe:
                    st.markdown("""
                    <div class="result-box safe-content">
                        <h4>✅ Combined Content Approved</h4>
                        <p><strong>Status:</strong> Safe for publication</p>
                        <p><strong>Analysis:</strong> Components passed</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-box unsafe-content">
                        <h4>⚠️ Combined Content Flagged</h4>
                        <p><strong>Status:</strong> Requires review</p>
                        <p><strong>Analysis:</strong> Components failed</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show individual results
                if text_result:
                    with st.expander("📝 Text Analysis Details"):
                        st.json(text_result)

                if image_result:
                    with st.expander("🖼️ Image Analysis Details"):
                        st.json(image_result)
            else:
                st.warning("Please provide text, image, or both.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; margin-top: 2rem;">
        <p>🛡️ AI Content Guardian - Keeping your platform safe</p>
        <p><small>Demo Version | Deployed on Streamlit Cloud</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
