# Multi-Modal Content Moderation System for Social Media Platforms

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-ff6b6b.svg)](https://ai-content-guardian-v2.streamlit.app/)

<div align="center">
  
🛡️ **AI-Powered Content Moderation System**

*Detect harmful content in text and images using state-of-the-art deep learning models*

**[🚀 Try Live Demo](https://ai-content-guardian-v2.streamlit.app/)** • [Features](#-features) • [Installation](#-installation) • [API Docs](#-api-documentation) • [Contributing](#-contributing)

</div>

---

## 🎮 Live Demo

**[📱 Interactive Demo: https://ai-content-guardian-v2.streamlit.app/](https://ai-content-guardian-v2.streamlit.app/)**

Try the AI Content Guardian with:
- ✅ **Text Analysis** - Test harmful content detection
- ✅ **Image Analysis** - Upload and analyze images  
- ✅ **Multi-Modal** - Combine text + image analysis
- ✅ **Real-time Results** - Instant feedback and confidence scores

---

## 📖 Overview

A comprehensive AI system that analyzes both text and images to detect harmful content including hate speech, violence, and inappropriate imagery for social media platforms. Built with modern ML frameworks and production-ready architecture.

### 🎯 **Key Capabilities:**
- **Real-time Content Analysis** - Process text and images in milliseconds
- **Multi-Modal Intelligence** - Combine text and visual analysis for better accuracy  
- **Production-Ready** - Scalable API with comprehensive monitoring
- **Interactive Interface** - User-friendly web interface for testing and demos

## 🚀 Features

- **Multi-Modal Analysis**: Combines text and image analysis for comprehensive content moderation
- **Advanced AI Models**: Uses BERT for text analysis and Vision Transformers for image analysis
- **Production-Ready API**: FastAPI backend with comprehensive endpoints
- **Interactive Web Interface**: Streamlit frontend for easy testing and demonstration
- **Comprehensive Evaluation**: Detailed metrics, bias analysis, and model interpretability
- **Scalable Architecture**: Designed for production deployment with monitoring

## 📁 Project Structure

```
├── data_collection/          # Data collection scripts
├── preprocessing/           # Data preprocessing pipelines
├── models/                 # Model architectures and training
├── training/               # Training scripts and utilities
├── api/                    # FastAPI backend
├── frontend/               # Streamlit web interface
├── interpretability/       # Model explanation tools
├── monitoring/             # Performance monitoring
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── configs/                # Configuration files
├── scripts/                # Utility scripts
└── notebooks/              # Jupyter notebooks for EDA
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended)
- Git

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd "Multi-Modal Content Moderation System for Social Media Platforms"
```

2. Create virtual environment:
```bash
python -m venv multimodal_env
# Windows
multimodal_env\Scripts\activate
# Linux/Mac
source multimodal_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models and data:
```bash
python scripts/setup_models.py
```

## 🚀 Quick Start

### 1. Data Collection
```bash
python data_collection/collect_datasets.py
```

### 2. Training Models
```bash
# Train individual models
python training/train_text_model.py
python training/train_image_model.py

# Train multi-modal fusion model
python training/train_multimodal_model.py
```

### 3. Start API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Launch Web Interface
```bash
streamlit run frontend/app.py
```

---

## 🚀 Deployment

### Live Demo
The system is deployed and accessible at:
**[https://ai-content-guardian-v2.streamlit.app/](https://ai-content-guardian-v2.streamlit.app/)**

### Deployment Options

#### Streamlit Cloud (Frontend Only)
1. Fork this repository
2. Connect to [Streamlit Cloud](https://share.streamlit.io/)
3. Set main file path to: `streamlit_app.py`
4. Deploy with minimal dependencies

#### Render/Railway (Full Stack)
1. Deploy API backend using `render.yaml` configuration
2. Update frontend API endpoint
3. Deploy frontend separately

#### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d
```

#### Local Development
```bash
# Start API server
python api/main.py

# Start frontend (separate terminal)
streamlit run streamlit_app.py
```

## 📊 Model Performance

| Model Type | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Text Only  | 87.3%    | 86.1%     | 88.2%  | 87.1%    |
| Image Only | 84.7%    | 83.9%     | 85.1%  | 84.5%    |
| Multi-Modal| 91.2%    | 90.8%     | 91.7%  | 91.2%    |

## 🔧 Configuration

Configure the system by editing `configs/config.yaml`:

```yaml
model:
  text_model: "bert-base-uncased"
  image_model: "google/vit-base-patch16-224"
  fusion_strategy: "attention"

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 10

api:
  host: "0.0.0.0"
  port: 8000
  max_content_length: 10485760  # 10MB
```

## 📚 API Documentation

The API provides comprehensive endpoints for content moderation:

- `POST /predict` - Analyze single text/image content
- `POST /batch_predict` - Batch analysis for multiple items
- `GET /health` - System health check
- `GET /metrics` - Model performance metrics

Full API documentation available at `http://localhost:8000/docs` when running.

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src/
```

## 📈 Monitoring

The system includes comprehensive monitoring:
- Real-time performance metrics
- Model drift detection
- Bias monitoring across demographic groups
- Latency and throughput tracking

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for providing pre-trained models
- PyTorch and TensorFlow communities
- Research papers and datasets used in this project

## 📞 Contact

For questions and support, please open an issue or contact the development team.

---

**Note**: This system is designed for research and educational purposes. Ensure compliance with relevant laws and platform policies when deploying for production use.
