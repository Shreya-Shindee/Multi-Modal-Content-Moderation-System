# Dockerfile for Multi-Modal Content Moderation System

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/processed data/images checkpoints logs

# Set environment variables
ENV PYTHONPATH=/app
ENV DEVICE=cpu

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    echo "Starting API server..."\n\
    uvicorn api.main:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "frontend" ]; then\n\
    echo "Starting Streamlit frontend..."\n\
    streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0\n\
elif [ "$1" = "setup" ]; then\n\
    echo "Running setup..."\n\
    python scripts/setup_models.py\n\
else\n\
    echo "Usage: $0 {api|frontend|setup}"\n\
    echo "  api      - Start the FastAPI backend"\n\
    echo "  frontend - Start the Streamlit frontend"\n\
    echo "  setup    - Run initial setup"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
