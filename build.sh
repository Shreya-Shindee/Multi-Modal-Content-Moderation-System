#!/bin/bash

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download and cache models (optional)
echo "Setting up models..."
python -c "
from transformers import AutoTokenizer, AutoModel
print('Downloading BERT model...')
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased')
print('Models cached successfully!')
"

echo "Build completed successfully!"
