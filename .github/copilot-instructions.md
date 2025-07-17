<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Copilot Instructions for Multi-Modal Content Moderation System

## Project Overview
This is a comprehensive multi-modal AI system for content moderation that analyzes both text and images to detect harmful content. The system uses advanced deep learning techniques including BERT for text analysis and Vision Transformers for image analysis.

## Code Style and Conventions
- Follow PEP 8 for Python code style
- Use type hints for all function parameters and return values
- Add comprehensive docstrings for all classes and functions
- Use descriptive variable names and avoid abbreviations
- Prefer composition over inheritance when designing classes

## Architecture Guidelines
- The system follows a modular architecture with clear separation of concerns
- Data processing, model training, and inference are separated into different modules
- Use dependency injection and configuration files for flexibility
- Implement proper error handling and logging throughout the system

## AI/ML Best Practices
- Always validate input data before processing
- Use proper train/validation/test splits for model evaluation
- Implement data augmentation techniques to improve model robustness
- Use appropriate evaluation metrics for multi-class classification problems
- Include bias detection and fairness analysis in model evaluation

## Security and Ethics
- Implement proper content filtering and validation
- Ensure user privacy and data protection
- Include bias monitoring and mitigation strategies
- Follow responsible AI principles in model development and deployment

## Testing Guidelines
- Write unit tests for all core functionality
- Include integration tests for API endpoints
- Test model performance on diverse datasets
- Implement performance benchmarking and monitoring

## Documentation Standards
- Maintain up-to-date API documentation
- Include code examples in documentation
- Document model architectures and training procedures
- Provide clear setup and deployment instructions

## Performance Considerations
- Optimize model inference for production use
- Implement proper caching strategies
- Use batch processing for improved throughput
- Monitor system performance and resource usage

## Dependencies and Libraries
- Primary ML frameworks: PyTorch, Transformers, TensorFlow
- Web frameworks: FastAPI, Streamlit
- Data processing: pandas, numpy, scikit-learn
- Image processing: PIL, OpenCV, albumentations
- Monitoring: wandb, tensorboard

When generating code, please consider these guidelines and the specific requirements of a content moderation system that needs to be both accurate and fair.
