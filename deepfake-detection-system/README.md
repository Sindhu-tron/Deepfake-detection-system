# Deepfake Detection System
#### Video demo: <https://youtu.be/GAwUhJUjxLU> 
#### Description: 

## Overview

Deepfakes represent a growing threat to information integrity, which allows malicious people to create convincing fake videos that can spread misinformation, damage reputations, and undermine trust in media. As these AI-generated forgeries become increasingly sophisticated, the need for reliable detection systems has never been more critical. 

This final project implements an end-to-end deepfake detection system that combines machine leaning with an user-friendly web interface. The system analyses the uploaded images and provides confidence-based predictions about whether the content uploaded is authentic or is artificially generated. Rather than just returning binary answers, the system provides detailed probability scores, allowing users to understand the certaintiy level of each prediction.  

The project is essential built as a comprehensive solution, emcompassing the entire machine learning pipeline from data preprocessing through model deployment, including a Flask web application, REST API, video processing capabilities, and a ready for production deployment infrastructure.  

## What this project does
The Deepfake Detection System provides an intuitive web interface where users can upload images to determine if they contian authentic or artifically generated faces. The system then processes the uploaded images through a trained neural network and returns detailed predictions including confidence scores, probability breakdowns and visual feedback. 

Users interact with the system through a web interface featuring drag and drop functionality. After uploading an image, the system displays the original image along with the prediction results, which either show the content uplaod to either be classifed as "Real" or "Fake", along with the percentage confidence scores for each probability. This type of feedback also allows the users to understand the certainty level of predictions rather than just getting a binary answer. 

The system also included API endpoints for a sort of computational / mathematical access, video processing capabilities for analysing longer content, and analytics tracking to monitor system usage and performance over time.

## Technical Structure
Essentially the structure is built around a Flask web application with multiple components working together:

**Machine Learning Pipeline:** The core detection model is a convolutional neural network with 656,834 parameters, trained on facial image data and saved as best_model.h5. The model processes 224x224 pixel images with ImageNet normalization and outputs binary classification probabilities for real vs. fake content.

**Web Application:** Built using Flask, the web interface handles file uploads up to 16MB, processes images through OpenCV and PIL, and returns JSON responses with base64-encoded images for display. The application includes proper error handling, file validation, and security measures.

**Data Processing:** Images undergo preprocessing including resizing, normalization, and tensor conversion before model inference. The system supports multiple image formats (PNG, JPG, JPEG, GIF, BMP) and includes robust error handling for corrupted or invalid files. 

**Infrastructure:** The project includes Docker containerization, database integration with SQLite, monitoring systems, and deployment configurations for both development and production environments.

## Key learning moments 

### Data Leakage Discovery
The most significant learning experience in this project occured during the model validation process. Initially, I achieved an impressive almost 97% accuracy on the test data, which almost seemed too good to be true. However, upon further investigation, I found that there was a data leakage in my train and test splits. So the same individuals appeared in both the training and testing sets, which was allowing the model to memorise the faces rather than learn just the generalisable features. 

I then created a fix_data_splits.py to try and implement a proper video-level splitting, which would ensure that no individual would appear in multiple splits. This correction resulted in although low but a more realistic 61.8% test accuracy with the AUC-ROC score of 0.863. Although this was lower than the initial results, this represents genuine model performance and taught me the importance of proper data validation in machine learning projects. 

### Production Deployment Challenges
Implementing the complete system required learning Docker, database management, and web security practices. I encountered numerous dependency conflicts and had to create comprehensive testing suites to ensure system reliability across different environments.

## File Structure and Purpose 

#### Core Application Files:

- web_app/app.py - Main Flask application with drag-drop interface, image processing, and prediction endpoints
- app.py - Alternative entry point (has template issues, superseded by web_app version)
- training_outputs/models/best_model.h5 - Trained CNN model with 656K parameters for deepfake detection

#### Data and Processing:

- fix_data_splits.py - Addresses data leakage by implementing proper train/validation/test separation at video level
- create_sample_data.py - Generates sample datasets for testing and development
- video_processor/ - Contains video analysis pipeline for frame-by-frame deepfake detection

#### Testing and Validation:

- test_complete_system.py - Comprehensive integration testing covering all system components
- test_inference.py - Model prediction testing and performance validation
- test_validation.py - Data validation and preprocessing testing
- check_system.py - System health check and dependency verification

#### Deployment and Infrastructure:

- docker-compose.yml - Development environment containerization
- docker-compose.prod.yml - Production deployment configuration
- Dockerfile.railway - Railway cloud platform deployment
- requirements.txt - Python dependency specification

#### Analytics and Monitoring:

- analytics_data.json - Usage tracking and system performance metrics
- deepfake_detection.db - SQLite database for storing predictions and user data
- errors.json - Error logging and system debugging information

## Design Decisions

**Model Architecture:** I chose a convolutional neural network over transformer-based approach due to the computational constraints and the visual nature of deepfake detection. CNNs excel at capturing spatial features in images, making them well-suited for detecting artificial artifacts in generated faces.

**Web Framework:** Flask was selected over alternatives like Django or FastAPI for its simplicity and lightweight nature. As a learning project, Flask provided better visibility into web development fundamentals without excessive abstraction. Additionally, CS50 introduced me to Flask, which was another reason why I wanted to integrate it into this system. 

**Database Choice:** SQLite was chosen for development simplicity while maintaining the ability to upgrade to PostgreSQL for production deployment. The local database approach reduces external dependencies during development.

**Confidence Scoring:** Rather than simple binary classification, I implemented probability-based confidence scoring to provide users with uncertainty estimates. This approach acknowledges that deepfake detection isn't perfect and gives users context for interpreting results.

## Challenges Faced

**Data Quality Management:** Ensuring proper dataset splits without leakage required significant time investment. I had to develop custom scripts to analyze data relationships and implement video-level splitting logic.

**Model Performance:** Balancing model accuracy with inference speed proved challenging. The final model represents a compromise between detection capability and real-time performance requirements.

**Web Interface Development:** Creating a professional-looking interface while maintaining functionality required learning CSS styling, a bit more than what was introduced during week 8 of CS50, responsive design, and JavaScript for drag-and-drop functionality.

**Deployment Complexity:** Setting up proper containerization, handling file uploads, and managing dependencies across different environments involved significant troubleshooting and iteration. 

## Results and Performance
The final system achieves 61.8% accuracy on held-out test data with an AUC-ROC score of 0.863, indicating good discriminative ability between real and fake images. While its not a  state-of-the-art system yet, these results represent honest performance after correcting data leakage issues.

**Model Metrics:**

Test Accuracy: 61.8%
AUC-ROC Score: 0.863
Model Size: 656,834 parameters
Inference Time: <2 seconds per image

**System Performance:**

Supports images up to 16MB
Processes multiple image formats
Web interface response time <3 seconds
Comprehensive error handling with 95%+ uptime in testing.   

The confidence scoring system provides probability breakdowns that help users interpret results. For example, a prediction might show 54.9% fake probability vs. 45.1% real probability, indicating uncertainty rather than false confidence.
While the accuracy could be improved with more sophisticated architectures or larger datasets, the current system successfully demonstrates the complete machine learning pipeline from data collection through deployment, achieving the educational goals of building a production-ready application that addresses a real-world problem. 


