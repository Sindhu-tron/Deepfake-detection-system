# Deepfake Detection System
#### Description: 
A CNN-based deepfake detection system with a Flask web interface. Upload an image, get a confidence-scored prediction of whether the face is real or AI-generated.

#### Video demo: <https://youtu.be/GAwUhJUjxLU> 

## How it works
 
1. User uploads an image through a drag-and-drop web interface
2. The image is preprocessed (resized to 224×224, ImageNet-normalised)
3. A trained CNN (656K parameters) outputs probability scores for real vs. fake
4. The interface displays the prediction with confidence percentages — not just a binary answer
The system also exposes a REST API and includes a video processing pipeline for frame-by-frame analysis.
 
## The data leakage story
 
This was the most important thing I learned on this project.
 
My initial model hit **~97% test accuracy**, which felt too good. On investigation, I found that the same individuals appeared in both training and test sets — the model was memorising faces, not learning generalisable deepfake artifacts.
 
I wrote `fix_data_splits.py` to implement proper **video-level splitting**, ensuring no individual appears across splits. After the fix, test accuracy dropped to **61.8%** with an **AUC-ROC of 0.863**. Lower numbers, but honest ones — and the AUC-ROC shows the model has real discriminative ability despite the modest accuracy.
 
The lesson: a 97% number you can't trust is worth less than a 62% number you can.
 
## Performance
 
| Metric | Value |
|---|---|
| Test accuracy (after leakage fix) | 61.8% |
| AUC-ROC | 0.863 |
| Model parameters | 656,834 |
| Inference time | <2s per image |
| Supported formats | PNG, JPG, JPEG, GIF, BMP |
| Max upload size | 16 MB |
 
## Tech stack
 
- **ML:** TensorFlow/Keras CNN, OpenCV, PIL
- **Backend:** Flask, SQLite
- **Frontend:** HTML/CSS/JS with drag-and-drop upload
- **Infrastructure:** Docker (dev + prod configs), Railway deployment config
## Project structure
 
```
├── src/                        # ML pipeline (training, preprocessing, evaluation)
├── api/                        # REST API endpoints
├── deepfake-detection-system/  # Flask web application
├── docker-compose.yml          # Development environment
├── docker-compose.prod.yml     # Production environment
├── Dockerfile.railway          # Railway cloud deployment
├── requirements.txt
├── fix_data_splits.py          # Data leakage fix (video-level splitting)
├── test_complete_system.py     # Integration tests
├── test_inference.py           # Model prediction tests
└── test_validation.py          # Data validation tests
```
 
## Design decisions
 
**CNN over transformers:** Chosen for computational efficiency and because CNNs are well-suited to detecting spatial artifacts in generated faces. A vision transformer would likely improve accuracy but at significant compute cost.
 
**Confidence scoring over binary classification:** Deepfake detection isn't perfect — returning probability breakdowns (e.g., 54.9% fake / 45.1% real) lets users understand model uncertainty rather than receiving false confidence.
 
**Flask over Django:** This is a single-purpose tool, not a content management system. Flask's minimal abstraction made the web layer transparent and easy to debug. Also: CS50 introduced me to Flask, so I wanted to push it further.
 
**SQLite:** Right for development; the Docker configs support swapping to PostgreSQL for production.
 
## Running it
 
```bash
git clone https://github.com/Sindhu-tron/Deepfake-detection-system.git
cd Deepfake-detection-system
pip install -r requirements.txt
python deepfake-detection-system/app.py
```
 
Or with Docker:
 
```bash
docker-compose up
```
 
## What I'd do differently
 
The 61.8% accuracy is honest but not competitive. With more time I'd try a pre-trained EfficientNet or ResNet backbone with transfer learning, which typically reaches 85%+ on FaceForensics++. I'd also add Grad-CAM visualisations so users can see *where* the model detects artifacts — that interpretability matters as much as raw accuracy for a tool people are supposed to trust.
 
## Context
 
Built as a final project for CS50. The full development story, including the data leakage discovery and fix, is documented in the [video walkthrough](https://youtu.be/GAwUhJUjxLU).
 
## Author
 
Sindhuja Dantuluri — [LinkedIn](https://www.linkedin.com/in/SindhujaDantuluri/)
 
## Licence
 
MIT


The confidence scoring system provides probability breakdowns that help users interpret results. For example, a prediction might show 54.9% fake probability vs. 45.1% real probability, indicating uncertainty rather than false confidence.
While the accuracy could be improved with more sophisticated architectures or larger datasets, the current system successfully demonstrates the complete machine learning pipeline from data collection through deployment, achieving the educational goals of building a production-ready application that addresses a real-world problem. 
