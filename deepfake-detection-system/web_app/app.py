"""
Flask web application for deepfake detection
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import sys

# Add project root to path
sys.path.append('..')
sys.path.append('../src')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detection-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variable
model = None

class DeepfakePredictor:
    """Handle model loading and predictions"""
    
    def __init__(self, model_path='../training_outputs/models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = ['fake', 'real']
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
                return True
            else:
                print(f"Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Resize to model input size
            image_resized = cv2.resize(image_rgb, (224, 224))
            
            # Normalize
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_normalized = (image_normalized - mean) / std
            
            # Add batch dimension
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            return image_batch
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return {'error': 'Image preprocessing failed'}
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Extract probabilities
            fake_prob = float(prediction[0][0])
            real_prob = float(prediction[0][1])
            
            # Determine class
            predicted_class = 'real' if real_prob > fake_prob else 'fake'
            confidence = max(real_prob, fake_prob)
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'raw_prediction': prediction.tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}

# Initialize predictor
predictor = DeepfakePredictor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Read image
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image file'})
            
            # Make prediction
            result = predictor.predict(image)
            
            # Convert image to base64 for display
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Add image data to result
            result['image'] = image_base64
            result['filename'] = secure_filename(file.filename)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = 'loaded' if predictor.model is not None else 'not loaded'
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'version': '1.0.0'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    print("Starting Deepfake Detection Web App...")
    print("Model status:", "Loaded" if predictor.model else "Not loaded")
    app.run(debug=True, host='0.0.0.0', port=5000)