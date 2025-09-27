"""
REST API for deepfake detection system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import base64
import io
from PIL import Image
import sys
import os

# Add project paths
sys.path.append('..')
sys.path.append('../src')

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

class DeepfakeAPI:
    """API wrapper for deepfake detection model"""
    
    def __init__(self, model_path='../training_outputs/models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"API: Model loaded from {self.model_path}")
                return True
            else:
                print(f"API: Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"API: Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        try:
            # Ensure RGB format
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:  # BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224
            image_resized = cv2.resize(image, (224, 224))
            
            # Normalize to [0,1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_normalized = (image_normalized - mean) / std
            
            # Add batch dimension
            return np.expand_dims(image_normalized, axis=0)
            
        except Exception as e:
            print(f"API: Preprocessing error: {e}")
            return None
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return {'error': 'Image preprocessing failed'}
            
            # Get prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Extract probabilities
            fake_prob = float(prediction[0][0])
            real_prob = float(prediction[0][1])
            
            # Determine class and confidence
            predicted_class = 'real' if real_prob > fake_prob else 'fake'
            confidence = max(real_prob, fake_prob)
            
            return {
                'success': True,
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'probabilities': {
                        'real': real_prob,
                        'fake': fake_prob
                    }
                },
                'model_info': {
                    'version': '1.0',
                    'input_shape': [224, 224, 3]
                }
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

# Initialize API
api = DeepfakeAPI()

def decode_base64_image(base64_string):
    """Decode base64 image string"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        return image_array
        
    except Exception as e:
        print(f"Base64 decode error: {e}")
        return None

@app.route('/', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Deepfake Detection API',
        'version': '1.0.0',
        'description': 'REST API for detecting deepfakes in images',
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Detect deepfakes in image',
            '/batch': 'POST - Batch prediction on multiple images'
        },
        'model_status': 'loaded' if api.model is not None else 'not loaded'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = api.model is not None
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'timestamp': tf.timestamp().numpy() if model_loaded else None,
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict_image():
    """Single image prediction endpoint"""
    try:
        # Check if model is loaded
        if api.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Handle different input formats
        if request.content_type.startswith('multipart/form-data'):
            # File upload
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read and decode image
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif request.content_type == 'application/json':
            # Base64 encoded image
            data = request.get_json()
            
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            
            image = decode_base64_image(data['image'])
            
        else:
            return jsonify({'error': 'Unsupported content type'}), 400
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Make prediction
        result = api.predict(image)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if api.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400
        
        if not isinstance(data['images'], list):
            return jsonify({'error': 'Images must be provided as a list'}), 400
        
        if len(data['images']) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum batch size is 10 images'}), 400
        
        results = []
        
        for i, image_data in enumerate(data['images']):
            try:
                image = decode_base64_image(image_data)
                if image is None:
                    results.append({
                        'index': i,
                        'error': 'Invalid image data'
                    })
                    continue
                
                result = api.predict(image)
                result['index'] = i
                results.append(result)
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': f'Processing error: {str(e)}'
                })
        
        return jsonify({
            'success': True,
            'batch_size': len(data['images']),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch processing error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Deepfake Detection API...")
    print(f"Model status: {'Loaded' if api.model else 'Not loaded'}")
    print("Available endpoints:")
    print("  GET  /       - API information")
    print("  GET  /health - Health check")
    print("  POST /predict - Single image prediction")
    print("  POST /batch  - Batch image prediction")
    
    app.run(debug=True, host='0.0.0.0', port=5001)