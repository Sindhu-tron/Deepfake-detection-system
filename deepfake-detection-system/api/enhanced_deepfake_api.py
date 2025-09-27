"""
Enhanced REST API with database integration
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
import time
import hashlib

# Add project paths
sys.path.append('..')
sys.path.append('../src')

# Import database models
from database.models import get_db_manager

app = Flask(__name__)
CORS(app)

class EnhancedDeepfakeAPI:
    """Enhanced API with database logging"""
    
    def __init__(self, model_path='../training_outputs/models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.db = get_db_manager()
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Enhanced API: Model loaded from {self.model_path}")
                return True
            else:
                print(f"Enhanced API: Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Enhanced API: Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        try:
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:  # BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image_resized = cv2.resize(image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_normalized = (image_normalized - mean) / std
            
            return np.expand_dims(image_normalized, axis=0)
            
        except Exception as e:
            print(f"Enhanced API: Preprocessing error: {e}")
            return None
    
    def predict(self, image, user_id=None, ip_address=None, user_agent=None):
        """Make prediction with database logging"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        start_time = time.time()
        
        try:
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return {'error': 'Image preprocessing failed'}
            
            # Get prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Extract probabilities
            fake_prob = float(prediction[0][0])
            real_prob = float(prediction[0][1])
            
            # Determine class and confidence
            predicted_class = 'real' if real_prob > fake_prob else 'fake'
            confidence = max(real_prob, fake_prob)
            
            result = {
                'success': True,
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'probabilities': {
                        'real': real_prob,
                        'fake': fake_prob
                    }
                },
                'processing_time_ms': processing_time,
                'model_info': {
                    'version': '1.0',
                    'input_shape': [224, 224, 3]
                }
            }
            
            # Log to database
            try:
                self.db.log_prediction(result, user_id, ip_address, user_agent)
            except Exception as e:
                print(f"Database logging error: {e}")
                # Don't fail the request if logging fails
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

# Initialize enhanced API
api = EnhancedDeepfakeAPI()

# Middleware for API usage logging
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        response_time = (time.time() - request.start_time) * 1000
        
        # Log API usage
        try:
            api.db.log_api_usage(
                endpoint=request.endpoint or request.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=response_time,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
        except Exception as e:
            print(f"API usage logging error: {e}")
    
    return response

def decode_base64_image(base64_string):
    """Decode base64 image string"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_data))
        image_array = np.array(pil_image)
        
        return image_array
        
    except Exception as e:
        print(f"Base64 decode error: {e}")
        return None

@app.route('/', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Enhanced Deepfake Detection API',
        'version': '2.0.0',
        'description': 'REST API for detecting deepfakes with database logging',
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Detect deepfakes in image',
            '/analytics': 'GET - Get usage analytics'
        },
        'model_status': 'loaded' if api.model is not None else 'not loaded',
        'database_status': 'connected' if api.db else 'not connected'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with database status"""
    model_loaded = api.model is not None
    
    # Test database connection
    try:
        analytics = api.db.get_analytics_summary()
        db_status = 'connected'
    except Exception as e:
        db_status = f'error: {str(e)}'
        analytics = {}
    
    return jsonify({
        'status': 'healthy' if model_loaded and db_status == 'connected' else 'degraded',
        'model_loaded': model_loaded,
        'database_status': db_status,
        'total_predictions': analytics.get('total_predictions', 0),
        'timestamp': time.time(),
        'version': '2.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict_image():
    """Enhanced prediction endpoint with logging"""
    try:
        if api.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Handle different input formats
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        elif request.content_type == 'application/json':
            data = request.get_json()
            
            if 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            
            image = decode_base64_image(data['image'])
            
        else:
            return jsonify({'error': 'Unsupported content type'}), 400
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Make prediction with logging
        result = api.predict(
            image,
            user_id=None,  # TODO: Add user authentication
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get system analytics"""
    try:
        analytics = api.db.get_analytics_summary()
        return jsonify({
            'success': True,
            'analytics': analytics,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': f'Analytics error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Enhanced Deepfake Detection API...")
    print(f"Model status: {'Loaded' if api.model else 'Not loaded'}")
    print(f"Database status: {'Connected' if api.db else 'Not connected'}")
    print("Available endpoints:")
    print("  GET  /         - API information")
    print("  GET  /health   - Health check with database status")
    print("  POST /predict  - Single image prediction with logging")
    print("  GET  /analytics - Usage analytics")
    
    app.run(debug=True, host='0.0.0.0', port=5001)