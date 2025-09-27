"""
Secure deepfake API with authentication and rate limiting
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

# Import security
from security.auth import require_api_key, auth_manager

app = Flask(__name__)
CORS(app)

# Security headers middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Add rate limit headers if available
    if hasattr(request, 'rate_limit_info'):
        info = request.rate_limit_info
        response.headers['X-RateLimit-Limit'] = str(info['limit'])
        response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
        response.headers['X-RateLimit-Used'] = str(info['current_usage'])
    
    return response

class SecureDeepfakeAPI:
    """Secure API with authentication"""
    
    def __init__(self, model_path='../training_outputs/models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Track usage statistics
        self.total_requests = 0
        self.successful_predictions = 0
        self.failed_requests = 0
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Secure API: Model loaded from {self.model_path}")
                return True
            else:
                print(f"Secure API: Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Secure API: Error loading model: {e}")
            return False
    
    def predict(self, image):
        """Make secure prediction"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            if processed_image is None:
                return {'error': 'Image preprocessing failed'}
            
            # Make prediction
            start_time = time.time()
            prediction = self.model.predict(processed_image, verbose=0)
            processing_time = (time.time() - start_time) * 1000
            
            # Extract results
            fake_prob = float(prediction[0][0])
            real_prob = float(prediction[0][1])
            predicted_class = 'real' if real_prob > fake_prob else 'fake'
            confidence = max(real_prob, fake_prob)
            
            self.successful_predictions += 1
            
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
                'processing_time_ms': processing_time,
                'model_info': {
                    'version': '1.0',
                    'input_shape': [224, 224, 3]
                }
            }
            
        except Exception as e:
            self.failed_requests += 1
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _preprocess_image(self, image):
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
            return None
    
    def get_stats(self):
        """Get API usage statistics"""
        return {
            'total_requests': self.total_requests,
            'successful_predictions': self.successful_predictions,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_predictions / max(1, self.total_requests)) * 100
        }

# Initialize secure API
api = SecureDeepfakeAPI()

# Request tracking
@app.before_request
def track_request():
    api.total_requests += 1

def decode_base64_image(base64_string):
    """Decode base64 image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_data))
        image_array = np.array(pil_image)
        
        return image_array
        
    except Exception as e:
        return None

@app.route('/', methods=['GET'])
def api_info():
    """Public API information"""
    return jsonify({
        'name': 'Secure Deepfake Detection API',
        'version': '2.0.0',
        'description': 'Authenticated REST API for deepfake detection',
        'authentication': {
            'type': 'API Key',
            'header': 'X-API-Key or Authorization: Bearer <key>',
            'rate_limit': 'Per API key limits apply'
        },
        'endpoints': {
            '/': 'GET - API information (public)',
            '/auth/key': 'POST - Request API key (public)',
            '/health': 'GET - Health check (authenticated)',
            '/predict': 'POST - Detect deepfakes (authenticated)',
            '/stats': 'GET - Usage statistics (authenticated)'
        },
        'model_status': 'loaded' if api.model is not None else 'not loaded'
    })

@app.route('/auth/key', methods=['POST'])
def request_api_key():
    """Request new API key (simplified - in production would require proper registration)"""
    try:
        data = request.get_json() or {}
        user_name = data.get('user_name', 'Anonymous User')
        user_email = data.get('user_email', 'unknown@example.com')
        
        # Generate user ID
        user_id = hashlib.md5(f"{user_email}{time.time()}".encode()).hexdigest()[:8]
        
        # Create API key
        api_key = auth_manager.create_api_key(user_id, user_name)
        
        return jsonify({
            'success': True,
            'api_key': api_key,
            'user_id': user_id,
            'rate_limit': 100,
            'message': 'API key created successfully. Include in X-API-Key header.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
@require_api_key
def health_check():
    """Authenticated health check"""
    user_info = request.current_user
    rate_info = request.rate_limit_info
    
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'user': {
            'user_id': user_info['user_id'],
            'user_name': user_info['user_name']
        },
        'rate_limit': rate_info,
        'model_loaded': api.model is not None,
        'version': '2.0.0'
    })

@app.route('/predict', methods=['POST'])
@require_api_key
def predict_image():
    """Secure prediction endpoint"""
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
        
        # Make prediction
        result = api.predict(image)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Add user context to response
        result['user_info'] = {
            'user_id': request.current_user['user_id'],
            'rate_limit_remaining': request.rate_limit_info['remaining']
        }
        
        return jsonify(result)
        
    except Exception as e:
        api.failed_requests += 1
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/stats', methods=['GET'])
@require_api_key
def get_stats():
    """Get API usage statistics"""
    try:
        stats = api.get_stats()
        stats['user_info'] = request.current_user
        stats['timestamp'] = time.time()
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Secure Deepfake Detection API...")
    print(f"Model status: {'Loaded' if api.model else 'Not loaded'}")
    print("Security features:")
    print("  - API Key authentication required")
    print("  - Rate limiting per API key")
    print("  - Security headers on all responses")
    print("  - Input validation and sanitization")
    print("\nTo get an API key:")
    print("  POST /auth/key with {'user_name': 'Your Name'}")
    print("\nThen use:")
    print("  X-API-Key: <your-key> header on requests")
    
    app.run(debug=False, host='0.0.0.0', port=5002)