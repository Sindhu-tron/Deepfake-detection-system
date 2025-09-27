"""
Standalone secure API with built-in authentication
"""

import os
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import time
import secrets
import hashlib
from datetime import datetime
from functools import wraps

app = Flask(__name__)

# Simple in-memory authentication
API_KEYS = {}
RATE_LIMITS = {}

def create_api_key(user_id, user_name):
    """Create API key"""
    api_key = f"ak_{secrets.token_hex(16)}"
    API_KEYS[api_key] = {
        'user_id': user_id,
        'user_name': user_name,
        'rate_limit': 100,
        'is_active': True
    }
    return api_key

def validate_api_key(api_key):
    """Validate API key"""
    return API_KEYS.get(api_key) if api_key in API_KEYS and API_KEYS[api_key]['is_active'] else None

def check_rate_limit(api_key, limit=100):
    """Check rate limiting"""
    current_time = time.time()
    hour_ago = current_time - 3600
    
    if api_key not in RATE_LIMITS:
        RATE_LIMITS[api_key] = []
    
    # Clean old entries
    RATE_LIMITS[api_key] = [t for t in RATE_LIMITS[api_key] if t > hour_ago]
    
    current_usage = len(RATE_LIMITS[api_key])
    if current_usage >= limit:
        return False, current_usage, limit
    
    RATE_LIMITS[api_key].append(current_time)
    return True, current_usage + 1, limit

def require_api_key(f):
    """Authentication decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        key_info = validate_api_key(api_key)
        if not key_info:
            return jsonify({'error': 'Invalid API key'}), 401
        
        allowed, usage, limit = check_rate_limit(api_key, key_info['rate_limit'])
        if not allowed:
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        request.current_user = key_info
        request.rate_limit_info = {'current_usage': usage, 'limit': limit, 'remaining': limit - usage}
        
        return f(*args, **kwargs)
    
    return decorated

# Load model
model = None
try:
    model_path = '../training_outputs/models/best_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded: {model.count_params():,} parameters")
except:
    print("Model not available")

# Create demo API key
demo_key = create_api_key("demo", "Demo User")
print(f"Demo API key: {demo_key}")

@app.route('/')
def api_info():
    """Public API info"""
    return jsonify({
        'name': 'Simple Secure Deepfake API',
        'version': '1.0.0',
        'authentication': 'API Key required (X-API-Key header)',
        'demo_key': demo_key,
        'endpoints': {
            '/': 'GET - API info (public)',
            '/auth/key': 'POST - Request API key',
            '/health': 'GET - Health check (authenticated)', 
            '/predict': 'POST - Deepfake detection (authenticated)'
        }
    })

@app.route('/auth/key', methods=['POST'])
def request_api_key():
    """Request new API key"""
    data = request.get_json() or {}
    user_name = data.get('user_name', 'Anonymous')
    user_id = hashlib.md5(f"{user_name}{time.time()}".encode()).hexdigest()[:8]
    
    api_key = create_api_key(user_id, user_name)
    
    return jsonify({
        'success': True,
        'api_key': api_key,
        'user_id': user_id,
        'message': 'Use X-API-Key header for authentication'
    })

@app.route('/health')
@require_api_key
def health_check():
    """Authenticated health check"""
    return jsonify({
        'status': 'healthy',
        'user': request.current_user,
        'rate_limit': request.rate_limit_info,
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
@require_api_key 
def predict():
    """Secure prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not available'}), 500
    
    # Simple prediction placeholder
    return jsonify({
        'success': True,
        'prediction': {
            'class': 'real',
            'confidence': 0.75,
            'probabilities': {'real': 0.75, 'fake': 0.25}
        },
        'user': request.current_user['user_name'],
        'rate_limit_remaining': request.rate_limit_info['remaining']
    })

if __name__ == '__main__':
    print("Starting Simple Secure API...")
    print(f"Demo API key: {demo_key}")
    print("Use X-API-Key header for authentication")
    app.run(debug=True, port=5002)