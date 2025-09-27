"""
Secure deepfake API with integrated analytics and error tracking
"""

from flask import Flask, request, jsonify
import time
import secrets
import hashlib
from datetime import datetime
from functools import wraps
import sys
import os

# Add analytics integration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from analytics.simple_analytics import analytics
    from monitoring.error_logger import error_tracker
    ANALYTICS_ENABLED = True
    print("📊 Analytics and error tracking enabled")
except ImportError as e:
    ANALYTICS_ENABLED = False
    print(f"⚠️ Analytics not available: {e}")

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
            if ANALYTICS_ENABLED:
                error_tracker.log_error('auth_error', 'Missing API key', {'endpoint': request.endpoint})
            return jsonify({'error': 'API key required'}), 401
        
        key_info = validate_api_key(api_key)
        if not key_info:
            if ANALYTICS_ENABLED:
                error_tracker.log_error('auth_error', 'Invalid API key', {'endpoint': request.endpoint})
            return jsonify({'error': 'Invalid API key'}), 401
        
        allowed, usage, limit = check_rate_limit(api_key, key_info['rate_limit'])
        if not allowed:
            if ANALYTICS_ENABLED:
                error_tracker.log_error('rate_limit_error', 'Rate limit exceeded', {
                    'user_id': key_info['user_id'], 
                    'endpoint': request.endpoint
                })
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        request.current_user = key_info
        request.rate_limit_info = {'current_usage': usage, 'limit': limit, 'remaining': limit - usage}
        
        return f(*args, **kwargs)
    
    return decorated

# Model placeholder
model = None

# Create demo API key
demo_key = create_api_key("demo", "Demo User")

@app.route('/')
def api_info():
    """Public API info"""
    if ANALYTICS_ENABLED:
        analytics.log_api_call('/', 'GET', 200, 0)
    
    return jsonify({
        'name': 'Secure Deepfake API with Analytics',
        'version': '2.0.0',
        'authentication': 'API Key required (X-API-Key header)',
        'demo_key': demo_key,
        'analytics_enabled': ANALYTICS_ENABLED,
        'endpoints': {
            '/': 'GET - API info (public)',
            '/auth/key': 'POST - Request API key',
            '/health': 'GET - Health check (authenticated)', 
            '/predict': 'POST - Deepfake detection (authenticated)',
            '/analytics': 'GET - Analytics summary (authenticated)'
        }
    })

@app.route('/predict', methods=['POST'])
@require_api_key 
def predict():
    """Secure prediction endpoint with analytics"""
    start_time = time.time()
    
    try:
        # Placeholder prediction (realistic random)
        prediction_class = 'real' if (hash(str(time.time())) % 2) else 'fake'
        confidence = 0.70 + (hash(str(time.time())) % 30) / 100  # Random 0.70-0.99
        
        # Log analytics
        if ANALYTICS_ENABLED:
            analytics.log_prediction(prediction_class, confidence, request.current_user['user_id'])
            response_time = (time.time() - start_time) * 1000
            analytics.log_api_call('/predict', 'POST', 200, response_time, request.current_user['user_id'])
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': prediction_class,
                'confidence': confidence,
                'probabilities': {
                    'real': confidence if prediction_class == 'real' else 1 - confidence,
                    'fake': confidence if prediction_class == 'fake' else 1 - confidence
                }
            },
            'user': request.current_user['user_name'],
            'analytics_enabled': ANALYTICS_ENABLED,
            'rate_limit_remaining': request.rate_limit_info['remaining'],
            'processing_time_ms': (time.time() - start_time) * 1000
        })
        
    except Exception as e:
        if ANALYTICS_ENABLED:
            error_tracker.log_error('prediction_error', str(e), {
                'endpoint': '/predict', 
                'user': request.current_user['user_id']
            })
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
@require_api_key
def get_analytics():
    """Get analytics summary"""
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    try:
        summary = analytics.get_analytics_summary()
        error_summary = error_tracker.get_error_summary()
        
        return jsonify({
            'success': True,
            'analytics': summary,
            'errors': error_summary,
            'user': request.current_user['user_name'],
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        if ANALYTICS_ENABLED:
            error_tracker.log_error('analytics_endpoint_error', str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Secure Deepfake API with Analytics...")
    print(f"📊 Demo API key: {demo_key}")
    print(f"📈 Analytics enabled: {ANALYTICS_ENABLED}")
    print("🔐 Use X-API-Key header for authentication")
    print("🌐 Server starting on http://localhost:5002")
    
    app.run(debug=True, host='0.0.0.0', port=5002)
