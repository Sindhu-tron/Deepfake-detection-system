"""
Simple authentication system
"""

import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

class AuthManager:
    """Simple API key authentication"""
    
    def __init__(self):
        self.api_keys = {}
        self.rate_limits = {}
        
        # Create demo API key
        demo_key = self.create_api_key("demo_user", "Demo User")
        print(f"Demo API key created: {demo_key}")
    
    def create_api_key(self, user_id, user_name, rate_limit=100):
        """Create new API key"""
        api_key = f"ak_{secrets.token_hex(16)}"
        
        self.api_keys[api_key] = {
            'user_id': user_id,
            'user_name': user_name,
            'created_at': datetime.utcnow().isoformat(),
            'rate_limit': rate_limit,
            'is_active': True
        }
        
        return api_key
    
    def validate_api_key(self, api_key):
        """Validate API key"""
        if not api_key or api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        if not key_info['is_active']:
            return None
        
        return key_info
    
    def check_rate_limit(self, api_key, limit_per_hour=100):
        """Simple rate limiting"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = []
        
        # Clean old entries
        self.rate_limits[api_key] = [
            timestamp for timestamp in self.rate_limits[api_key]
            if timestamp > hour_ago
        ]
        
        current_usage = len(self.rate_limits[api_key])
        
        if current_usage >= limit_per_hour:
            return False, current_usage, limit_per_hour
        
        self.rate_limits[api_key].append(current_time)
        return True, current_usage + 1, limit_per_hour

# Global auth manager
auth_manager = AuthManager()

def require_api_key(f):
    """Decorator for API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')
        
        if api_key and api_key.startswith('Bearer '):
            api_key = api_key[7:]
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        key_info = auth_manager.validate_api_key(api_key)
        if not key_info:
            return jsonify({'error': 'Invalid API key'}), 401
        
        allowed, current_usage, limit = auth_manager.check_rate_limit(
            api_key, key_info['rate_limit']
        )
        
        if not allowed:
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        request.current_user = key_info
        request.rate_limit_info = {
            'current_usage': current_usage,
            'limit': limit,
            'remaining': limit - current_usage
        }
        
        return f(*args, **kwargs)
    
    return decorated_function