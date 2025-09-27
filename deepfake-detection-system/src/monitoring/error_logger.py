"""
Simple error tracking system
"""

import json
from datetime import datetime
from pathlib import Path

class ErrorTracker:
    """Simple error tracking"""
    
    def __init__(self, error_file="errors.json"):
        self.error_file = Path(error_file)
        self.errors = self._load_errors()
    
    def _load_errors(self):
        """Load existing errors"""
        if self.error_file.exists():
            try:
                with open(self.error_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_errors(self):
        """Save errors to file"""
        with open(self.error_file, 'w') as f:
            json.dump(self.errors[-50:], f, indent=2)  # Keep last 50 errors
    
    def log_error(self, error_type, message, context=None):
        """Log an error"""
        error_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': error_type,
            'message': str(message),
            'context': context or {}
        }
        
        self.errors.append(error_entry)
        self._save_errors()
        
        print(f"ERROR [{error_type}]: {message}")
    
    def get_error_summary(self):
        """Get error summary"""
        if not self.errors:
            return {'total_errors': 0, 'error_types': {}, 'recent_errors': []}
        
        error_types = {}
        for error in self.errors:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.errors),
            'error_types': error_types,
            'recent_errors': self.errors[-5:]  # Last 5 errors
        }

# Global error tracker
error_tracker = ErrorTracker()