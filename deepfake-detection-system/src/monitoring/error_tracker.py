"""
Simple error tracking system
"""

import time
import psutil
from collections import defaultdict, deque
from datetime import datetime, timedelta

class SystemMonitor:
    """Simple system monitor for dashboard"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=100)
        self.error_count = 0
    
    def record_metrics(self, metrics):
        """Record system metrics"""
        metric_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_percent': metrics.get('cpu_percent', 0),
            'memory_percent': metrics.get('memory_percent', 0),
            'disk_usage': metrics.get('disk_usage', 0)
        }
        
        self.metrics_history.append(metric_entry)
    
    def get_health_report(self):
        """Get simple health report"""
        return {
            'status': 'healthy',
            'health_score': 85,
            'health_issues': [],
            'current_metrics': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'error_summary': {
                'total_errors': self.error_count,
                'errors_last_hour': 0,
                'errors_last_day': 0
            },
            'timestamp': datetime.utcnow().isoformat()
        }

# Global monitor instance
system_monitor = None

def get_system_monitor():
    """Get global system monitor instance"""
    global system_monitor
    if system_monitor is None:
        system_monitor = SystemMonitor()
    return system_monitor