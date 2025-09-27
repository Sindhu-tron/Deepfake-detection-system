"""
Standalone monitoring dashboard
"""
from flask import Flask, render_template, jsonify
import psutil
import time
from datetime import datetime

app = Flask(__name__)
metrics_history = []

@app.route('/dashboard')
def monitoring_dashboard():
    return render_template('monitoring_dashboard.html')

@app.route('/api/monitoring/metrics')
def get_current_metrics():
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        current_metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_usage': disk.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'memory_total_gb': round(memory.total / (1024**3), 2),
            'timestamp': time.time()
        }
        
        global metrics_history
        metrics_history.append(current_metrics)
        if len(metrics_history) > 20:
            metrics_history.pop(0)
        
        return jsonify({'success': True, 'metrics': current_metrics})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/monitoring/health')
def get_health_status():
    try:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        health_score = 100
        health_issues = []
        
        if cpu_percent > 80:
            health_score -= 20
            health_issues.append(f"High CPU: {cpu_percent:.1f}%")
        
        if memory_percent > 85:
            health_score -= 25
            health_issues.append(f"High Memory: {memory_percent:.1f}%")
        
        status = "healthy" if health_score >= 80 else ("degraded" if health_score >= 60 else "unhealthy")
        
        health_report = {
            'status': status,
            'health_score': max(0, health_score),
            'health_issues': health_issues,
            'current_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_usage': disk_percent
            },
            'error_summary': {'total_errors': 0, 'errors_last_hour': 0, 'errors_last_day': 0},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({'success': True, 'health': health_report})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("Starting Monitoring Dashboard...")
    print("Available at: http://localhost:5003/dashboard")
    app.run(debug=True, host='0.0.0.0', port=5003)