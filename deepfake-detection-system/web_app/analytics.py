"""
Analytics dashboard for deepfake detection system
"""

from flask import Flask, render_template, jsonify, request
import sys
import json
from datetime import datetime, timedelta

sys.path.append('..')
sys.path.append('../src')

from database.models import get_db_manager

app = Flask(__name__)

@app.route('/analytics')
def analytics_dashboard():
    """Analytics dashboard page"""
    return render_template('analytics.html')

@app.route('/api/analytics/summary')
def analytics_summary():
    """Get analytics summary data"""
    try:
        db = get_db_manager()
        analytics = db.get_analytics_summary()
        
        return jsonify({
            'success': True,
            'data': analytics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/analytics/predictions/timeline')
def predictions_timeline():
    """Get predictions timeline data"""
    try:
        db = get_db_manager()
        session = db.get_session()
        
        # Get predictions from last 7 days
        cutoff = datetime.utcnow() - timedelta(days=7)
        
        from database.models import Prediction
        predictions = session.query(Prediction).filter(
            Prediction.created_at >= cutoff
        ).order_by(Prediction.created_at).all()
        
        # Group by day and class
        daily_stats = {}
        for pred in predictions:
            day = pred.created_at.strftime('%Y-%m-%d')
            if day not in daily_stats:
                daily_stats[day] = {'real': 0, 'fake': 0}
            daily_stats[day][pred.prediction_class] += 1
        
        # Convert to chart format
        timeline_data = []
        for day, stats in sorted(daily_stats.items()):
            timeline_data.append({
                'date': day,
                'real': stats['real'],
                'fake': stats['fake'],
                'total': stats['real'] + stats['fake']
            })
        
        session.close()
        
        return jsonify({
            'success': True,
            'data': timeline_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5002)