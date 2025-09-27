"""
Simple file-based analytics system
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class SimpleAnalytics:
    """File-based analytics"""
    
    def __init__(self, data_file="analytics_data.json"):
        self.data_file = Path(data_file)
        self.data = self._load_data()
    
    def _load_data(self):
        if self.data_file.exists():
            try:
                with open(self.data_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"predictions": [], "api_usage": [], "created_at": datetime.utcnow().isoformat()}
    
    def _save_data(self):
        with open(self.data_file, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def log_prediction(self, prediction_class, confidence, user_id=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_class": prediction_class,
            "confidence": confidence,
            "user_id": user_id or "anonymous"
        }
        self.data["predictions"].append(entry)
        self._save_data()
        print(f"Analytics: Logged {prediction_class} prediction (confidence: {confidence:.3f})")
    
    def log_api_call(self, endpoint, method, status_code, response_time_ms, user_id=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "user_id": user_id or "anonymous"
        }
        self.data["api_usage"].append(entry)
        self._save_data()
    
    def get_analytics_summary(self):
        total_predictions = len(self.data["predictions"])
        fake_predictions = len([p for p in self.data["predictions"] if p["prediction_class"] == "fake"])
        avg_confidence = sum(p["confidence"] for p in self.data["predictions"]) / max(1, total_predictions)
        
        return {
            "total_predictions": total_predictions,
            "fake_predictions": fake_predictions,
            "real_predictions": total_predictions - fake_predictions,
            "fake_percentage": (fake_predictions / max(1, total_predictions)) * 100,
            "average_confidence": avg_confidence,
            "total_api_calls": len(self.data["api_usage"]),
            "last_updated": datetime.utcnow().isoformat()
        }

analytics = SimpleAnalytics()
