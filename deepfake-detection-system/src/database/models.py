"""
Database models for deepfake detection system (SQLite compatible)
"""

import os
import hashlib
import time
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user")
    video_analyses = relationship("VideoAnalysis", back_populates="user")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    image_hash = Column(String(64), nullable=False)
    prediction_class = Column(String(10), nullable=False)  # 'real' or 'fake'
    confidence = Column(Float, nullable=False)
    real_probability = Column(Float, nullable=False)
    fake_probability = Column(Float, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    model_version = Column(String(20), default='1.0')
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))  # Changed from INET to String
    user_agent = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="predictions")

class VideoAnalysis(Base):
    __tablename__ = 'video_analyses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False)
    duration_seconds = Column(Float, nullable=False)
    frames_analyzed = Column(Integer, nullable=False)
    deepfake_frames = Column(Integer, nullable=False)
    deepfake_percentage = Column(Float, nullable=False)
    classification = Column(String(20), nullable=False)  # 'real', 'fake', 'suspicious'
    confidence = Column(Float, nullable=False)
    processing_time_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="video_analyses")

class APIUsage(Base):
    __tablename__ = 'api_usage'
    
    id = Column(Integer, primary_key=True)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    ip_address = Column(String(45))  # Changed from INET to String
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemMetric(Base):
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self, database_url=None):
        if database_url is None:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///deepfake_detection.db')
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def log_prediction(self, prediction_data, user_id=None, ip_address=None, user_agent=None):
        """Log prediction to database"""
        session = self.get_session()
        
        try:
            # Create image hash
            image_hash = hashlib.sha256(str(prediction_data).encode()).hexdigest()[:64]
            
            prediction = Prediction(
                user_id=user_id,
                image_hash=image_hash,
                prediction_class=prediction_data['prediction']['class'],
                confidence=prediction_data['prediction']['confidence'],
                real_probability=prediction_data['prediction']['probabilities']['real'],
                fake_probability=prediction_data['prediction']['probabilities']['fake'],
                processing_time_ms=prediction_data.get('processing_time_ms', 0),
                ip_address=str(ip_address) if ip_address else None,
                user_agent=user_agent
            )
            
            session.add(prediction)
            session.commit()
            
            return prediction.id
            
        except Exception as e:
            session.rollback()
            print(f"Error logging prediction: {e}")
            return None
        finally:
            session.close()
    
    def log_api_usage(self, endpoint, method, status_code, response_time_ms, user_id=None, ip_address=None, user_agent=None):
        """Log API usage to database"""
        session = self.get_session()
        
        try:
            api_usage = APIUsage(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                user_id=user_id,
                ip_address=str(ip_address) if ip_address else None,
                user_agent=user_agent
            )
            
            session.add(api_usage)
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"Error logging API usage: {e}")
        finally:
            session.close()
    
    def get_analytics_summary(self):
        """Get analytics summary"""
        session = self.get_session()
        
        try:
            # Total predictions
            total_predictions = session.query(Prediction).count()
            
            # Predictions by class
            real_predictions = session.query(Prediction).filter(Prediction.prediction_class == 'real').count()
            fake_predictions = session.query(Prediction).filter(Prediction.prediction_class == 'fake').count()
            
            # Recent predictions (last 24 hours)
            from datetime import timedelta
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_predictions = session.query(Prediction).filter(
                Prediction.created_at >= recent_cutoff
            ).count()
            
            # API usage stats
            total_api_calls = session.query(APIUsage).count()
            
            analytics = {
                'total_predictions': total_predictions,
                'real_predictions': real_predictions,
                'fake_predictions': fake_predictions,
                'recent_predictions_24h': recent_predictions,
                'total_api_calls': total_api_calls,
                'fake_detection_rate': (fake_predictions / total_predictions * 100) if total_predictions > 0 else 0
            }
            
            return analytics
            
        except Exception as e:
            print(f"Error getting analytics: {e}")
            return {}
        finally:
            session.close()

# Global database manager instance
db_manager = None

def get_db_manager():
    """Get global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

# Test database functionality
if __name__ == "__main__":
    print("Testing database models...")
    
    # Create database manager
    db = DatabaseManager()
    
    # Create test user
    session = db.get_session()
    
    test_user = User(username="testuser", email="test@example.com")
    session.add(test_user)
    session.commit()
    
    print(f"Created test user: {test_user.id}")
    
    # Test prediction logging
    test_prediction = {
        'prediction': {
            'class': 'fake',
            'confidence': 0.85,
            'probabilities': {
                'real': 0.15,
                'fake': 0.85
            }
        },
        'processing_time_ms': 45.2
    }
    
    prediction_id = db.log_prediction(test_prediction, user_id=test_user.id)
    print(f"Logged prediction: {prediction_id}")
    
    # Get analytics
    analytics = db.get_analytics_summary()
    print(f"Analytics: {analytics}")
    
    session.close()
    print("Database test completed successfully!")