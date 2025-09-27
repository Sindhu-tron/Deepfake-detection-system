"""
Deepfake API with comprehensive monitoring
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
import psutil

# Add project paths
sys.path.append('..')
sys.path.append('../src')

# Import monitoring
from monitoring.logger import get_logger, PerformanceMonitor

app = Flask(__name__)
CORS(app)

# Initialize logging
logger = get_logger()
perf_monitor = PerformanceMonitor(logger)

class MonitoredDeepfakeAPI:
    """API with comprehensive monitoring and logging"""
    
    def __init__(self, model_path='../training_outputs/models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # Track API statistics
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
    
    def load_model(self):
        """Load model with logging"""
        logger.info("Loading deepfake detection model", model_path=self.model_path)
        
        try:
            if os.path.exists(self.model_path):
                perf_monitor.start_timer("model_loading")
                self.model = tf.keras.models.load_model(self.model_path)
                load_time = perf_monitor.end_timer("model_loading")
                
                logger.info(
                    "Model loaded successfully",
                    model_path=self.model_path,
                    parameters=self.model.count_params(),
                    load_time_ms=load_time * 1000 if load_time else 0
                )
                return True
            else:
                logger.error("Model file not found", model_path=self.model_path)
                return False
        except Exception as e:
            logger.error("Model loading failed", error=str(e), model_path=self.model_path)
            return False
    
    def predict(self, image, request_id=None):
        """Make prediction with comprehensive monitoring"""
        perf_monitor.start_timer("total_prediction")
        
        if self.model is None:
            logger.error("Prediction attempted with no model loaded", request_id=request_id)
            return {'error': 'Model not loaded'}
        
        try:
            # Preprocess image
            perf_monitor.start_timer("image_preprocessing")
            processed_image = self._preprocess_image(image)
            preprocess_time = perf_monitor.end_timer("image_preprocessing")
            
            if processed_image is None:
                logger.error("Image preprocessing failed", request_id=request_id)
                return {'error': 'Image preprocessing failed'}
            
            # Make prediction
            perf_monitor.start_timer("model_inference")
            prediction = self.model.predict(processed_image, verbose=0)
            inference_time = perf_monitor.end_timer("model_inference")
            
            # Process results
            fake_prob = float(prediction[0][0])
            real_prob = float(prediction[0][1])
            predicted_class = 'real' if real_prob > fake_prob else 'fake'
            confidence = max(real_prob, fake_prob)
            
            total_time = perf_monitor.end_timer("total_prediction")
            
            result = {
                'success': True,
                'prediction': {
                    'class': predicted_class,
                    'confidence': confidence,
                    'probabilities': {
                        'real': real_prob,
                        'fake': fake_prob
                    }
                },
                'performance': {
                    'total_time_ms': total_time * 1000 if total_time else 0,
                    'preprocessing_time_ms': preprocess_time * 1000 if preprocess_time else 0,
                    'inference_time_ms': inference_time * 1000 if inference_time else 0
                },
                'model_info': {
                    'version': '1.0',
                    'input_shape': [224, 224, 3]
                }
            }
            
            # Log prediction metrics
            logger.info(
                "Prediction completed successfully",
                request_id=request_id,
                predicted_class=predicted_class,
                confidence=confidence,
                total_time_ms=total_time * 1000 if total_time else 0,
                image_shape=image.shape if hasattr(image, 'shape') else None
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Prediction failed",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _preprocess_image(self, image):
        """Preprocess image with error handling"""
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
            logger.error("Image preprocessing error", error=str(e))
            return None
    
    def get_system_metrics(self):
        """Get current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'uptime_seconds': time.time() - self.start_time,
            'model_loaded': self.model is not None
        }

# Initialize API
api = MonitoredDeepfakeAPI()

# Request logging middleware
@app.before_request
def log_request():
    request.start_time = time.time()
    request.request_id = hashlib.md5(f"{time.time()}{request.remote_addr}".encode()).hexdigest()[:8]
    
    api.request_count += 1
    
    logger.info(
        "Request started",
        request_id=request.request_id,
        method=request.method,
        path=request.path,
        remote_addr=request.remote_addr,
        user_agent=request.headers.get('User-Agent', 'Unknown')
    )

@app.after_request
def log_response(response):
    if hasattr(request, 'start_time'):
        response_time = (time.time() - request.start_time) * 1000
        
        if response.status_code >= 400:
            api.error_count += 1
            log_level = "error"
        elif response.status_code >= 300:
            log_level = "warning"
        else:
            log_level = "info"
        
        getattr(logger, log_level)(
            "Request completed",
            request_id=getattr(request, 'request_id', 'unknown'),
            method=request.method,
            path=request.path,
            status_code=response.status_code,
            response_time_ms=response_time,
            content_length=response.content_length
        )
    
    return response

def decode_base64_image(base64_string):
    """Decode base64 image with error handling"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_data))
        image_array = np.array(pil_image)
        
        return image_array
        
    except Exception as e:
        logger.error("Base64 decode error", error=str(e))
        return None

@app.route('/', methods=['GET'])
def api_info():
    """API information with system metrics"""
    system_metrics = api.get_system_metrics()
    
    return jsonify({
        'name': 'Monitored Deepfake Detection API',
        'version': '3.0.0',
        'description': 'REST API with comprehensive monitoring',
        'model_status': 'loaded' if api.model is not None else 'not loaded',
        'system_metrics': system_metrics,
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Detect deepfakes',
            '/metrics': 'GET - System metrics'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check"""
    system_metrics = api.get_system_metrics()
    
    # Determine health status
    health_status = "healthy"
    if system_metrics['memory_percent'] > 90:
        health_status = "degraded"
    if not system_metrics['model_loaded']:
        health_status = "unhealthy"
    
    return jsonify({
        'status': health_status,
        'timestamp': time.time(),
        'system_metrics': system_metrics,
        'version': '3.0.0'
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Prometheus-style metrics endpoint"""
    system_metrics = api.get_system_metrics()
    
    metrics_text = f"""
# HELP deepfake_requests_total Total number of requests
# TYPE deepfake_requests_total counter
deepfake_requests_total {system_metrics['request_count']}

# HELP deepfake_errors_total Total number of errors
# TYPE deepfake_errors_total counter
deepfake_errors_total {system_metrics['error_count']}

# HELP deepfake_cpu_percent CPU usage percentage
# TYPE deepfake_cpu_percent gauge
deepfake_cpu_percent {system_metrics['cpu_percent']}

# HELP deepfake_memory_percent Memory usage percentage
# TYPE deepfake_memory_percent gauge
deepfake_memory_percent {system_metrics['memory_percent']}

# HELP deepfake_uptime_seconds Uptime in seconds
# TYPE deepfake_uptime_seconds gauge
deepfake_uptime_seconds {system_metrics['uptime_seconds']}
"""
    
    return metrics_text.strip(), 200, {'Content-Type': 'text/plain'}

@app.route('/predict', methods=['POST'])
def predict_image():
    """Enhanced prediction endpoint with monitoring"""
    try:
        if api.model is None:
            logger.error("Prediction attempted with no model", request_id=request.request_id)
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
        result = api.predict(image, request_id=request.request_id)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(
            "Prediction endpoint error",
            request_id=request.request_id,
            error=str(e),
            error_type=type(e).__name__
        )
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting Monitored Deepfake Detection API")
    logger.info("Logging configured", log_level="INFO")
    
    app.run(debug=False, host='0.0.0.0', port=5001)