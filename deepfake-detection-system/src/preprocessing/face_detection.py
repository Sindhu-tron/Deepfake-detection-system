"""
Face detection utilities using OpenCV only
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    """Face detection using OpenCV Haar cascades"""
    
    def __init__(self, method='opencv', confidence_threshold=0.5):
        """Initialize face detector with OpenCV"""
        self.method = 'opencv'
        self.confidence_threshold = confidence_threshold
        
        # Load OpenCV's Haar cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info("Face detector initialized with OpenCV")
    
    def detect_faces(self, image):
        """Detect faces in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def extract_face(self, image, bbox, target_size=(224, 224), padding=0.2):
        """Extract and resize face from image"""
        x, y, w, h = bbox
        
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return None
        
        try:
            face_resized = cv2.resize(face_region, target_size)
            return face_resized
        except Exception as e:
            logger.error(f"Error resizing face: {e}")
            return None