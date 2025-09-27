"""
Basic video processing utilities for deepfake detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Basic video processing for deepfake detection"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.MOV']
    
    def read_video_info(self, video_path):
        """Get basic information about a video file"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'path': str(video_path),
            'filename': video_path.name,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'file_size_mb': video_path.stat().st_size / (1024 * 1024)
        }
        
        info['duration_seconds'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
        info['resolution'] = f"{info['width']}x{info['height']}"
        
        cap.release()
        return info
    
    def extract_frames(self, video_path, max_frames=None, frame_interval=1):
        """Extract frames from video"""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame.copy())
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def find_videos(self, directory):
        """Find all video files in a directory"""
        directory = Path(directory)
        video_files = []
        
        if not directory.exists():
            return video_files
        
        for ext in self.supported_formats:
            video_files.extend(directory.glob(f"*{ext}"))
        
        return sorted(video_files)