"""
Data quality assessment for deepfake detection datasets
"""

import cv2 # type: ignore
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from .video_utils import VideoProcessor
from .face_detection import FaceDetector

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Assess quality of video datasets for deepfake detection"""
    
    def __init__(self):
        self.processor = VideoProcessor()
        self.face_detector = FaceDetector(method='mediapipe', confidence_threshold=0.3)
    
    def assess_video_quality(self, video_path: Path) -> Dict:
        """
        Assess quality of a single video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Get basic video info
            video_info = self.processor.read_video_info(video_path)
            
            # Extract sample frames for quality assessment
            frames = self.processor.extract_frames(
                video_path, 
                max_frames=5, 
                frame_interval=max(1, video_info['frame_count'] // 10)
            )
            
            if not frames:
                return {'error': 'No frames extracted'}
            
            quality_metrics = {
                'filename': video_path.name,
                'basic_info': video_info,
                'frame_quality': self._assess_frame_quality(frames),
                'face_detection': self._assess_face_detection(frames),
                'overall_score': 0.0
            }
            
            # Calculate overall quality score
            quality_metrics['overall_score'] = self._calculate_overall_score(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error assessing {video_path}: {e}")
            return {'error': str(e)}
    
    def _assess_frame_quality(self, frames: List[np.ndarray]) -> Dict:
        """Assess quality of individual frames"""
        brightness_scores = []
        blur_scores = []
        
        for frame in frames:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness assessment
            brightness = np.mean(gray)
            brightness_scores.append(brightness)
            
            # Blur assessment using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(blur_score)
        
        return {
            'brightness': {
                'mean': float(np.mean(brightness_scores)),
                'std': float(np.std(brightness_scores)),
                'min': float(np.min(brightness_scores)),
                'max': float(np.max(brightness_scores))
            },
            'blur': {
                'mean': float(np.mean(blur_scores)),
                'std': float(np.std(blur_scores)),
                'min': float(np.min(blur_scores)),
                'max': float(np.max(blur_scores))
            },
            'frames_analyzed': len(frames)
        }
    
    def _assess_face_detection(self, frames: List[np.ndarray]) -> Dict:
        """Assess face detection success rate and quality"""
        face_counts = []
        detection_confidences = []
        
        for frame in frames:
            faces = self.face_detector.detect_faces(frame)
            face_counts.append(len(faces))
            
            # For MediaPipe, we can't easily get confidence scores
            # So we'll estimate based on face size and position
            if faces:
                for face_bbox in faces:
                    x, y, w, h = face_bbox
                    # Estimate confidence based on face size
                    face_area = w * h
                    frame_area = frame.shape[0] * frame.shape[1]
                    face_ratio = face_area / frame_area
                    
                    # Simple confidence estimation
                    confidence = min(1.0, face_ratio * 10)
                    detection_confidences.append(confidence)
        
        return {
            'face_detection_rate': sum(1 for count in face_counts if count > 0) / len(face_counts),
            'avg_faces_per_frame': float(np.mean(face_counts)),
            'face_count_consistency': float(np.std(face_counts)),
            'estimated_avg_confidence': float(np.mean(detection_confidences)) if detection_confidences else 0.0
        }
    
    def _calculate_overall_score(self, quality_metrics: Dict) -> float:
        """Calculate overall quality score (0-1)"""
        try:
            frame_quality = quality_metrics['frame_quality']
            face_detection = quality_metrics['face_detection']
            
            # Brightness score (0-1, best around 100-150)
            brightness_score = 1.0 - abs(frame_quality['brightness']['mean'] - 125) / 125
            brightness_score = max(0, min(1, brightness_score))
            
            # Blur score (0-1, higher is better, but normalize to 0-1)
            blur_score = min(1.0, frame_quality['blur']['mean'] / 500)
            
            # Face detection score
            face_score = face_detection['face_detection_rate']
            
            # Weighted overall score
            overall_score = (
                brightness_score * 0.3 +
                blur_score * 0.3 +
                face_score * 0.4
            )
            
            return float(overall_score)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def assess_dataset_quality(self, dataset_path: Path, max_videos: int = None) -> Dict:
        """
        Assess quality of entire dataset
        
        Args:
            dataset_path: Path to dataset directory
            max_videos: Maximum number of videos to assess
            
        Returns:
            Dictionary with dataset quality report
        """
        video_files = self.processor.find_videos(dataset_path)
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        logger.info(f"Assessing quality of {len(video_files)} videos in {dataset_path}")
        
        quality_results = []
        
        for video_path in video_files:
            result = self.assess_video_quality(video_path)
            quality_results.append(result)
        
        # Generate summary statistics
        valid_results = [r for r in quality_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid videos found'}
        
        overall_scores = [r['overall_score'] for r in valid_results]
        face_detection_rates = [r['face_detection']['face_detection_rate'] for r in valid_results]
        
        summary = {
            'dataset_path': str(dataset_path),
            'videos_assessed': len(valid_results),
            'videos_with_errors': len(quality_results) - len(valid_results),
            'overall_quality': {
                'mean': float(np.mean(overall_scores)),
                'std': float(np.std(overall_scores)),
                'min': float(np.min(overall_scores)),
                'max': float(np.max(overall_scores))
            },
            'face_detection_summary': {
                'mean_detection_rate': float(np.mean(face_detection_rates)),
                'videos_with_faces': sum(1 for rate in face_detection_rates if rate > 0),
                'videos_without_faces': sum(1 for rate in face_detection_rates if rate == 0)
            },
            'detailed_results': quality_results
        }
        
        return summary

# Test quality checker
if __name__ == "__main__":
    checker = DataQualityChecker()
    
    # Test with sample dataset
    sample_dir = Path("data/raw/sample_dataset/real")
    
    if sample_dir.exists():
        print("Testing quality checker on sample videos...")
        
        quality_report = checker.assess_dataset_quality(sample_dir, max_videos=3)
        
        if 'error' not in quality_report:
            print(f"\nQuality Assessment Summary:")
            print(f"Videos assessed: {quality_report['videos_assessed']}")
            print(f"Mean quality score: {quality_report['overall_quality']['mean']:.3f}")
            print(f"Face detection rate: {quality_report['face_detection_summary']['mean_detection_rate']:.3f}")
            print(f"Videos with faces: {quality_report['face_detection_summary']['videos_with_faces']}")
        else:
            print(f"Error: {quality_report['error']}")
    else:
        print("Sample dataset not found. Run create_sample_data.py first.")
EOF # type: ignore

# Test quality checker (with import fix)
#python -c "
import sys
sys.path.append('src')

from preprocessing.quality_checker import DataQualityChecker
from pathlib import Path

checker = DataQualityChecker()
sample_dir = Path('data/raw/sample_dataset/real')

if sample_dir.exists():
    print('Testing quality assessment...')
    report = checker.assess_dataset_quality(sample_dir, max_videos=2)
    
    if 'error' not in report:
        print(f'✅ Quality assessment successful!')
        print(f"Videos assessed: {report['videos_assessed']}")
        print(f"Mean quality score: {report['overall_quality']['mean']:.3f}")
    else:
        print(f'Error: {report['error']}')
else:
    print('❌ Sample dataset not found')
