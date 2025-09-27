"""
Quality filtering system for face crops
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

class FaceQualityFilter:
    """Filter face crops based on quality metrics"""
    
    def __init__(self):
        # Quality thresholds
        self.min_brightness = 30
        self.max_brightness = 220
        self.min_blur_score = 50
        self.min_contrast = 20
        
    def calculate_quality_metrics(self, image: np.ndarray) -> Dict:
        """Calculate quality metrics for a face image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness metrics
        brightness_mean = np.mean(gray)
        
        # Blur assessment (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast assessment
        contrast = gray.max() - gray.min()
        
        return {
            'brightness_mean': brightness_mean,
            'blur_score': blur_score,
            'contrast': contrast
        }
    
    def is_good_quality(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """Determine if image meets quality standards"""
        issues = []
        
        # Check brightness
        if metrics['brightness_mean'] < self.min_brightness:
            issues.append("too_dark")
        elif metrics['brightness_mean'] > self.max_brightness:
            issues.append("too_bright")
        
        # Check blur
        if metrics['blur_score'] < self.min_blur_score:
            issues.append("too_blurry")
        
        # Check contrast
        if metrics['contrast'] < self.min_contrast:
            issues.append("low_contrast")
        
        is_good = len(issues) == 0
        return is_good, issues
    
    def filter_dataset(self, input_dir: Path, output_dir: Path) -> Dict:
        """Filter entire dataset based on quality"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_images': 0,
            'kept_images': 0,
            'rejected_images': 0,
            'rejection_reasons': {}
        }
        
        # Process each label directory
        for label_dir in input_dir.iterdir():
            if not label_dir.is_dir():
                continue
            
            label = label_dir.name
            output_label_dir = output_dir / label
            output_label_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all images
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(list(label_dir.glob(f'*{ext}')))
            
            print(f"Processing {len(image_files)} {label} images...")
            
            for img_path in image_files:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    stats['rejected_images'] += 1
                    continue
                
                # Calculate quality metrics
                metrics = self.calculate_quality_metrics(image)
                
                # Check if good quality
                is_good, issues = self.is_good_quality(metrics)
                
                if is_good:
                    # Copy good image to output
                    output_path = output_label_dir / img_path.name
                    cv2.imwrite(str(output_path), image)
                    stats['kept_images'] += 1
                else:
                    # Track rejection reasons
                    for issue in issues:
                        if issue not in stats['rejection_reasons']:
                            stats['rejection_reasons'][issue] = 0
                        stats['rejection_reasons'][issue] += 1
                    stats['rejected_images'] += 1
                
                stats['total_images'] += 1
        
        print(f"Quality filtering complete: {stats['kept_images']} kept, {stats['rejected_images']} rejected")
        return stats
    