"""
Data augmentation for deepfake detection training
"""

import cv2
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple
import json

class FaceAugmentor:
    """Basic data augmentation for face images using OpenCV"""
    
    def __init__(self):
        pass
    
    def augment_image(self, image: np.ndarray, augment_type="basic") -> np.ndarray:
        """Apply augmentation to a single image"""
        result = image.copy()
        
        if augment_type in ["basic", "heavy"]:
            # Random horizontal flip
            if random.random() > 0.5:
                result = cv2.flip(result, 1)
            
            # Random brightness adjustment
            brightness_factor = random.uniform(0.7, 1.3)
            result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
            
            # Random contrast adjustment
            contrast_factor = random.uniform(0.8, 1.2)
            result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=0)
            
            # Random rotation (small angle)
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                center = (result.shape[1]//2, result.shape[0]//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                result = cv2.warpAffine(result, rotation_matrix, (result.shape[1], result.shape[0]))
        
        if augment_type == "heavy":
            # Additional heavy augmentations
            # Add slight noise
            if random.random() > 0.6:
                noise = np.random.randint(0, 15, result.shape, dtype=np.uint8)
                result = cv2.add(result, noise)
            
            # Slight blur
            if random.random() > 0.7:
                result = cv2.GaussianBlur(result, (3, 3), 0.5)
        
        return result
    
    def create_augmented_dataset(self, input_dir: Path, output_dir: Path, 
                               augmentations_per_image=3):
        """Create augmented dataset from face crops"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'original_images': 0,
            'augmented_images': 0,
            'labels_processed': []
        }
        
        # Process each label directory
        for label_dir in input_dir.iterdir():
            if not label_dir.is_dir():
                continue
                
            label = label_dir.name
            output_label_dir = output_dir / label
            output_label_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all face images recursively
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(list(label_dir.glob(f'**/*{ext}')))
            
            print(f"Processing {len(image_files)} {label} images...")
            
            original_count = 0
            augmented_count = 0
            
            for img_path in image_files:
                # Load original image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Save original image to output
                original_filename = f"orig_{img_path.name}"
                original_output_path = output_label_dir / original_filename
                cv2.imwrite(str(original_output_path), image)
                original_count += 1
                
                # Create augmented versions
                for aug_idx in range(augmentations_per_image):
                    # Randomly choose augmentation type
                    aug_type = random.choice(['basic', 'basic', 'heavy'])  # Favor basic
                    
                    # Apply augmentation
                    augmented_image = self.augment_image(image, aug_type)
                    
                    # Save augmented image
                    aug_filename = f"aug{aug_idx}_{img_path.stem}_{aug_type}.jpg"
                    aug_output_path = output_label_dir / aug_filename
                    cv2.imwrite(str(aug_output_path), augmented_image)
                    augmented_count += 1
            
            stats['labels_processed'].append({
                'label': label,
                'original_images': original_count,
                'augmented_images': augmented_count
            })
            
            stats['original_images'] += original_count
            stats['augmented_images'] += augmented_count
            
            print(f"  {label}: {original_count} original + {augmented_count} augmented")
        
        # Save stats
        with open(output_dir / 'augmentation_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n=== Augmentation Complete ===")
        print(f"Total original images: {stats['original_images']}")
        print(f"Total augmented images: {stats['augmented_images']}")
        print(f"Total dataset size: {stats['original_images'] + stats['augmented_images']}")
        
        return stats