#!/usr/bin/env python3
"""
Fix data leakage by splitting at the video level, not image level
"""

import random
from pathlib import Path
import shutil
import json

def fix_data_splits():
    print("Fixing data splits to prevent leakage...")
    
    # Start from the original extracted faces (before augmentation)
    source_dir = Path("data/processed/extracted_faces")
    output_dir = Path("data/processed/fixed_splits")
    
    if not source_dir.exists():
        print("Original extracted faces not found")
        return
    
    # Clear existing fixed splits
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Create new directory structure
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    random.seed(42)
    
    # Process each label
    for label in ['real', 'fake']:
        label_dir = source_dir / label
        if not label_dir.exists():
            continue
        
        # Group images by original video
        video_groups = {}
        
        for img_path in label_dir.glob('**/*.jpg'):
            # Extract video name from filename (e.g., "real_video_1_f001_face0.jpg")
            video_name = '_'.join(img_path.stem.split('_')[:3])  # Gets "real_video_1"
            
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(img_path)
        
        # Split videos, not individual images
        video_names = list(video_groups.keys())
        random.shuffle(video_names)
        
        # Split videos: 70% train, 15% val, 15% test
        n_videos = len(video_names)
        train_videos = video_names[:int(0.7 * n_videos)]
        val_videos = video_names[int(0.7 * n_videos):int(0.85 * n_videos)]
        test_videos = video_names[int(0.85 * n_videos):]
        
        print(f"{label}: {len(train_videos)} train videos, {len(val_videos)} val videos, {len(test_videos)} test videos")
        
        # Copy images based on video assignment
        for video_name, images in video_groups.items():
            if video_name in train_videos:
                split = 'train'
            elif video_name in val_videos:
                split = 'val'
            else:
                split = 'test'
            
            for img_path in images:
                dst_path = output_dir / split / label / img_path.name
                shutil.copy2(img_path, dst_path)
    
    # Create new config
    config = {
        'note': 'Fixed data splits - no data leakage',
        'file_paths': {
            'train_dir': str(output_dir / 'train'),
            'val_dir': str(output_dir / 'val'),
            'test_dir': str(output_dir / 'test')
        }
    }
    
    with open('config/fixed_dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Fixed splits saved to: {output_dir}")
    
    # Count final results
    for split in ['train', 'val', 'test']:
        real_count = len(list((output_dir / split / 'real').glob('*.jpg')))
        fake_count = len(list((output_dir / split / 'fake').glob('*.jpg')))
        print(f"{split}: {real_count + fake_count} images ({real_count} real, {fake_count} fake)")

if __name__ == "__main__":
    fix_data_splits()