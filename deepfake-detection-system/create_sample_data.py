# Create sample video generator
#!/usr/bin/env python3
"""
Create sample videos for testing the deepfake detection pipeline
"""

import cv2 # type: ignore
import numpy as np
from pathlib import Path
import random

def create_face_like_video(output_path, label='real', duration_seconds=3, fps=15):
    """
    Create a simple video with face-like shapes for testing
    
    Args:
        output_path: Where to save the video
        label: 'real' or 'fake' - affects visual characteristics
        duration_seconds: Length of video
        fps: Frames per second
    """
    
    # Video properties
    width, height = 640, 480
    total_frames = duration_seconds * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Creating {label} video: {output_path}")
    
    for frame_num in range(total_frames):
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background color based on label
        if label == 'real':
            frame[:, :] = (20, 40, 60)  # Dark blue background
        else:
            frame[:, :] = (40, 20, 60)  # Dark purple background (slightly different)
        
        # Create face-like oval
        center_x = width // 2 + int(10 * np.sin(frame_num * 0.1))  # Slight movement
        center_y = height // 2 + int(5 * np.cos(frame_num * 0.1))
        
        # Face oval
        cv2.ellipse(frame, (center_x, center_y), (100, 130), 0, 0, 360, 
                   (180, 150, 120), -1)  # Skin-like color
        
        # Eyes
        eye_y = center_y - 30
        cv2.circle(frame, (center_x - 35, eye_y), 15, (255, 255, 255), -1)  # Left eye white
        cv2.circle(frame, (center_x + 35, eye_y), 15, (255, 255, 255), -1)  # Right eye white
        cv2.circle(frame, (center_x - 35, eye_y), 8, (0, 0, 0), -1)  # Left pupil
        cv2.circle(frame, (center_x + 35, eye_y), 8, (0, 0, 0), -1)  # Right pupil
        
        # Nose
        nose_points = np.array([
            [center_x, center_y - 10],
            [center_x - 8, center_y + 10], 
            [center_x + 8, center_y + 10]
        ], np.int32)
        cv2.fillPoly(frame, [nose_points], (160, 130, 100))
        
        # Mouth
        cv2.ellipse(frame, (center_x, center_y + 40), (25, 15), 0, 0, 180, 
                   (100, 50, 50), -1)
        
        # Add some "artifacts" for fake videos
        if label == 'fake':
            # Add some inconsistent pixels (simulating deepfake artifacts)
            for _ in range(10):
                x = random.randint(center_x - 80, center_x + 80)
                y = random.randint(center_y - 100, center_y + 100)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.circle(frame, (x, y), 2, color, -1)
            
            # Add slight color inconsistency in face region
            noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Created {output_path} with {total_frames} frames")

def create_sample_dataset():
    """Create a complete sample dataset"""
    
    # Create directories
    real_dir = Path("data/raw/sample_dataset/real")
    fake_dir = Path("data/raw/sample_dataset/fake")
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample dataset...")
    
    # Create real videos
    for i in range(5):
        video_path = real_dir / f"real_sample_{i+1}.mp4"
        create_face_like_video(video_path, label='real', duration_seconds=4)
    
    # Create fake videos  
    for i in range(5):
        video_path = fake_dir / f"fake_sample_{i+1}.mp4"
        create_face_like_video(video_path, label='fake', duration_seconds=4)
    
    print("\n✅ Sample dataset created!")
    print(f"Real videos: {len(list(real_dir.glob('*.mp4')))}")
    print(f"Fake videos: {len(list(fake_dir.glob('*.mp4')))}")
    print(f"Total size: ~{sum(p.stat().st_size for p in real_dir.glob('*.mp4')) / 1024 / 1024:.1f}MB")

if __name__ == "__main__":
    create_sample_dataset()
# Run the sample data creator
# python create_sample_data.py