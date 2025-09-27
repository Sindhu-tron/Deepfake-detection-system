"""
Test video processing capabilities with sample videos
"""

import sys
from pathlib import Path
from video_analyzer import VideoDeepfakeAnalyzer
import cv2
import numpy as np

def create_test_video(output_path, label='real', duration=3, fps=15):
    """Create a test video with face-like shapes"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    
    print(f"Creating test video: {output_path}")
    
    for frame_num in range(total_frames):
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Background color based on label
        if label == 'real':
            frame[:, :] = (30, 50, 70)
        else:
            frame[:, :] = (50, 30, 70)
        
        # Create moving face-like shape
        center_x = 320 + int(50 * np.sin(frame_num * 0.15))
        center_y = 240 + int(30 * np.cos(frame_num * 0.1))
        
        # Face oval
        cv2.ellipse(frame, (center_x, center_y), (100, 120), 0, 0, 360, (200, 180, 150), -1)
        
        # Eyes
        cv2.circle(frame, (center_x - 35, center_y - 25), 15, (255, 255, 255), -1)
        cv2.circle(frame, (center_x + 35, center_y - 25), 15, (255, 255, 255), -1)
        cv2.circle(frame, (center_x - 35, center_y - 25), 8, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + 35, center_y - 25), 8, (0, 0, 0), -1)
        
        # Nose
        pts = np.array([[center_x, center_y], [center_x-15, center_y+20], [center_x+15, center_y+20]], np.int32)
        cv2.fillPoly(frame, [pts], (180, 160, 130))
        
        # Mouth
        cv2.ellipse(frame, (center_x, center_y + 40), (25, 15), 0, 0, 180, (120, 80, 80), -1)
        
        # Add some variation for "fake" videos
        if label == 'fake':
            # Add random noise
            noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")

def test_video_processing():
    """Test the video processing system"""
    print("Testing Video Processing System")
    print("="*50)
    
    # Create test videos directory
    test_videos_dir = Path("test_videos")
    test_videos_dir.mkdir(exist_ok=True)
    
    # Create test videos
    real_video = test_videos_dir / "test_real_video.mp4"
    fake_video = test_videos_dir / "test_fake_video.mp4"
    
    create_test_video(real_video, label='real', duration=5)
    create_test_video(fake_video, label='fake', duration=5)
    
    # Initialize analyzer
    print("\nInitializing video analyzer...")
    analyzer = VideoDeepfakeAnalyzer()
    
    if not analyzer.model:
        print("Error: Model not loaded")
        return False
    
    print("Model loaded successfully")
    
    # Test on real video
    print(f"\nAnalyzing REAL test video...")
    real_results = analyzer.analyze_video(real_video, output_dir="test_results/real")
    
    if 'error' in real_results:
        print(f"Error analyzing real video: {real_results['error']}")
    else:
        classification = real_results['analysis_summary']['classification']
        print(f"Real video result: {classification['class']} ({classification['confidence']:.1%})")
    
    # Test on fake video
    print(f"\nAnalyzing FAKE test video...")
    fake_results = analyzer.analyze_video(fake_video, output_dir="test_results/fake")
    
    if 'error' in fake_results:
        print(f"Error analyzing fake video: {fake_results['error']}")
    else:
        classification = fake_results['analysis_summary']['classification']
        print(f"Fake video result: {classification['class']} ({classification['confidence']:.1%})")
    
    print("\nVideo processing test complete!")
    print("Test videos created in: test_videos/")
    print("Results saved in: test_results/")
    
    return True

if __name__ == "__main__":
    test_video_processing()