#!/usr/bin/env python3
"""
Command-line script for video deepfake analysis
"""

import sys
import argparse
from pathlib import Path
from video_analyzer import VideoDeepfakeAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze video for deepfake content')
    parser.add_argument('video_path', help='Path to video file to analyze')
    parser.add_argument('--output', '-o', help='Output directory for results', default='video_analysis_results')
    parser.add_argument('--model', '-m', help='Path to model file', default='../training_outputs/models/best_model.h5')
    parser.add_argument('--skip-frames', '-s', type=int, default=5, help='Frame skip interval (default: 5)')
    
    args = parser.parse_args()
    
    # Validate input file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    # Create analyzer
    print("Initializing video analyzer...")
    analyzer = VideoDeepfakeAnalyzer(model_path=args.model)
    analyzer.frame_skip = args.skip_frames
    
    if not analyzer.model:
        print("Error: Could not load model")
        return 1
    
    # Analyze video
    print(f"Starting analysis of: {video_path}")
    results = analyzer.analyze_video(video_path, output_dir=args.output)
    
    if 'error' in results:
        print(f"Analysis failed: {results['error']}")
        return 1
    
    # Display results
    print("\n" + "="*60)
    print("VIDEO ANALYSIS RESULTS")
    print("="*60)
    
    video_info = results['video_info']
    analysis = results['analysis_summary']
    
    print(f"Video: {video_info['filename']}")
    print(f"Duration: {video_info['duration_seconds']:.1f} seconds")
    print(f"Resolution: {video_info['resolution']}")
    print(f"FPS: {video_info['fps']:.1f}")
    
    print(f"\nFrames Analyzed: {analysis['frames_analyzed']}")
    print(f"Total Faces Detected: {analysis['total_faces_detected']}")
    print(f"Deepfake Frames: {analysis['deepfake_frames']} ({analysis['deepfake_percentage']:.1f}%)")
    print(f"Average Fake Probability: {analysis['average_fake_probability']:.3f}")
    
    classification = analysis['classification']
    print(f"\nFINAL CLASSIFICATION: {classification['class'].upper()}")
    print(f"Confidence: {classification['confidence']:.1%}")
    
    # Interpretation
    if classification['class'] == 'fake':
        print("\n🚨 DEEPFAKE DETECTED: This video likely contains artificially generated content")
    elif classification['class'] == 'suspicious':
        print("\n⚠️  SUSPICIOUS: This video shows some signs of manipulation")
    else:
        print("\n✅ APPEARS AUTHENTIC: No significant deepfake patterns detected")
    
    print(f"\nDetailed results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())