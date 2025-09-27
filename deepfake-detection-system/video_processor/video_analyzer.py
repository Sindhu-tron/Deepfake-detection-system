"""
Video processing system for deepfake detection
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Add project paths
sys.path.append('..')
sys.path.append('../src')

try:
    from preprocessing.face_detection import FaceDetector
    from preprocessing.video_utils import VideoProcessor
except ImportError:
    print("Warning: Could not import preprocessing modules")

import tensorflow as tf

class VideoDeepfakeAnalyzer:
    """Analyze entire videos for deepfake content"""
    
    def __init__(self, model_path='../training_outputs/models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.face_detector = FaceDetector()
        self.video_processor = VideoProcessor()
        
        # Analysis parameters
        self.frame_skip = 5  # Process every 5th frame for efficiency
        self.max_faces_per_frame = 3  # Analyze up to 3 faces per frame
        self.confidence_threshold = 0.6  # Minimum prediction confidence
        
        self.load_model()
    
    def load_model(self):
        """Load the deepfake detection model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Video Analyzer: Model loaded from {self.model_path}")
                return True
            else:
                print(f"Video Analyzer: Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Video Analyzer: Error loading model: {e}")
            return False
    
    def preprocess_face(self, face_image):
        """Preprocess face for model prediction"""
        try:
            # Resize to model input size
            face_resized = cv2.resize(face_image, (224, 224))
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            face_normalized = (face_normalized - mean) / std
            
            # Add batch dimension
            return np.expand_dims(face_normalized, axis=0)
            
        except Exception as e:
            print(f"Face preprocessing error: {e}")
            return None
    
    def analyze_frame(self, frame):
        """Analyze a single frame for deepfakes"""
        frame_results = {
            'faces_detected': 0,
            'predictions': [],
            'average_fake_probability': 0.0,
            'max_fake_probability': 0.0,
            'deepfake_detected': False
        }
        
        if self.model is None:
            return frame_results
        
        try:
            # Detect faces in frame
            faces = self.face_detector.detect_faces(frame)
            frame_results['faces_detected'] = len(faces)
            
            if not faces:
                return frame_results
            
            fake_probabilities = []
            
            # Analyze each face (up to max limit)
            for i, bbox in enumerate(faces[:self.max_faces_per_frame]):
                x, y, w, h = bbox
                
                # Extract face
                face_crop = self.face_detector.extract_face(
                    frame, bbox, target_size=(224, 224), padding=0.3
                )
                
                if face_crop is None:
                    continue
                
                # Preprocess and predict
                processed_face = self.preprocess_face(face_crop)
                if processed_face is None:
                    continue
                
                prediction = self.model.predict(processed_face, verbose=0)
                
                fake_prob = float(prediction[0][0])
                real_prob = float(prediction[0][1])
                
                predicted_class = 'real' if real_prob > fake_prob else 'fake'
                confidence = max(real_prob, fake_prob)
                
                face_result = {
                    'face_index': i,
                    'bbox': bbox,
                    'class': predicted_class,
                    'confidence': confidence,
                    'fake_probability': fake_prob,
                    'real_probability': real_prob
                }
                
                frame_results['predictions'].append(face_result)
                fake_probabilities.append(fake_prob)
            
            # Calculate frame-level statistics
            if fake_probabilities:
                frame_results['average_fake_probability'] = np.mean(fake_probabilities)
                frame_results['max_fake_probability'] = np.max(fake_probabilities)
                
                # Determine if frame contains deepfake
                frame_results['deepfake_detected'] = (
                    frame_results['max_fake_probability'] > 0.5 or
                    frame_results['average_fake_probability'] > 0.4
                )
            
            return frame_results
            
        except Exception as e:
            print(f"Frame analysis error: {e}")
            return frame_results
    
    def analyze_video(self, video_path, output_dir=None):
        """Analyze entire video for deepfake content"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            return {'error': f'Video file not found: {video_path}'}
        
        if output_dir is None:
            output_dir = Path('video_analysis_results')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Analyzing video: {video_path}")
        
        try:
            # Get video info
            video_info = self.video_processor.read_video_info(video_path)
            
            # Extract frames for analysis
            frames = self.video_processor.extract_frames(
                video_path, 
                max_frames=None,
                frame_interval=self.frame_skip
            )
            
            if not frames:
                return {'error': 'No frames could be extracted from video'}
            
            # Analyze each frame
            frame_results = []
            deepfake_frames = 0
            total_fake_probability = 0
            total_faces = 0
            
            print(f"Processing {len(frames)} frames...")
            
            for frame_idx, frame in enumerate(frames):
                if frame_idx % 10 == 0:
                    print(f"Processing frame {frame_idx}/{len(frames)}")
                
                result = self.analyze_frame(frame)
                result['frame_number'] = frame_idx * self.frame_skip
                result['timestamp'] = (frame_idx * self.frame_skip) / video_info['fps']
                
                frame_results.append(result)
                
                if result['deepfake_detected']:
                    deepfake_frames += 1
                
                total_fake_probability += result['average_fake_probability']
                total_faces += result['faces_detected']
            
            # Calculate video-level statistics
            total_frames = len(frames)
            deepfake_percentage = (deepfake_frames / total_frames) * 100 if total_frames > 0 else 0
            average_fake_probability = total_fake_probability / total_frames if total_frames > 0 else 0
            
            # Determine overall video classification
            video_classification = self._classify_video(
                deepfake_percentage, 
                average_fake_probability,
                deepfake_frames,
                total_frames
            )
            
            # Compile results
            analysis_results = {
                'video_info': {
                    'filename': video_path.name,
                    'path': str(video_path),
                    'duration_seconds': video_info['duration_seconds'],
                    'fps': video_info['fps'],
                    'total_frames': video_info['frame_count'],
                    'resolution': video_info['resolution']
                },
                'analysis_summary': {
                    'frames_analyzed': total_frames,
                    'frame_skip_interval': self.frame_skip,
                    'total_faces_detected': total_faces,
                    'deepfake_frames': deepfake_frames,
                    'deepfake_percentage': deepfake_percentage,
                    'average_fake_probability': average_fake_probability,
                    'classification': video_classification,
                    'confidence': video_classification['confidence']
                },
                'frame_results': frame_results,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save results
            results_file = output_dir / f"{video_path.stem}_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Create visualization
            self._create_analysis_plots(analysis_results, output_dir)
            
            print(f"Analysis complete: {video_classification['class']} ({video_classification['confidence']:.1%})")
            print(f"Results saved to: {output_dir}")
            
            return analysis_results
            
        except Exception as e:
            return {'error': f'Video analysis failed: {str(e)}'}
    
    def _classify_video(self, deepfake_percentage, avg_fake_prob, deepfake_frames, total_frames):
        """Classify video as real or fake based on analysis"""
        
        # Classification logic
        if deepfake_percentage > 30 and avg_fake_prob > 0.6:
            classification = 'fake'
            confidence = min(0.95, 0.5 + (deepfake_percentage / 100) + avg_fake_prob / 2)
        elif deepfake_percentage > 15 and avg_fake_prob > 0.5:
            classification = 'fake'
            confidence = min(0.85, 0.4 + (deepfake_percentage / 100) + avg_fake_prob / 2)
        elif deepfake_percentage > 10 or avg_fake_prob > 0.45:
            classification = 'suspicious'
            confidence = 0.6 + (deepfake_percentage / 100) * 0.3
        else:
            classification = 'real'
            confidence = min(0.9, 0.8 - avg_fake_prob)
        
        return {
            'class': classification,
            'confidence': confidence,
            'reasoning': {
                'deepfake_frame_percentage': deepfake_percentage,
                'average_fake_probability': avg_fake_prob,
                'deepfake_frames': deepfake_frames,
                'total_frames': total_frames
            }
        }
    
    def _create_analysis_plots(self, results, output_dir):
        """Create visualization plots for video analysis"""
        try:
            frame_results = results['frame_results']
            
            if not frame_results:
                return
            
            # Extract data for plotting
            timestamps = [r['timestamp'] for r in frame_results]
            fake_probs = [r['average_fake_probability'] for r in frame_results]
            faces_detected = [r['faces_detected'] for r in frame_results]
            deepfake_flags = [1 if r['deepfake_detected'] else 0 for r in frame_results]
            
            # Create plots
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f'Video Analysis: {results["video_info"]["filename"]}', fontsize=14)
            
            # Plot 1: Fake probability over time
            axes[0].plot(timestamps, fake_probs, 'b-', alpha=0.7)
            axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Threshold')
            axes[0].set_ylabel('Fake Probability')
            axes[0].set_title('Deepfake Probability Over Time')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Plot 2: Faces detected over time
            axes[1].bar(timestamps, faces_detected, width=0.1, alpha=0.7, color='green')
            axes[1].set_ylabel('Faces Detected')
            axes[1].set_title('Number of Faces Detected Per Frame')
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Deepfake detection flags
            axes[2].fill_between(timestamps, deepfake_flags, alpha=0.7, color='red', step='mid')
            axes[2].set_ylabel('Deepfake Detected')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_title('Deepfake Detection Timeline')
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = output_dir / f"{results['video_info']['filename']}_analysis_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Analysis plots saved to: {plot_file}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")

# Test the video analyzer
if __name__ == "__main__":
    print("Testing Video Deepfake Analyzer...")
    
    analyzer = VideoDeepfakeAnalyzer()
    
    if analyzer.model:
        print("Video analyzer ready")
        print("Usage: analyzer.analyze_video('path/to/video.mp4')")
    else:
        print("Model not loaded - check model path")