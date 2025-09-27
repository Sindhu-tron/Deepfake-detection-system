"""
Batch processing system for extracting faces from video datasets
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List
from tqdm import tqdm
from .video_utils import VideoProcessor
from .face_detection import FaceDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchFaceExtractor:
    """Extract faces from entire video datasets in batches"""
    
    def __init__(self, output_dir="data/processed/extracted_faces"):
        self.processor = VideoProcessor()
        self.detector = FaceDetector()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.max_frames_per_video = 50
        self.face_size = (224, 224)
        self.min_face_size = 50  # Minimum face size to keep
        
    def extract_faces_from_video(self, video_path: Path, label: str) -> Dict:
        """Extract all faces from a single video"""
        try:
            # Create output directory for this video
            video_name = video_path.stem
            video_output_dir = self.output_dir / label / video_name
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            frames = self.processor.extract_frames(
                video_path, 
                max_frames=self.max_frames_per_video,
                frame_interval=2  # Every 2nd frame
            )
            
            extracted_faces = []
            face_count = 0
            
            for frame_idx, frame in enumerate(frames):
                # Detect faces in frame
                faces = self.detector.detect_faces(frame)
                
                for face_idx, bbox in enumerate(faces):
                    x, y, w, h = bbox
                    
                    # Skip very small faces
                    if w < self.min_face_size or h < self.min_face_size:
                        continue
                    
                    # Extract face
                    face_crop = self.detector.extract_face(
                        frame, bbox, 
                        target_size=self.face_size,
                        padding=0.3
                    )
                    
                    if face_crop is not None:
                        # Save face crop
                        face_filename = f"{video_name}_f{frame_idx:03d}_face{face_idx}.jpg"
                        face_path = video_output_dir / face_filename
                        
                        cv2.imwrite(str(face_path), face_crop)
                        
                        # Store metadata
                        extracted_faces.append({
                            'filename': face_filename,
                            'path': str(face_path),
                            'frame_idx': frame_idx,
                            'face_idx': face_idx,
                            'bbox': bbox,
                            'face_size': (w, h),
                            'label': label
                        })
                        
                        face_count += 1
            
            # Create metadata for this video
            video_metadata = {
                'video_path': str(video_path),
                'video_name': video_name,
                'label': label,
                'frames_processed': len(frames),
                'faces_extracted': face_count,
                'faces': extracted_faces
            }
            
            # Save metadata
            metadata_path = video_output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(video_metadata, f, indent=2)
            
            logger.info(f"Extracted {face_count} faces from {video_path.name}")
            return video_metadata
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return {'error': str(e), 'video_path': str(video_path)}
    
    def process_dataset(self, dataset_dir: Path, max_videos_per_label=None) -> Dict:
        """Process entire dataset directory"""
        dataset_dir = Path(dataset_dir)
        
        # Find real and fake video directories
        real_dir = dataset_dir / "real"
        fake_dir = dataset_dir / "fake"
        
        processing_stats = {
            'dataset_dir': str(dataset_dir),
            'real_videos': [],
            'fake_videos': [],
            'total_faces_extracted': 0,
            'processing_errors': []
        }
        
        # Process real videos
        if real_dir.exists():
            real_videos = self.processor.find_videos(real_dir)
            if max_videos_per_label:
                real_videos = real_videos[:max_videos_per_label]
            
            logger.info(f"Processing {len(real_videos)} real videos...")
            
            for video_path in real_videos:
                result = self.extract_faces_from_video(video_path, "real")
                
                if 'error' in result:
                    processing_stats['processing_errors'].append(result)
                else:
                    processing_stats['real_videos'].append(result)
                    processing_stats['total_faces_extracted'] += result['faces_extracted']
        
        # Process fake videos
        if fake_dir.exists():
            fake_videos = self.processor.find_videos(fake_dir)
            if max_videos_per_label:
                fake_videos = fake_videos[:max_videos_per_label]
            
            logger.info(f"Processing {len(fake_videos)} fake videos...")
            
            for video_path in fake_videos:
                result = self.extract_faces_from_video(video_path, "fake")
                
                if 'error' in result:
                    processing_stats['processing_errors'].append(result)
                else:
                    processing_stats['fake_videos'].append(result)
                    processing_stats['total_faces_extracted'] += result['faces_extracted']
        
        # Save overall statistics
        stats_path = self.output_dir / "extraction_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(processing_stats, f, indent=2)
        
        # Print summary
        total_real_faces = sum(v['faces_extracted'] for v in processing_stats['real_videos'])
        total_fake_faces = sum(v['faces_extracted'] for v in processing_stats['fake_videos'])
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"Real videos processed: {len(processing_stats['real_videos'])}")
        print(f"Fake videos processed: {len(processing_stats['fake_videos'])}")
        print(f"Total faces extracted: {processing_stats['total_faces_extracted']}")
        print(f"  - Real faces: {total_real_faces}")
        print(f"  - Fake faces: {total_fake_faces}")
        print(f"Processing errors: {len(processing_stats['processing_errors'])}")
        print(f"Results saved to: {self.output_dir}")
        
        return processing_stats