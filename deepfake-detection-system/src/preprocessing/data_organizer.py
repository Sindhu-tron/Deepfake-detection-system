"""
Data organization utilities for deepfake detection
"""

import json
import shutil
from pathlib import Path
import pandas as pd # type: ignore
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DatasetOrganizer:
    """Organize and manage deepfake detection datasets"""
    
    def __init__(self, base_data_dir: Path = Path("data")):
        """
        Initialize dataset organizer
        
        Args:
            base_data_dir: Base directory for all data
        """
        self.base_data_dir = Path(base_data_dir)
        self.raw_dir = self.base_data_dir / "raw"
        self.processed_dir = self.base_data_dir / "processed"
        
        # Create directory structure
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create the standard directory structure"""
        directories = [
            self.raw_dir / "sample_dataset" / "real",
            self.raw_dir / "sample_dataset" / "fake", 
            self.raw_dir / "celeb_df" / "real",
            self.raw_dir / "celeb_df" / "fake",
            self.processed_dir / "faces" / "real",
            self.processed_dir / "faces" / "fake",
            self.processed_dir / "train" / "real",
            self.processed_dir / "train" / "fake",
            self.processed_dir / "val" / "real", 
            self.processed_dir / "val" / "fake",
            self.processed_dir / "test" / "real",
            self.processed_dir / "test" / "fake"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created")
    
    def scan_dataset(self, dataset_name: str) -> Dict:
        """
        Scan a dataset and return statistics
        
        Args:
            dataset_name: Name of dataset (e.g., 'sample_dataset', 'celeb_df')
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset_dir = self.raw_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return {}
        
        stats = {
            'dataset_name': dataset_name,
            'dataset_path': str(dataset_dir),
            'real_videos': [],
            'fake_videos': [],
            'total_size_mb': 0
        }
        
        # Scan real videos
        real_dir = dataset_dir / "real"
        if real_dir.exists():
            video_extensions = ['.mp4', '.avi', '.mov', '.MOV']
            for ext in video_extensions:
                videos = list(real_dir.glob(f'*{ext}'))
                for video in videos:
                    stats['real_videos'].append({
                        'filename': video.name,
                        'path': str(video),
                        'size_mb': video.stat().st_size / (1024 * 1024)
                    })
                    stats['total_size_mb'] += video.stat().st_size / (1024 * 1024)
        
        # Scan fake videos
        fake_dir = dataset_dir / "fake"
        if fake_dir.exists():
            for ext in video_extensions:
                videos = list(fake_dir.glob(f'*{ext}'))
                for video in videos:
                    stats['fake_videos'].append({
                        'filename': video.name,
                        'path': str(video),
                        'size_mb': video.stat().st_size / (1024 * 1024)
                    })
                    stats['total_size_mb'] += video.stat().st_size / (1024 * 1024)
        
        # Calculate summary statistics
        stats['counts'] = {
            'real_videos': len(stats['real_videos']),
            'fake_videos': len(stats['fake_videos']),
            'total_videos': len(stats['real_videos']) + len(stats['fake_videos'])
        }
        
        return stats
    
    def generate_dataset_report(self, datasets: List[str] = None) -> Dict:
        """
        Generate comprehensive dataset report
        
        Args:
            datasets: List of dataset names to include
            
        Returns:
            Complete dataset report
        """
        if datasets is None:
            # Find all available datasets
            datasets = [d.name for d in self.raw_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'base_directory': str(self.base_data_dir),
            'datasets': {}
        }
        
        total_videos = 0
        total_size_mb = 0
        
        for dataset_name in datasets:
            stats = self.scan_dataset(dataset_name)
            if stats:
                report['datasets'][dataset_name] = stats
                total_videos += stats['counts']['total_videos']
                total_size_mb += stats['total_size_mb']
        
        report['summary'] = {
            'total_datasets': len([d for d in report['datasets'].values() if d]),
            'total_videos': total_videos,
            'total_size_mb': total_size_mb,
            'total_size_gb': total_size_mb / 1024
        }
        
        return report
    
    def save_dataset_report(self, report: Dict, output_path: Path = None):
        """
        Save dataset report to file
        
        Args:
            report: Report dictionary from generate_dataset_report
            output_path: Where to save the report
        """
        if output_path is None:
            output_path = self.base_data_dir / "dataset_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Dataset report saved to: {output_path}")
        
        # Also create a readable summary
        summary_path = output_path.parent / "dataset_summary.txt"
        self._create_readable_summary(report, summary_path)
    
    def _create_readable_summary(self, report: Dict, output_path: Path):
        """Create human-readable dataset summary"""
        with open(output_path, 'w') as f:
            f.write("DEEPFAKE DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall summary
            summary = report['summary']
            f.write(f"Total Datasets: {summary['total_datasets']}\n")
            f.write(f"Total Videos: {summary['total_videos']}\n")
            f.write(f"Total Size: {summary['total_size_gb']:.2f} GB\n\n")
            
            # Dataset breakdown
            for dataset_name, stats in report['datasets'].items():
                if stats:
                    f.write(f"{dataset_name.upper()}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Real videos: {stats['counts']['real_videos']}\n")
                    f.write(f"Fake videos: {stats['counts']['fake_videos']}\n")
                    f.write(f"Total: {stats['counts']['total_videos']}\n")
                    f.write(f"Size: {stats['total_size_mb']:.1f} MB\n\n")
            
            f.write(f"Report generated: {report['timestamp']}\n")
        
        logger.info(f"Readable summary saved to: {output_path}")

# Test the organizer
if __name__ == "__main__":
    organizer = DatasetOrganizer()
    
    # Generate report
    report = organizer.generate_dataset_report()
    
    # Print summary
    print("Dataset Organization Report")
    print("=" * 40)
    
    if report['datasets']:
        for dataset_name, stats in report['datasets'].items():
            if stats:
                print(f"\n{dataset_name}:")
                print(f"  Real videos: {stats['counts']['real_videos']}")
                print(f"  Fake videos: {stats['counts']['fake_videos']}")
                print(f"  Size: {stats['total_size_mb']:.1f} MB")
    else:
        print("No datasets found")
    
    # Save report
    organizer.save_dataset_report(report)
    print(f"\nReport saved to: {organizer.base_data_dir}/dataset_report.json")
EOF # type: ignore

# Test the organizer
# python src/preprocessing/data_organizer.py