"""
Feature analysis for deepfake detection dataset
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from collections import defaultdict

class DatasetAnalyzer:
    """Analyze features and characteristics of the deepfake dataset"""
    
    def __init__(self):
        self.stats = defaultdict(dict)
        
    def analyze_image_properties(self, dataset_dir: Path):
        """Analyze basic image properties across the dataset"""
        dataset_dir = Path(dataset_dir)
        
        results = {
            'brightness': {'real': [], 'fake': []},
            'contrast': {'real': [], 'fake': []},
            'blur': {'real': [], 'fake': []},
            'edge_density': {'real': [], 'fake': []}
        }
        
        # Analyze train split
        train_dir = dataset_dir / "train"
        
        for label in ['real', 'fake']:
            label_dir = train_dir / label
            if not label_dir.exists():
                continue
                
            images = list(label_dir.glob('*.jpg'))
            print(f"Analyzing {len(images)} {label} training images...")
            
            # Sample subset for analysis (to save time)
            sample_size = min(50, len(images))
            sample_images = images[:sample_size]
            
            for img_path in sample_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Brightness analysis
                brightness = np.mean(gray)
                results['brightness'][label].append(brightness)
                
                # Contrast analysis
                contrast = np.std(gray)
                results['contrast'][label].append(contrast)
                
                # Blur analysis (Laplacian variance)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                results['blur'][label].append(blur_score)
                
                # Edge density
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                results['edge_density'][label].append(edge_density)
        
        return results
    
    def create_feature_visualizations(self, analysis_results: dict, output_dir: Path):
        """Create visualizations of feature differences"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        
        # Create comparison plots
        features = ['brightness', 'contrast', 'blur', 'edge_density']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            real_data = analysis_results[feature]['real']
            fake_data = analysis_results[feature]['fake']
            
            if len(real_data) > 0 and len(fake_data) > 0:
                # Create histogram comparison
                ax.hist(real_data, alpha=0.6, label='Real', bins=20, color='blue')
                ax.hist(fake_data, alpha=0.6, label='Fake', bins=20, color='red')
                
                ax.set_title(f'{feature.capitalize()} Distribution')
                ax.set_xlabel(feature.capitalize())
                ax.set_ylabel('Frequency')
                ax.legend()
                
                # Add statistics text
                real_mean = np.mean(real_data)
                fake_mean = np.mean(fake_data)
                ax.text(0.05, 0.95, f'Real mean: {real_mean:.2f}\nFake mean: {fake_mean:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature analysis plots saved to {output_dir}")
    
    def generate_dataset_report(self, dataset_dir: Path, output_dir: Path):
        """Generate comprehensive dataset analysis report"""
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'dataset_overview': {},
            'feature_statistics': {}
        }
        
        # Dataset overview
        splits = ['train', 'val', 'test']
        for split in splits:
            split_dir = dataset_dir / split
            if split_dir.exists():
                real_count = len(list((split_dir / 'real').glob('*.jpg')))
                fake_count = len(list((split_dir / 'fake').glob('*.jpg')))
                
                report['dataset_overview'][split] = {
                    'real_images': real_count,
                    'fake_images': fake_count,
                    'total_images': real_count + fake_count,
                    'balance_ratio': min(real_count, fake_count) / max(real_count, fake_count) if max(real_count, fake_count) > 0 else 0
                }
        
        # Feature analysis
        print("Running feature analysis...")
        feature_results = self.analyze_image_properties(dataset_dir)
        
        # Calculate feature statistics
        for feature in ['brightness', 'contrast', 'blur', 'edge_density']:
            real_data = feature_results[feature]['real']
            fake_data = feature_results[feature]['fake']
            
            if real_data and fake_data:
                report['feature_statistics'][feature] = {
                    'real_mean': float(np.mean(real_data)),
                    'real_std': float(np.std(real_data)),
                    'fake_mean': float(np.mean(fake_data)),
                    'fake_std': float(np.std(fake_data)),
                    'difference': float(np.mean(fake_data) - np.mean(real_data))
                }
        
        # Create visualizations
        self.create_feature_visualizations(feature_results, output_dir)
        
        # Save report
        with open(output_dir / 'dataset_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n=== Dataset Analysis Summary ===")
        for split, stats in report['dataset_overview'].items():
            print(f"{split.capitalize()}: {stats['total_images']} images "
                  f"(balance: {stats['balance_ratio']:.2f})")
        
        print("\nFeature Differences (Fake - Real):")
        for feature, stats in report['feature_statistics'].items():
            diff = stats['difference']
            print(f"  {feature.capitalize()}: {diff:+.2f}")
        
        return report