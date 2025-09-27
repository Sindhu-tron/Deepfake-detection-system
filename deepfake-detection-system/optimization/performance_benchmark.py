"""
Performance benchmarking and optimization for deepfake detection system
"""

import time
import psutil
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import json
import sys
import os
from datetime import datetime

# Add project paths
sys.path.append('..')
sys.path.append('../src')

class PerformanceBenchmark:
    """Benchmark system performance"""
    
    def __init__(self, model_path='../training_outputs/models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.benchmark_results = {}
        self.load_model()
    
    def load_model(self):
        """Load the model for benchmarking"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Benchmark: Model loaded from {self.model_path}")
                return True
            else:
                print(f"Benchmark: Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Benchmark: Error loading model: {e}")
            return False
    
    def get_system_info(self):
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'platform': sys.platform,
            'tensorflow_version': tf.__version__,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
        }
    
    def create_test_images(self, batch_sizes=[1, 4, 8, 16, 32]):
        """Create test images for benchmarking"""
        test_images = {}
        
        for batch_size in batch_sizes:
            # Create random test images (224x224x3)
            images = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8)
            
            # Normalize like real preprocessing
            images_normalized = images.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            images_normalized = (images_normalized - mean) / std
            
            test_images[batch_size] = images_normalized
        
        return test_images
    
    def benchmark_inference_speed(self, test_images, warmup_runs=5, benchmark_runs=20):
        """Benchmark model inference speed"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        print("Benchmarking inference speed...")
        results = {}
        
        for batch_size, images in test_images.items():
            print(f"  Testing batch size: {batch_size}")
            
            # Warmup runs
            for _ in range(warmup_runs):
                _ = self.model.predict(images, verbose=0)
            
            # Benchmark runs
            times = []
            memory_usage = []
            
            for _ in range(benchmark_runs):
                # Monitor memory
                mem_before = psutil.virtual_memory().used / (1024**2)  # MB
                
                # Time inference
                start_time = time.time()
                predictions = self.model.predict(images, verbose=0)
                end_time = time.time()
                
                mem_after = psutil.virtual_memory().used / (1024**2)  # MB
                
                times.append(end_time - start_time)
                memory_usage.append(mem_after - mem_before)
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            images_per_second = batch_size / avg_time
            ms_per_image = (avg_time * 1000) / batch_size
            
            results[batch_size] = {
                'batch_size': batch_size,
                'average_time_seconds': avg_time,
                'std_time_seconds': std_time,
                'min_time_seconds': min_time,
                'max_time_seconds': max_time,
                'images_per_second': images_per_second,
                'ms_per_image': ms_per_image,
                'average_memory_usage_mb': np.mean(memory_usage)
            }
        
        return results
    
    def benchmark_preprocessing_speed(self, num_tests=100):
        """Benchmark image preprocessing speed"""
        print("Benchmarking preprocessing speed...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        times = []
        
        for _ in range(num_tests):
            start_time = time.time()
            
            # Simulate preprocessing pipeline
            # Resize
            resized = cv2.resize(test_image, (224, 224))
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize
            normalized = rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            final = (normalized - mean) / std
            
            # Add batch dimension
            batched = np.expand_dims(final, axis=0)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        images_per_second = 1.0 / avg_time
        
        return {
            'average_preprocessing_time_ms': avg_time * 1000,
            'preprocessing_images_per_second': images_per_second,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000
        }
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        print("Benchmarking memory usage...")
        
        # Get initial memory
        initial_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        # Load different batch sizes and monitor memory
        memory_results = {}
        
        for batch_size in [1, 4, 16, 64]:
            # Create test batch
            test_batch = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
            
            mem_before = psutil.virtual_memory().used / (1024**2)
            
            # Run prediction
            predictions = self.model.predict(test_batch, verbose=0)
            
            mem_after = psutil.virtual_memory().used / (1024**2)
            
            memory_results[batch_size] = {
                'memory_before_mb': mem_before,
                'memory_after_mb': mem_after,
                'memory_increase_mb': mem_after - mem_before,
                'memory_per_image_mb': (mem_after - mem_before) / batch_size
            }
        
        return {
            'initial_memory_mb': initial_memory,
            'batch_memory_usage': memory_results
        }
    
    def run_full_benchmark(self):
        """Run complete performance benchmark"""
        print("="*60)
        print("DEEPFAKE DETECTION PERFORMANCE BENCHMARK")
        print("="*60)
        
        start_time = datetime.now()
        
        # System info
        print("Collecting system information...")
        system_info = self.get_system_info()
        
        # Create test data
        print("Creating test data...")
        test_images = self.create_test_images([1, 4, 8, 16])
        
        # Benchmark components
        inference_results = self.benchmark_inference_speed(test_images)
        preprocessing_results = self.benchmark_preprocessing_speed()
        memory_results = self.benchmark_memory_usage()
        
        # Compile results
        self.benchmark_results = {
            'benchmark_info': {
                'timestamp': start_time.isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            },
            'system_info': system_info,
            'inference_performance': inference_results,
            'preprocessing_performance': preprocessing_results,
            'memory_usage': memory_results
        }
        
        # Save results
        output_dir = Path('performance_results')
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / f'benchmark_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        # Display summary
        self._display_results()
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return self.benchmark_results
    
    def _display_results(self):
        """Display benchmark results summary"""
        results = self.benchmark_results
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        # System info
        system = results['system_info']
        print(f"CPU Cores: {system['cpu_count']}")
        print(f"Memory: {system['memory_total_gb']:.1f}GB total, {system['memory_available_gb']:.1f}GB available")
        print(f"GPU Available: {system['gpu_available']}")
        print(f"TensorFlow: {system['tensorflow_version']}")
        
        # Inference performance
        print(f"\nINFERENCE PERFORMANCE:")
        inference = results['inference_performance']
        
        if 'error' not in inference:
            for batch_size, stats in inference.items():
                print(f"  Batch {batch_size}: {stats['ms_per_image']:.1f}ms/image, {stats['images_per_second']:.1f} images/sec")
        
        # Preprocessing performance  
        preprocessing = results['preprocessing_performance']
        print(f"\nPREPROCESSING PERFORMANCE:")
        print(f"  {preprocessing['average_preprocessing_time_ms']:.1f}ms per image")
        print(f"  {preprocessing['preprocessing_images_per_second']:.1f} images/sec")
        
        # Memory usage
        memory = results['memory_usage']
        if 'error' not in memory:
            print(f"\nMEMORY USAGE:")
            print(f"  Base memory: {memory['initial_memory_mb']:.1f}MB")
            
            batch_mem = memory['batch_memory_usage']
            if '1' in batch_mem:
                print(f"  Per image: ~{batch_mem['1']['memory_per_image_mb']:.1f}MB")
    
    def get_optimization_recommendations(self):
        """Provide optimization recommendations based on benchmark"""
        if not self.benchmark_results:
            return ["Run benchmark first"]
        
        recommendations = []
        results = self.benchmark_results
        
        # Check inference speed
        if 'inference_performance' in results and '1' in results['inference_performance']:
            ms_per_image = results['inference_performance']['1']['ms_per_image']
            
            if ms_per_image > 500:
                recommendations.append("Consider model quantization for faster inference")
                recommendations.append("Use batch processing when possible")
            
            if ms_per_image > 100:
                recommendations.append("Consider using TensorFlow Lite for mobile deployment")
        
        # Check memory usage
        if 'memory_usage' in results and 'batch_memory_usage' in results['memory_usage']:
            memory_per_image = results['memory_usage']['batch_memory_usage'].get('1', {}).get('memory_per_image_mb', 0)
            
            if memory_per_image > 50:
                recommendations.append("High memory usage - consider model pruning")
        
        # Check GPU availability
        if not results['system_info']['gpu_available']:
            recommendations.append("Consider GPU acceleration for better performance")
        
        # Check batch efficiency
        if 'inference_performance' in results:
            batch_1_speed = results['inference_performance'].get('1', {}).get('images_per_second', 0)
            batch_16_speed = results['inference_performance'].get('16', {}).get('images_per_second', 0)
            
            if batch_16_speed > batch_1_speed * 8:  # Good batching efficiency
                recommendations.append("Use batch processing for better throughput")
        
        if not recommendations:
            recommendations.append("Performance looks good! No major optimizations needed.")
        
        return recommendations

# Test the benchmark
if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    
    if benchmark.model:
        results = benchmark.run_full_benchmark()
        
        print("\nOPTIMIZATION RECOMMENDATIONS:")
        recommendations = benchmark.get_optimization_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("Cannot run benchmark - model not loaded")