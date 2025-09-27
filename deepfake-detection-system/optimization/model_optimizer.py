"""
Model optimization tools for better performance
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time
import sys

sys.path.append('..')

class ModelOptimizer:
    """Optimize trained models for deployment"""
    
    def __init__(self, model_path='../training_outputs/models/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if Path(self.model_path).exists():
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Optimizer: Model loaded from {self.model_path}")
                return True
            else:
                print(f"Optimizer: Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Optimizer: Error loading model: {e}")
            return False
    
    def quantize_model(self, output_path=None):
        """Convert model to quantized version for faster inference"""
        if self.model is None:
            return None
        
        if output_path is None:
            output_path = Path(self.model_path).parent / "optimized_model_quantized.tflite"
        
        print("Converting model to TensorFlow Lite with quantization...")
        
        try:
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Enable quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Set supported types for quantization
            converter.target_spec.supported_types = [tf.float16]
            
            # Representative dataset for better quantization
            def representative_dataset():
                for _ in range(100):
                    yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Quantized model saved to: {output_path}")
            
            # Compare sizes
            original_size = Path(self.model_path).stat().st_size / (1024 * 1024)
            quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
            
            print(f"Original model size: {original_size:.2f} MB")
            print(f"Quantized model size: {quantized_size:.2f} MB")
            print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
            
            return output_path
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            return None
    
    def create_savedmodel_format(self, output_dir=None):
        """Convert to SavedModel format for serving"""
        if self.model is None:
            return None
        
        if output_dir is None:
            output_dir = Path(self.model_path).parent / "optimized_savedmodel"
        
        print("Converting to SavedModel format...")
        
        try:
            # Save in SavedModel format
            tf.saved_model.save(self.model, str(output_dir))
            
            print(f"SavedModel saved to: {output_dir}")
            return output_dir
            
        except Exception as e:
            print(f"SavedModel conversion failed: {e}")
            return None
    
    def benchmark_optimized_models(self):
        """Benchmark different model formats"""
        if self.model is None:
            print("Original model not loaded")
            return
        
        print("Benchmarking model formats...")
        
        # Create test data
        test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        warmup_runs = 10
        benchmark_runs = 50
        
        results = {}
        
        # Benchmark original model
        print("  Testing original Keras model...")
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.model.predict(test_input, verbose=0)
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.time()
            predictions = self.model.predict(test_input, verbose=0)
            end = time.time()
            times.append(end - start)
        
        results['original'] = {
            'avg_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'model_size_mb': Path(self.model_path).stat().st_size / (1024 * 1024)
        }
        
        # Benchmark quantized model if available
        quantized_path = Path(self.model_path).parent / "optimized_model_quantized.tflite"
        
        if quantized_path.exists():
            print("  Testing quantized TFLite model...")
            
            try:
                # Load TFLite model
                interpreter = tf.lite.Interpreter(model_path=str(quantized_path))
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Warmup
                for _ in range(warmup_runs):
                    interpreter.set_tensor(input_details[0]['index'], test_input)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_details[0]['index'])
                
                # Benchmark
                times = []
                for _ in range(benchmark_runs):
                    start = time.time()
                    interpreter.set_tensor(input_details[0]['index'], test_input)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details[0]['index'])
                    end = time.time()
                    times.append(end - start)
                
                results['quantized'] = {
                    'avg_time_ms': np.mean(times) * 1000,
                    'std_time_ms': np.std(times) * 1000,
                    'model_size_mb': quantized_path.stat().st_size / (1024 * 1024)
                }
                
            except Exception as e:
                print(f"Error benchmarking quantized model: {e}")
        
        # Display results
        print("\n" + "="*50)
        print("MODEL FORMAT BENCHMARK RESULTS")
        print("="*50)
        
        for model_type, stats in results.items():
            print(f"\n{model_type.upper()} MODEL:")
            print(f"  Average inference time: {stats['avg_time_ms']:.2f} ± {stats['std_time_ms']:.2f} ms")
            print(f"  Model size: {stats['model_size_mb']:.2f} MB")
            print(f"  Throughput: {1000 / stats['avg_time_ms']:.1f} images/second")
        
        # Calculate improvements
        if 'quantized' in results and 'original' in results:
            speed_improvement = (results['original']['avg_time_ms'] / results['quantized']['avg_time_ms'])
            size_reduction = (1 - results['quantized']['model_size_mb'] / results['original']['model_size_mb'])
            
            print(f"\nQUANTIZATION IMPROVEMENTS:")
            print(f"  Speed improvement: {speed_improvement:.1f}x faster")
            print(f"  Size reduction: {size_reduction*100:.1f}% smaller")
        
        return results
    
    def optimize_for_deployment(self):
        """Create optimized versions for different deployment scenarios"""
        print("Creating optimized models for deployment...")
        
        optimization_results = {
            'timestamp': time.time(),
            'original_model': self.model_path,
            'optimizations': {}
        }
        
        # Quantized model for mobile/edge deployment
        quantized_path = self.quantize_model()
        if quantized_path:
            optimization_results['optimizations']['quantized'] = str(quantized_path)
        
        # SavedModel for server deployment
        savedmodel_path = self.create_savedmodel_format()
        if savedmodel_path:
            optimization_results['optimizations']['savedmodel'] = str(savedmodel_path)
        
        # Benchmark all formats
        benchmark_results = self.benchmark_optimized_models()
        optimization_results['benchmark_results'] = benchmark_results
        
        # Save optimization summary
        output_dir = Path(self.model_path).parent
        summary_file = output_dir / "optimization_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        print(f"\nOptimization summary saved to: {summary_file}")
        
        # Provide recommendations
        print(self._get_deployment_recommendations(benchmark_results))
        
        return optimization_results
    
    def _get_deployment_recommendations(self, benchmark_results):
        """Provide deployment recommendations based on benchmark results"""
        recommendations = [
            "\nDEPLOYMENT RECOMMENDATIONS:",
            "=" * 40
        ]
        
        if 'quantized' in benchmark_results:
            recommendations.extend([
                "",
                "📱 MOBILE/EDGE DEPLOYMENT:",
                f"   Use quantized TFLite model ({benchmark_results['quantized']['model_size_mb']:.1f}MB)",
                f"   Inference time: {benchmark_results['quantized']['avg_time_ms']:.1f}ms",
                "   Best for: Mobile apps, IoT devices, resource-constrained environments"
            ])
        
        if 'original' in benchmark_results:
            recommendations.extend([
                "",
                "🖥️  SERVER DEPLOYMENT:",
                f"   Use original Keras model ({benchmark_results['original']['model_size_mb']:.1f}MB)",
                f"   Inference time: {benchmark_results['original']['avg_time_ms']:.1f}ms",
                "   Best for: Server APIs, high-accuracy requirements, GPU acceleration"
            ])
        
        recommendations.extend([
            "",
            "⚡ PERFORMANCE TIPS:",
            "   • Use batch processing for multiple images",
            "   • Enable GPU acceleration if available",
            "   • Consider caching for repeated predictions",
            "   • Monitor memory usage in production"
        ])
        
        return "\n".join(recommendations)

# Test the optimizer
if __name__ == "__main__":
    optimizer = ModelOptimizer()
    
    if optimizer.model:
        print("Starting model optimization...")
        results = optimizer.optimize_for_deployment()
        print("Optimization complete!")
    else:
        print("Cannot optimize - model not loaded")