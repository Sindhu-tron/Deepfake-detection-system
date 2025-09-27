#!/usr/bin/env python3
"""
Evaluate trained deepfake detection model
"""

import sys
sys.path.append('src')

from models.deepfake_cnn import DeepfakeCNN
from data.data_loader import DeepfakeDataLoader
from evaluation.evaluator import ModelEvaluator
from tensorflow import keras
from pathlib import Path

def main():
    print("="*60)
    print("DEEPFAKE DETECTION MODEL EVALUATION")
    print("="*60)
    
    # Check for trained model
    model_path = Path("training_outputs/models/best_model.h5")
    
    if not model_path.exists():
        print("No trained model found!")
        print("Please run training first with: python simple_train.py")
        return
    
    # Load trained model
    print("Loading trained model...")
    try:
        model = keras.models.load_model(str(model_path))
        print(f"Model loaded: {model.count_params():,} parameters")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load data
    print("\nLoading test datasets...")
    loader = DeepfakeDataLoader()
    datasets = loader.get_datasets()
    
    # Create evaluator
    evaluator = ModelEvaluator(model)
    
    # Evaluate on all available datasets
    results = {}
    
    # Test set evaluation
    if 'test' in datasets:
        print("\n" + "="*50)
        print("EVALUATING ON TEST SET")
        print("="*50)
        
        test_metrics = evaluator.evaluate_model(datasets['test'], 'test')
        results['test'] = test_metrics
    
    # Validation set evaluation
    if 'val' in datasets:
        print("\n" + "="*50)
        print("EVALUATING ON VALIDATION SET")
        print("="*50)
        
        val_metrics = evaluator.evaluate_model(datasets['val'], 'validation')
        results['validation'] = val_metrics
    
    # Training set evaluation (sample)
    if 'train' in datasets:
        print("\n" + "="*50)
        print("EVALUATING ON TRAINING SET (SAMPLE)")
        print("="*50)
        
        # Take only a subset of training data for evaluation
        train_sample = datasets['train'].take(50)  # 50 batches
        train_metrics = evaluator.evaluate_model(train_sample, 'train_sample')
        results['train_sample'] = train_metrics
    
    # Save results
    evaluator.save_evaluation_results()
    
    # Overall summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for dataset, metrics in results.items():
        print(f"\n{dataset.upper()} PERFORMANCE:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    
    # Performance interpretation
    if 'test' in results:
        test_acc = results['test']['accuracy']
        
        if test_acc > 0.9:
            print("\n🌟 EXCELLENT: Model performance is outstanding!")
        elif test_acc > 0.8:
            print("\n✅ GOOD: Model performance is solid!")
        elif test_acc > 0.7:
            print("\n⚠️  ACCEPTABLE: Model performance is adequate")
        else:
            print("\n❌ POOR: Model needs improvement")
        
        print(f"Test Accuracy: {test_acc:.1%}")
    
    print(f"\nDetailed results and plots saved to: evaluation_results/")
    
if __name__ == "__main__":
    main()