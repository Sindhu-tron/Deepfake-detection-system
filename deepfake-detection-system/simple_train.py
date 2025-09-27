#!/usr/bin/env python3
"""
Simple training script for deepfake detection
"""

import sys
sys.path.append('src')

from models.deepfake_cnn import DeepfakeCNN
from data.data_loader import DeepfakeDataLoader
from training.trainer import DeepfakeTrainer

def main():
    print("Starting Deepfake Detection Training")
    print("="*50)
    
    # Create data loader
    print("1. Loading dataset...")
    loader = DeepfakeDataLoader()
    info = loader.get_dataset_info()
    
    for split, data in info.items():
        if isinstance(data, dict) and 'total' in data:
            print(f"   {split}: {data['total']} images")
    
    datasets = loader.get_datasets()
    
    if 'train' not in datasets:
        print("No training data found!")
        return
    
    # Create model
    print("\n2. Creating model...")
    cnn = DeepfakeCNN()
    model = cnn.create_simple_cnn()
    cnn.compile_model()
    
    print(f"   Model parameters: {model.count_params():,}")
    
    # Setup trainer
    print("\n3. Setting up training...")
    config = {
        'epochs': 5,  # Start with few epochs for testing
        'early_stopping_patience': 2
    }
    
    trainer = DeepfakeTrainer(cnn.model, loader, config)
    
    # Start training
    print("\n4. Starting training...")
    history = trainer.train(datasets)
    
    if history:
        final_acc = history.history['accuracy'][-1]
        val_acc = history.history.get('val_accuracy', [0])[-1]
        print(f"\nFinal Training Accuracy: {final_acc:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()