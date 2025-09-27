#!/usr/bin/env python3
"""
Train model with properly split data (no leakage)
"""

import sys
sys.path.append('src')

from models.deepfake_cnn import DeepfakeCNN
from data.fixed_data_loader import FixedDataLoader
from training.trainer import DeepfakeTrainer

def main():
    print("Training with Fixed Data Splits (No Leakage)")
    print("=" * 50)
    
    # Use fixed data loader
    loader = FixedDataLoader()
    info = loader.get_dataset_info()
    
    print("Dataset with proper splits:")
    for split, data in info.items():
        if isinstance(data, dict) and 'total' in data:
            print(f"  {split}: {data['total']} images")
    
    datasets = loader.get_datasets()
    
    if 'train' not in datasets:
        print("No training data found")
        return
    
    # Create model
    cnn = DeepfakeCNN()
    model = cnn.create_simple_cnn()
    cnn.compile_model()
    
    # Train with more realistic expectations
    config = {
        'epochs': 8,
        'early_stopping_patience': 4
    }
    
    trainer = DeepfakeTrainer(cnn.model, loader, config)
    
    print("Starting training with proper data splits...")
    history = trainer.train(datasets)
    
    if history:
        final_acc = history.history['accuracy'][-1]
        val_acc = history.history.get('val_accuracy', [0])[-1]
        print(f"\nRealistic Results:")
        print(f"Final Training Accuracy: {final_acc:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        print("(These should be more realistic now)")

if __name__ == "__main__":
    main()