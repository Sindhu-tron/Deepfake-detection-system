"""
Training pipeline for deepfake detection model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

class DeepfakeTrainer:
    """Training pipeline for deepfake detection"""
    
    def __init__(self, model, data_loader, config=None):
        self.model = model
        self.data_loader = data_loader
        self.config = config or self._default_config()
        self.history = None
        
        # Create output directories
        self.output_dir = Path("training_outputs")
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        
        for directory in [self.output_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _default_config(self):
        """Default training configuration"""
        return {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 3
        }
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = self.models_dir / "best_model.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def train(self, datasets):
        """Execute training process"""
        print("Starting training...")
        
        train_dataset = datasets['train']
        val_dataset = datasets.get('val', None)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Calculate class weights
        class_weights = self.data_loader.calculate_class_weights()
        
        # Start training
        try:
            self.history = self.model.fit(
                train_dataset,
                epochs=self.config['epochs'],
                validation_data=val_dataset,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            print("Training completed!")
            self._save_results()
            return self.history
            
        except Exception as e:
            print(f"Training error: {e}")
            return None
    
    def _save_results(self):
        """Save training results"""
        if self.history is None:
            return
        
        # Save history
        history_data = {
            'history': self.history.history,
            'config': self.config
        }
        
        with open(self.logs_dir / "training_history.json", 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Results saved to: {self.output_dir}")