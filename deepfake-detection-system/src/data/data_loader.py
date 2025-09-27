"""
Data loading and preprocessing for deepfake detection training
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
from sklearn.utils.class_weight import compute_class_weight

class DeepfakeDataLoader:
    """Data loader for deepfake detection dataset"""
    
    def __init__(self, config_path="config/dataset_config.json"):
        self.config = self._load_config(config_path)
        self.image_size = (224, 224)
        self.batch_size = 32
        self.class_names = ['fake', 'real']  # Note: alphabetical order for tf.keras
        
    def _load_config(self, config_path):
        """Load dataset configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}, using defaults")
            return {
                'file_paths': {
                    'train_dir': 'data/processed/train_ready/train',
                    'val_dir': 'data/processed/train_ready/val',
                    'test_dir': 'data/processed/train_ready/test'
                }
            }
    
    def create_dataset(self, data_dir, is_training=True, shuffle=True):
        """Create TensorFlow dataset from directory"""
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Create dataset from directory
        dataset = keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            class_names=self.class_names,
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=shuffle,
            seed=42
        )
        
        # Apply preprocessing
        if is_training:
            dataset = dataset.map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.map(self._preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Optimize for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _preprocess_data(self, images, labels):
        """Normalize images"""
        # Normalize to [0, 1] range
        images = tf.cast(images, tf.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        images = (images - mean) / std
        
        return images, labels
    
    def _augment_data(self, images, labels):
        """Apply data augmentation during training"""
        # Random horizontal flip
        images = tf.image.random_flip_left_right(images)
        
        # Random brightness
        images = tf.image.random_brightness(images, max_delta=0.1)
        
        # Random contrast
        images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
        
        # Random saturation
        images = tf.image.random_saturation(images, lower=0.9, upper=1.1)
        
        return images, labels
    
    def get_datasets(self):
        """Get train, validation, and test datasets"""
        datasets = {}
        
        # Training dataset
        train_dir = self.config['file_paths']['train_dir']
        if Path(train_dir).exists():
            datasets['train'] = self.create_dataset(train_dir, is_training=True, shuffle=True)
        
        # Validation dataset
        val_dir = self.config['file_paths']['val_dir']
        if Path(val_dir).exists():
            datasets['val'] = self.create_dataset(val_dir, is_training=False, shuffle=False)
        
        # Test dataset
        test_dir = self.config['file_paths']['test_dir']
        if Path(test_dir).exists():
            datasets['test'] = self.create_dataset(test_dir, is_training=False, shuffle=False)
        
        return datasets
    
    def calculate_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        train_dir = Path(self.config['file_paths']['train_dir'])
        
        if not train_dir.exists():
            return None
        
        # Count samples per class
        class_counts = {}
        for class_name in self.class_names:
            class_dir = train_dir / class_name
            if class_dir.exists():
                class_counts[class_name] = len(list(class_dir.glob('*.jpg')))
        
        # Calculate weights
        total_samples = sum(class_counts.values())
        class_weights = {}
        
        for i, class_name in enumerate(self.class_names):
            if class_name in class_counts:
                weight = total_samples / (len(self.class_names) * class_counts[class_name])
                class_weights[i] = weight
        
        print(f"Class weights: {class_weights}")
        return class_weights
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        info = {
            'class_names': self.class_names,
            'image_size': self.image_size,
            'batch_size': self.batch_size
        }
        
        # Count samples in each split
        for split in ['train', 'val', 'test']:
            split_dir = Path(self.config['file_paths'][f'{split}_dir'])
            if split_dir.exists():
                split_info = {}
                total = 0
                
                for class_name in self.class_names:
                    class_dir = split_dir / class_name
                    if class_dir.exists():
                        count = len(list(class_dir.glob('*.jpg')))
                        split_info[class_name] = count
                        total += count
                
                split_info['total'] = total
                info[split] = split_info
        
        return info

# Test data loader
if __name__ == "__main__":
    print("Testing data loader...")
    
    # Create data loader
    loader = DeepfakeDataLoader()
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("Dataset info:")
    for split, data in info.items():
        if isinstance(data, dict) and 'total' in data:
            print(f"  {split}: {data}")
    
    # Test dataset creation
    try:
        datasets = loader.get_datasets()
        
        if 'train' in datasets:
            print(f"Training dataset created successfully")
            
            # Check first batch
            for batch_images, batch_labels in datasets['train'].take(1):
                print(f"Batch shape: {batch_images.shape}")
                print(f"Labels shape: {batch_labels.shape}")
                break
        
        print("Data loader test completed successfully")
        
    except Exception as e:
        print(f"Error testing data loader: {e}")