#!/usr/bin/env python3
"""
Test model inference on individual images
"""

import sys
sys.path.append('src')

import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model inference"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_image(model, image_path):
    """Predict if image is real or fake"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get probabilities
        fake_prob = prediction[0][0]
        real_prob = prediction[0][1]
        
        # Determine class
        predicted_class = 'real' if real_prob > fake_prob else 'fake'
        confidence = max(real_prob, fake_prob)
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'fake_probability': float(fake_prob),
            'real_probability': float(real_prob)
        }
        
    except Exception as e:
        return {'error': str(e)}

def main():
    print("Testing Model Inference")
    print("="*30)
    
    # Load trained model
    model_path = Path("training_outputs/models/best_model.h5")
    
    if not model_path.exists():
        print("No trained model found!")
        print("Run training first: python simple_train.py")
        return
    
    try:
        model = keras.models.load_model(str(model_path))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test with some images from test set
    test_dir = Path("data/processed/train_ready/test")
    
    if not test_dir.exists():
        print("No test images found")
        return
    
    # Test real images
    real_dir = test_dir / "real"
    if real_dir.exists():
        real_images = list(real_dir.glob("*.jpg"))[:3]
        
        print("\nTesting REAL images:")
        for img_path in real_images:
            result = predict_image(model, img_path)
            
            if 'error' not in result:
                print(f"  {img_path.name}:")
                print(f"    Predicted: {result['class'].upper()}")
                print(f"    Confidence: {result['confidence']:.3f}")
                print(f"    Real prob: {result['real_probability']:.3f}")
                
                # Check if correct
                correct = "✅" if result['class'] == 'real' else "❌"
                print(f"    {correct}")
            else:
                print(f"  Error with {img_path.name}: {result['error']}")
    
    # Test fake images
    fake_dir = test_dir / "fake"
    if fake_dir.exists():
        fake_images = list(fake_dir.glob("*.jpg"))[:3]
        
        print("\nTesting FAKE images:")
        for img_path in fake_images:
            result = predict_image(model, img_path)
            
            if 'error' not in result:
                print(f"  {img_path.name}:")
                print(f"    Predicted: {result['class'].upper()}")
                print(f"    Confidence: {result['confidence']:.3f}")
                print(f"    Fake prob: {result['fake_probability']:.3f}")
                
                # Check if correct
                correct = "✅" if result['class'] == 'fake' else "❌"
                print(f"    {correct}")
            else:
                print(f"  Error with {img_path.name}: {result['error']}")
    
    print("\nInference testing complete!")

if __name__ == "__main__":
    main()