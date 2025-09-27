#!/usr/bin/env python3
"""Test script to verify the development environment setup"""

import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import cv2 # type: ignore
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp # type: ignore
        print(f"✅ MediaPipe: {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import sklearn # type: ignore
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import numpy as np
        import pandas as pd # type: ignore
        import matplotlib.pyplot as plt # type: ignore
        print("✅ NumPy, Pandas, Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Data science libraries import failed: {e}")
        return False
    
    return True

def test_tensorflow_gpu():
    """Test TensorFlow GPU availability"""
    import tensorflow as tf
    
    print("\nTesting TensorFlow setup...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} GPU(s) found")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
        
        # Test GPU computation
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
            print("✅ GPU computation test successful")
        except Exception as e:
            print(f"⚠️ GPU computation test failed: {e}")
    else:
        print("⚠️ No GPU found - will use CPU (slower but still works)")
    
    # Test basic computation
    try:
        x = tf.constant([1, 2, 3, 4, 5])
        y = tf.constant([2, 3, 4, 5, 6])
        z = tf.add(x, y)
        print("✅ TensorFlow basic computation working")
    except Exception as e:
        print(f"❌ TensorFlow computation failed: {e}")
        return False
    
    return True

def test_opencv():
    """Test OpenCV functionality"""
    import cv2 # type: ignore
    import numpy as np
    
    print("\nTesting OpenCV...")
    
    try:
        # Test basic image operations
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (100, 150, 200)  # BGR color
        
        # Test basic operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        print("✅ OpenCV basic operations working")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe face detection"""
    import mediapipe as mp # type: ignore
    import cv2 # type: ignore
    import numpy as np
    
    print("\nTesting MediaPipe...")
    
    try:
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Create a dummy image (just to test the initialization)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test processing (won't find faces in black image, but should not crash)
        results = face_detection.process(dummy_img)
        
        print("✅ MediaPipe face detection initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_directory_structure():
    """Test that directory structure was created correctly"""
    from pathlib import Path
    
    print("\nTesting directory structure...")
    
    expected_dirs = [
        'data/raw',
        'data/processed', 
        'data/models',
        'src/models',
        'src/preprocessing',
        'src/utils',
        'notebooks',
        'tests'
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("="*50)
    print("DEVELOPMENT ENVIRONMENT SETUP TEST")
    print("="*50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test TensorFlow
    if test_tensorflow_gpu():
        tests_passed += 1
    
    # Test OpenCV
    if test_opencv():
        tests_passed += 1
    
    # Test MediaPipe
    if test_mediapipe():
        tests_passed += 1
    
    # Test directory structure
    if test_directory_structure():
        tests_passed += 1
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 SUCCESS: Your development environment is ready!")
        print("\nNext steps:")
        print("1. You can now start downloading datasets")
        print("2. Create some test notebooks in the 'notebooks' directory")
        print("3. Begin implementing the preprocessing pipeline")
    else:
        print("⚠️  ISSUES FOUND: Fix the failed tests before proceeding")
        print("\nCommon solutions:")
        if tests_passed < 2:
            print("- Reinstall packages: pip install -r requirements.txt")
        print("- Check that your virtual environment is activated")
        print("- On Mac/Linux: install system dependencies with brew/apt")
        print("- On Windows: install Visual Studio Build Tools")
    
    print("="*50)

if __name__ == "__main__":
    main()
EOF # type: ignore