#!/usr/bin/env python3
"""
Quick system status check
"""

import subprocess
import sys
from pathlib import Path
import requests
import time

def check_file_structure():
    """Check if all required directories exist"""
    required_dirs = ['web_app', 'api', 'video_processor', 'optimization', 'src']
    required_files = [
        'web_app/app.py',
        'api/deepfake_api.py', 
        'video_processor/video_analyzer.py',
        'optimization/performance_benchmark.py',
        'training_outputs/models/best_model.h5'
    ]
    
    print("📁 File Structure Check:")
    
    all_good = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - MISSING")
            all_good = False
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_good = False
    
    return all_good

def check_model():
    """Check if model loads correctly"""
    print("\n🤖 Model Loading Check:")
    
    try:
        import tensorflow as tf
        model_path = 'training_outputs/models/best_model.h5'
        
        if Path(model_path).exists():
            model = tf.keras.models.load_model(model_path)
            print(f"  ✅ Model loads successfully ({model.count_params():,} params)")
            return True
        else:
            print(f"  ❌ Model file not found: {model_path}")
            return False
    except Exception as e:
        print(f"  ❌ Model loading error: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Dependencies Check:")
    
    required_packages = [
        'tensorflow', 'flask', 'opencv-python', 
        'scikit-learn', 'matplotlib', 'pillow'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"  ✅ opencv-python")
            elif package == 'pillow':
                import PIL
                print(f"  ✅ pillow")
            else:
                __import__(package)
                print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            all_good = False
    
    return all_good

def main():
    print("🔍 DEEPFAKE DETECTION SYSTEM STATUS CHECK")
    print("=" * 50)
    
    # Run all checks
    structure_ok = check_file_structure()
    model_ok = check_model()
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    print("📊 SYSTEM STATUS SUMMARY")
    print("=" * 50)
    
    if structure_ok and model_ok and deps_ok:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ System is ready for full testing")
        print("\nNext steps:")
        print("1. Run: cd web_app && python app.py")
        print("2. Run: cd api && python deepfake_api.py") 
        print("3. Test: cd tests && python integration_test.py")
        return True
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("❌ Fix issues above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)