#!/usr/bin/env python3
print("=== Testing Day 2 Validation ===")

try:
    print("1. Testing basic imports...")
    import sys
    import json
    from pathlib import Path
    print("   ✅ Basic imports work")
    
    print("2. Testing path setup...")
    src_path = Path(__file__).parent / "src"
    preprocessing_path = Path(__file__).parent / "preprocessing"
    if str(src_path.resolve()) not in sys.path:
        sys.path.append(str(src_path.resolve()))
    if str(preprocessing_path.resolve()) not in sys.path:
        sys.path.append(str(preprocessing_path.resolve()))
    # Ensure preprocessing is a package
    init_file = preprocessing_path / "__init__.py"
    if not init_file.exists():
        init_file.touch()
    print(f"   ✅ Path setup complete: {src_path.resolve()}, {preprocessing_path.resolve()}")
    
    print("3. Testing directory check...")
    sample_dir = Path("data/raw/sample_dataset/real")
    print(f"   Sample directory exists: {sample_dir.exists()}")
    
    if sample_dir.exists():
        videos = list(sample_dir.glob("*.mp4"))
        print(f"   Found {len(videos)} videos")
    
    print("4. Testing preprocessing imports...")
    try:
        from preprocessing.video_utils import VideoProcessor
        print("   ✅ VideoProcessor imported")
    except Exception as e:
        print(f"   ❌ VideoProcessor import failed: {e}")
    
    try:
        from preprocessing.face_detection import FaceDetector
        print("   ✅ FaceDetector imported")
    except Exception as e:
        print(f"   ❌ FaceDetector import failed: {e}")
    
    print("\n=== Test Complete ===")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

