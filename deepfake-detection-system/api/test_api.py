"""
Test script for deepfake detection API
"""

import requests
import base64
import json
from pathlib import Path

class APITester:
    def __init__(self, api_url='http://localhost:5001'):
        self.api_url = api_url
    
    def test_health(self):
        """Test health endpoint"""
        print("Testing health endpoint...")
        try:
            response = requests.get(f"{self.api_url}/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def test_api_info(self):
        """Test API info endpoint"""
        print("\nTesting API info...")
        try:
            response = requests.get(f"{self.api_url}/")
            print(f"Status: {response.status_code}")
            print(f"API Info: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"API info failed: {e}")
            return False
    
    def encode_image_to_base64(self, image_path):
        """Convert image file to base64"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def test_single_prediction_json(self, image_path):
        """Test single prediction with JSON payload"""
        print(f"\nTesting single prediction (JSON) with {image_path}...")
        
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return False
        
        try:
            payload = {'image': base64_image}
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                return False
            else:
                print(f"Prediction: {result['prediction']['class']}")
                print(f"Confidence: {result['prediction']['confidence']:.3f}")
                return True
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            return False
    
    def test_single_prediction_file(self, image_path):
        """Test single prediction with file upload"""
        print(f"\nTesting single prediction (File) with {image_path}...")
        
        try:
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                response = requests.post(
                    f"{self.api_url}/predict",
                    files=files
                )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                return False
            else:
                print(f"Prediction: {result['prediction']['class']}")
                print(f"Confidence: {result['prediction']['confidence']:.3f}")
                return True
                
        except Exception as e:
            print(f"File prediction failed: {e}")
            return False
    
    def test_batch_prediction(self, image_paths):
        """Test batch prediction"""
        print(f"\nTesting batch prediction with {len(image_paths)} images...")
        
        # Encode all images to base64
        encoded_images = []
        for path in image_paths:
            encoded = self.encode_image_to_base64(path)
            if encoded:
                encoded_images.append(encoded)
        
        if not encoded_images:
            print("No valid images to process")
            return False
        
        try:
            payload = {'images': encoded_images}
            response = requests.post(
                f"{self.api_url}/batch",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                return False
            else:
                print(f"Batch size: {result['batch_size']}")
                for res in result['results']:
                    if 'error' in res:
                        print(f"Image {res['index']}: Error - {res['error']}")
                    else:
                        pred = res['prediction']
                        print(f"Image {res['index']}: {pred['class']} ({pred['confidence']:.3f})")
                return True
                
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        print("=" * 50)
        print("DEEPFAKE DETECTION API TESTS")
        print("=" * 50)
        
        tests_passed = 0
        total_tests = 0
        
        # Basic tests
        total_tests += 1
        if self.test_health():
            tests_passed += 1
        
        total_tests += 1
        if self.test_api_info():
            tests_passed += 1
        
        # Find test images
        test_images = []
        image_dirs = [
            Path("../data/processed/fixed_splits/test/real"),
            Path("../data/processed/fixed_splits/test/fake"),
            Path("../data/processed/train_ready/test/real"),
            Path("../data/processed/train_ready/test/fake")
        ]
        
        for image_dir in image_dirs:
            if image_dir.exists():
                test_images.extend(list(image_dir.glob("*.jpg"))[:2])
                if len(test_images) >= 4:
                    break
        
        if test_images:
            # Single prediction tests
            for image_path in test_images[:2]:
                total_tests += 1
                if self.test_single_prediction_json(image_path):
                    tests_passed += 1
                
                total_tests += 1
                if self.test_single_prediction_file(image_path):
                    tests_passed += 1
            
            # Batch prediction test
            total_tests += 1
            if self.test_batch_prediction(test_images[:3]):
                tests_passed += 1
        else:
            print("No test images found")
        
        # Results
        print(f"\n{'=' * 50}")
        print(f"API TESTS COMPLETE: {tests_passed}/{total_tests} passed")
        print(f"{'=' * 50}")
        
        return tests_passed >= total_tests * 0.8

if __name__ == "__main__":
    tester = APITester()
    tester.run_all_tests()