"""
Test security features
"""

import requests
import time
import json

def test_api_security():
    """Test secure API functionality"""
    base_url = "http://localhost:5002"
    
    print("Testing Secure Deepfake Detection API")
    print("=" * 40)
    
    # Test 1: Public endpoints
    print("\n1. Testing public API info...")
    response = requests.get(f"{base_url}/")
    if response.status_code == 200:
        print("✅ Public API info accessible")
        print(f"   API: {response.json()['name']}")
    else:
        print("❌ Public API info failed")
    
    # Test 2: Request API key
    print("\n2. Requesting API key...")
    key_request = {
        "user_name": "Test User",
        "user_email": "test@example.com"
    }
    
    response = requests.post(f"{base_url}/auth/key", json=key_request)
    if response.status_code == 200:
        api_key = response.json()['api_key']
        print(f"✅ API key obtained: {api_key[:20]}...")
    else:
        print("❌ API key request failed")
        return False
    
    # Test 3: Authenticated health check
    print("\n3. Testing authenticated health check...")
    headers = {"X-API-Key": api_key}
    
    response = requests.get(f"{base_url}/health", headers=headers)
    if response.status_code == 200:
        health_data = response.json()
        print("✅ Authenticated health check passed")
        print(f"   User: {health_data['user']['user_name']}")
        print(f"   Rate limit: {health_data['rate_limit']['remaining']}/{health_data['rate_limit']['limit']}")
    else:
        print("❌ Authenticated health check failed")
    
    # Test 4: Unauthenticated request
    print("\n4. Testing unauthenticated request...")
    response = requests.get(f"{base_url}/health")
    if response.status_code == 401:
        print("✅ Unauthenticated request properly rejected")
    else:
        print("❌ Unauthenticated request should be rejected")
    
    # Test 5: Invalid API key
    print("\n5. Testing invalid API key...")
    invalid_headers = {"X-API-Key": "invalid_key_12345"}
    response = requests.get(f"{base_url}/health", headers=invalid_headers)
    if response.status_code == 401:
        print("✅ Invalid API key properly rejected")
    else:
        print("❌ Invalid API key should be rejected")
    
    # Test 6: Rate limiting (make many requests quickly)
    print("\n6. Testing rate limiting...")
    rate_limit_headers = {"X-API-Key": api_key}
    
    # Make multiple requests to test rate limiting
    success_count = 0
    rate_limited = False
    
    for i in range(10):
        response = requests.get(f"{base_url}/health", headers=rate_limit_headers)
        if response.status_code == 200:
            success_count += 1
        elif response.status_code == 429:
            rate_limited = True
            break
        time.sleep(0.1)
    
    print(f"   Made {success_count} successful requests")
    if rate_limited:
        print("✅ Rate limiting activated when appropriate")
    else:
        print("⚠️  Rate limiting not triggered (may need more requests)")
    
    # Test 7: Security headers
    print("\n7. Testing security headers...")
    response = requests.get(f"{base_url}/", headers=headers)
    
    security_headers = [
        'X-Content-Type-Options',
        'X-Frame-Options', 
        'X-XSS-Protection',
        'Content-Security-Policy'
    ]
    
    headers_present = 0
    for header in security_headers:
        if header in response.headers:
            headers_present += 1
    
    print(f"✅ Security headers present: {headers_present}/{len(security_headers)}")
    
    # Test 8: Usage statistics
    print("\n8. Testing usage statistics...")
    response = requests.get(f"{base_url}/stats", headers=rate_limit_headers)
    if response.status_code == 200:
        stats = response.json()
        print("✅ Usage statistics accessible")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
    else:
        print("❌ Usage statistics failed")
    
    print(f"\n📊 Security test completed")
    print(f"🔑 Test API key: {api_key}")
    
    return True

if __name__ == "__main__":
    test_api_security()