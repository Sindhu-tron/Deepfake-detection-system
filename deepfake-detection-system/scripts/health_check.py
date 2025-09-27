#!/usr/bin/env python3
"""
Production health check script
"""

import requests
import sys
import time
import json

def check_service(name, url, timeout=10):
    """Check if a service is healthy"""
    try:
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            print(f"✅ {name}: Healthy")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {name}: {str(e)}")
        return False

def main():
    print("🏥 Production Health Check")
    print("=" * 40)
    
    services = [
        ("Web App", "http://localhost/"),
        ("API Health", "http://localhost/api/health"),
        ("Direct API", "http://localhost:5001/health"),
        ("Direct Web", "http://localhost:5000/")
    ]
    
    healthy_count = 0
    
    for name, url in services:
        if check_service(name, url):
            healthy_count += 1
        time.sleep(1)
    
    print(f"\n📊 Status: {healthy_count}/{len(services)} services healthy")
    
    if healthy_count == len(services):
        print("🎉 All services are healthy!")
        return 0
    else:
        print("⚠️  Some services are unhealthy")
        return 1

if __name__ == "__main__":
    sys.exit(main())