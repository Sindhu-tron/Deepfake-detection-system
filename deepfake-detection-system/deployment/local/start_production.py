#!/usr/bin/env python3
"""
Local production deployment script
"""

import subprocess
import sys
import time
import signal
import os

class ProductionManager:
    def __init__(self):
        self.processes = []
    
    def start_services(self):
        """Start all production services"""
        print("🚀 Starting Production Deepfake Detection System")
        print("=" * 50)
        
        try:
            # Start with Docker Compose
            print("📦 Starting Docker services...")
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.prod.yml', 
                'up', '--build', '-d'
            ], check=True, capture_output=True, text=True)
            
            print("✅ Docker services started")
            
            # Wait for services to be ready
            print("⏳ Waiting for services to be ready...")
            time.sleep(15)
            
            # Health check
            self.health_check()
            
            print("\n🎉 Production system is running!")
            print("📊 Access points:")
            print("   Web App: http://localhost")
            print("   API: http://localhost/api/")
            print("   Direct API: http://localhost:5001")
            print("   Direct Web: http://localhost:5000")
            
            print("\n📝 To stop: Ctrl+C or run 'docker-compose -f docker-compose.prod.yml down'")
            
            # Keep script running
            self.wait_for_interrupt()
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start services: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
            return False
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.stop_services()
        
        return True
    
    def health_check(self):
        """Run health check"""
        try:
            subprocess.run([sys.executable, 'scripts/health_check.py'], check=True)
        except subprocess.CalledProcessError:
            print("⚠️  Health check failed - but services may still be starting")
    
    def stop_services(self):
        """Stop all services"""
        print("🛑 Stopping production services...")
        
        try:
            subprocess.run([
                'docker-compose', '-f', 'docker-compose.prod.yml', 'down'
            ], check=True)
            print("✅ All services stopped")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error stopping services: {e}")
    
    def wait_for_interrupt(self):
        """Wait for user interrupt"""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

def main():
    manager = ProductionManager()
    
    # Register signal handler
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal")
        manager.stop_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = manager.start_services()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())