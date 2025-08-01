#!/usr/bin/env python3
"""
Test script for Een Consciousness API
"""

import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health_check() -> bool:
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_registration() -> bool:
    """Test user registration"""
    try:
        data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPass123!",
            "confirm_password": "TestPass123!"
        }
        response = requests.post(f"{BASE_URL}/auth/register", json=data)
        if response.status_code == 200:
            print("✅ User registration passed")
            return True
        else:
            print(f"❌ User registration failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ User registration error: {e}")
        return False

def test_login() -> str:
    """Test user login and return token"""
    try:
        data = {
            "username": "user",
            "password": "user123"
        }
        response = requests.post(f"{BASE_URL}/auth/login", json=data)
        if response.status_code == 200:
            token_data = response.json()
            print("✅ User login passed")
            return token_data["access_token"]
        else:
            print(f"❌ User login failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ User login error: {e}")
        return None

def test_authenticated_endpoint(token: str) -> bool:
    """Test authenticated endpoint"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/api/consciousness/status", headers=headers)
        if response.status_code == 200:
            print("✅ Authenticated endpoint passed")
            return True
        else:
            print(f"❌ Authenticated endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Authenticated endpoint error: {e}")
        return False

def test_consciousness_endpoints(token: str) -> bool:
    """Test consciousness endpoints"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test consciousness process
        data = {
            "input_data": "test consciousness data",
            "consciousness_type": "unity"
        }
        response = requests.post(f"{BASE_URL}/api/consciousness/process", json=data, headers=headers)
        if response.status_code == 200:
            print("✅ Consciousness process endpoint passed")
        else:
            print(f"❌ Consciousness process failed: {response.status_code}")
            return False
        
        # Test unity evaluation
        data = {
            "equation": "1 + 1 = 1",
            "parameters": {}
        }
        response = requests.post(f"{BASE_URL}/api/consciousness/unity/evaluate", json=data, headers=headers)
        if response.status_code == 200:
            print("✅ Unity evaluation endpoint passed")
        else:
            print(f"❌ Unity evaluation failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Consciousness endpoints error: {e}")
        return False

def test_agent_endpoints(token: str) -> bool:
    """Test agent endpoints"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test agent chat
        data = {
            "message": "Hello, consciousness agent!",
            "agent_type": "chat"
        }
        response = requests.post(f"{BASE_URL}/api/agents/chat", json=data, headers=headers)
        if response.status_code == 200:
            print("✅ Agent chat endpoint passed")
        else:
            print(f"❌ Agent chat failed: {response.status_code}")
            return False
        
        # Test agent list
        response = requests.get(f"{BASE_URL}/api/agents/list", headers=headers)
        if response.status_code == 200:
            print("✅ Agent list endpoint passed")
        else:
            print(f"❌ Agent list failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Agent endpoints error: {e}")
        return False

def test_visualization_endpoints(token: str) -> bool:
    """Test visualization endpoints"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test unity proof visualization
        response = requests.get(f"{BASE_URL}/api/visualizations/unity-proof", headers=headers)
        if response.status_code == 200:
            print("✅ Unity proof visualization endpoint passed")
        else:
            print(f"❌ Unity proof visualization failed: {response.status_code}")
            return False
        
        # Test available visualizations
        response = requests.get(f"{BASE_URL}/api/visualizations/available", headers=headers)
        if response.status_code == 200:
            print("✅ Available visualizations endpoint passed")
        else:
            print(f"❌ Available visualizations failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Visualization endpoints error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Testing Een Consciousness API...")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("❌ API is not running or not accessible")
        sys.exit(1)
    
    # Test registration
    test_registration()
    
    # Test login
    token = test_login()
    if not token:
        print("❌ Cannot proceed without authentication token")
        sys.exit(1)
    
    # Test authenticated endpoints
    if not test_authenticated_endpoint(token):
        print("❌ Authentication test failed")
        sys.exit(1)
    
    # Test consciousness endpoints
    if not test_consciousness_endpoints(token):
        print("❌ Consciousness endpoints test failed")
        sys.exit(1)
    
    # Test agent endpoints
    if not test_agent_endpoints(token):
        print("❌ Agent endpoints test failed")
        sys.exit(1)
    
    # Test visualization endpoints
    if not test_visualization_endpoints(token):
        print("❌ Visualization endpoints test failed")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 All tests passed! API is working correctly.")
    print(f"📚 API Documentation: {BASE_URL}/docs")
    print(f"🔍 ReDoc Documentation: {BASE_URL}/redoc")

if __name__ == "__main__":
    main() 