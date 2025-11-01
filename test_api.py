"""
Simple test script for the API
Run this after starting the Flask server to test predictions
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"✓ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_prediction():
    """Test prediction endpoint"""
    print("\nTesting /predict endpoint...")
    
    test_data = {
        "age": 35,
        "gender": "Male",
        "city": "New York",
        "session_duration": 20.5,
        "pages_viewed": 8,
        "total_clicks": 15,
        "cart_value": 250.50,
        "discount_applied": 1,
        "payment_method": "Credit Card",
        "product_category": "Electronics",
        "device_type": "Desktop",
        "month": "Jan"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Prediction successful!")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    print("\nTesting batch prediction...")
    
    test_data = [
        {
            "age": 25,
            "gender": "Female",
            "city": "Los Angeles",
            "session_duration": 15.0,
            "pages_viewed": 5,
            "total_clicks": 10,
            "cart_value": 150.0,
            "discount_applied": 0,
            "payment_method": "PayPal",
            "product_category": "Clothing",
            "device_type": "Mobile",
            "month": "Feb"
        },
        {
            "age": 45,
            "gender": "Male",
            "city": "Chicago",
            "session_duration": 30.0,
            "pages_viewed": 12,
            "total_clicks": 25,
            "cart_value": 350.0,
            "discount_applied": 1,
            "payment_method": "Credit Card",
            "product_category": "Electronics",
            "device_type": "Desktop",
            "month": "Mar"
        }
    ]
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Batch prediction successful!")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"✗ Batch prediction failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Batch prediction test failed: {e}")
        return False

def test_features():
    """Test features endpoint"""
    print("\nTesting /features endpoint...")
    try:
        response = requests.get(f"{API_URL}/features")
        if response.status_code == 200:
            print("✓ Features retrieved:")
            result = response.json()
            print(f"  Features: {result.get('features', [])}")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Features test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("API TESTING SCRIPT")
    print("=" * 80)
    print("\n⚠ Make sure the Flask server is running on http://localhost:5000")
    print("   Start it with: python app.py\n")
    
    # Install requests if needed
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        import subprocess
        subprocess.check_call(["pip", "install", "requests"])
        import requests
    
    results = []
    results.append(test_health())
    results.append(test_prediction())
    results.append(test_batch_prediction())
    results.append(test_features())
    
    print("\n" + "=" * 80)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 80)

