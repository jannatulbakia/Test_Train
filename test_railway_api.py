"""
Python script to test Railway API endpoints
"""

import requests
import json

RAILWAY_URL = "https://web-production-e5415.up.railway.app"

print("Testing Railway API endpoints...")
print("=" * 80)

# 1. Root endpoint
print("\n1. Root endpoint (GET /)")
try:
    response = requests.get(f"{RAILWAY_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

# 2. Health check
print("\n2. Health check (GET /health)")
try:
    response = requests.get(f"{RAILWAY_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

# 3. Get features
print("\n3. Get features (GET /features)")
try:
    response = requests.get(f"{RAILWAY_URL}/features")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

# 4. Get model info
print("\n4. Get model info (GET /model/info)")
try:
    response = requests.get(f"{RAILWAY_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

# 5. Make prediction
print("\n5. Make prediction (POST /predict)")
try:
    data = {
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
    response = requests.post(f"{RAILWAY_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("Done!")

