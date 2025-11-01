#!/bin/bash
# Test script for Railway API endpoints
# Replace YOUR_RAILWAY_URL with your actual Railway URL

RAILWAY_URL="https://web-production-e5415.up.railway.app"

echo "Testing Railway API endpoints..."
echo "=========================================="

echo -e "\n1. Root endpoint (GET /)"
curl $RAILWAY_URL/

echo -e "\n\n2. Health check (GET /health)"
curl $RAILWAY_URL/health

echo -e "\n\n3. Get features (GET /features)"
curl $RAILWAY_URL/features

echo -e "\n\n4. Get model info (GET /model/info)"
curl $RAILWAY_URL/model/info

echo -e "\n\n5. Make prediction (POST /predict)"
curl -X POST $RAILWAY_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

echo -e "\n\nDone!"

