# ML Model API Deployment Guide

This guide will help you deploy your ML model as a REST API that can be used from your React.js application or any other frontend.

## 📋 Table of Contents

1. [Backend API Setup](#backend-api-setup)
2. [Local Development](#local-development)
3. [Production Deployment](#production-deployment)
4. [Using from React.js](#using-from-reactjs)
5. [Testing the API](#testing-the-api)

---

## 🔧 Backend API Setup

### Step 1: Install API Dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, Flask-CORS, Gunicorn, and all ML dependencies.

### Step 2: Train the Model

Make sure you have trained the model first:

```bash
python train_model.py
```

This generates:
- `best_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoders.pkl` - Categorical encoders
- `feature_columns.pkl` - Feature column names

---

## 🚀 Local Development

### Start the API Server

```bash
python app.py
```

The API will run on `http://localhost:5000`

### API Endpoints

- **POST `/predict`** - Make a single prediction
  ```json
  {
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
  ```
  
  **Response:**
  ```json
  {
    "prediction": 1,
    "probability": 0.85,
    "class": "Purchase"
  }
  ```

- **POST `/predict`** (Batch) - Multiple predictions
  ```json
  [
    { "age": 35, "gender": "Male", ... },
    { "age": 28, "gender": "Female", ... }
  ]
  ```

- **GET `/health`** - Health check endpoint
- **GET `/features`** - Get list of required features
- **GET `/model/info`** - Get model information

---

## 🌐 Production Deployment

### Option 1: Deploy to Heroku

1. **Install Heroku CLI**: [https://devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)

2. **Deploy**:
   ```bash
   heroku login
   heroku create your-app-name
   git init
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

3. **Note**: Ensure all `.pkl` model files are committed (or upload them separately)

### Option 2: Deploy to Railway

1. **Install Railway CLI**: `npm i -g @railway/cli`

2. **Deploy**:
   ```bash
   railway login
   railway init
   railway up
   ```

### Option 3: Deploy to DigitalOcean / AWS / GCP (VPS)

1. **Create a VPS** (Ubuntu recommended)

2. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip nginx
   pip3 install -r requirements.txt
   ```

3. **Run with Gunicorn**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

4. **Set up Nginx as reverse proxy** (optional but recommended)

### Option 4: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY *.pkl ./

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t ml-api .
docker run -p 5000:5000 ml-api
```

---

## ⚛️ Using from React.js

Since your React project is in a separate folder, here's how to connect to the API:

### Step 1: Create API Service

In your React project, create `src/services/api.js`:

```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const predictPurchase = async (data) => {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Prediction failed');
  }
  
  return await response.json();
};

export const predictBatch = async (dataArray) => {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(dataArray),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Batch prediction failed');
  }
  
  return await response.json();
};

export const checkHealth = async () => {
  const response = await fetch(`${API_URL}/health`);
  return await response.json();
};
```

### Step 2: Use in Your Component

```jsx
import { useState } from 'react';
import { predictPurchase } from './services/api';

function MyComponent() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    const data = {
      age: 35,
      gender: 'Male',
      city: 'New York',
      session_duration: 20.5,
      pages_viewed: 8,
      total_clicks: 15,
      cart_value: 250.50,
      discount_applied: 1,
      payment_method: 'Credit Card',
      product_category: 'Electronics',
      device_type: 'Desktop',
      month: 'Jan'
    };
    
    try {
      const result = await predictPurchase(data);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <button onClick={handlePredict} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict Purchase'}
      </button>
      
      {error && <p>Error: {error}</p>}
      
      {prediction && (
        <div>
          <p>Prediction: {prediction.class}</p>
          {prediction.probability && (
            <p>Confidence: {(prediction.probability * 100).toFixed(2)}%</p>
          )}
        </div>
      )}
    </div>
  );
}
```

### Step 3: Set Environment Variable

Create `.env` file in your React project root:

```env
REACT_APP_API_URL=http://localhost:5000
```

For production, update to your deployed API URL:

```env
REACT_APP_API_URL=https://your-api-url.herokuapp.com
```

---

## 🧪 Testing the API

### Using Python Test Script

Run the included test script:

```bash
# Start the API server first (in one terminal)
python app.py

# Then run tests (in another terminal)
python test_api.py
```

### Using curl

```bash
curl -X POST http://localhost:5000/predict \
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
```

### Using Postman

1. Create a new POST request
2. URL: `http://localhost:5000/predict`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON): Use the example JSON from the API endpoints section

---

## 🔒 CORS Configuration

The API already includes CORS support for all origins. If you need to restrict origins (for production):

In `app.py`, modify:

```python
CORS(app, origins=['http://localhost:3000', 'https://yourdomain.com'])
```

---

## 📝 API Response Format

### Single Prediction Response

```json
{
  "prediction": 1,
  "probability": 0.85,
  "class": "Purchase"
}
```

- `prediction`: 0 or 1 (No Purchase / Purchase)
- `probability`: Confidence score (0.0 to 1.0)
- `class`: Human-readable class name

### Batch Prediction Response

```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.85,
      "class": "Purchase"
    },
    {
      "prediction": 0,
      "probability": 0.25,
      "class": "No Purchase"
    }
  ],
  "count": 2
}
```

### Error Response

```json
{
  "error": "Error message here"
}
```

---

## 🐛 Troubleshooting

### CORS Errors
- Ensure `flask-cors` is installed: `pip install flask-cors`
- Check that CORS is enabled in `app.py` (it should be by default)

### Model Not Found
- Ensure all `.pkl` files are in the same directory as `app.py`
- Run `python train_model.py` first to generate model files

### Port Already in Use
- Change port in `app.py`: `app.run(port=5001)`
- Or kill the process using port 5000

### API Not Responding
- Check if Flask server is running
- Verify model files exist
- Check server logs for errors
- Test with `python test_api.py`

### Connection Refused (from React)
- Ensure API server is running
- Check API URL in `.env` file
- Verify CORS is enabled
- Check firewall settings

---

## ✅ Deployment Checklist

- [ ] Model trained and `.pkl` files generated
- [ ] API dependencies installed (`pip install -r requirements.txt`)
- [ ] API tested locally (`python test_api.py`)
- [ ] API server starts successfully (`python app.py`)
- [ ] CORS configured (enabled by default)
- [ ] Environment variable set in React project (`REACT_APP_API_URL`)
- [ ] Production API URL configured (if deploying)
- [ ] HTTPS enabled (production)
- [ ] Error handling implemented in React components

---

## 📞 Quick Reference

**API Base URL**: `http://localhost:5000` (local) or your deployed URL

**Endpoints**:
- `POST /predict` - Single or batch prediction
- `GET /health` - Health check
- `GET /features` - Get required features
- `GET /model/info` - Model information

**Required Features**:
- age (number)
- gender (string: Male/Female/Other)
- city (string)
- session_duration (number)
- pages_viewed (number)
- total_clicks (number)
- cart_value (number)
- discount_applied (0 or 1)
- payment_method (string)
- product_category (string)
- device_type (string: Mobile/Desktop/Tablet)
- month (string)

---

Happy Deploying! 🚀
