"""
Flask API Server for ML Model Predictions
Deploy this to serve predictions to your React.js application
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for React.js frontend

# Global variables to store loaded models
model = None
scaler = None
label_encoders = None
feature_columns = None

def load_model_artifacts():
    """Load the trained model and preprocessing objects"""
    global model, scaler, label_encoders, feature_columns
    
    try:
        if not os.path.exists('best_model.pkl'):
            raise FileNotFoundError("Model file 'best_model.pkl' not found. Please train the model first.")
        
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        print("✓ Model and preprocessing objects loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocess input data using saved transformers"""
    global scaler, label_encoders, feature_columns
    
    # Create DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame([data])
    
    # Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            # Handle unknown categories
            unique_values = set(df[col].astype(str).unique())
            known_values = set(le.classes_)
            
            for val in unique_values:
                if val not in known_values:
                    df.loc[df[col].astype(str) == val, col] = le.classes_[0]
            
            df[col] = le.transform(df[col].astype(str))
        elif col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only feature columns and scale
    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_columns)
    
    return X

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API information"""
    return jsonify({
        'message': 'ML Model API - Online Shopping Purchase Predictor',
        'version': '1.0.0',
        'status': 'online',
        'model_loaded': model is not None,
        'endpoints': {
            'health': 'GET /health - Health check',
            'predict': 'POST /predict - Make predictions (single or batch)',
            'features': 'GET /features - Get required features',
            'model_info': 'GET /model/info - Get model information'
        },
        'usage': {
            'single_prediction': 'POST /predict with JSON body containing features',
            'batch_prediction': 'POST /predict with JSON array of feature objects'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint - accepts single or batch predictions"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files exist.'
            }), 500
        
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'error': 'No data provided. Please send JSON data in the request body.'
            }), 400
        
        # Check if it's a single prediction or batch
        if isinstance(data, dict):
            # Single prediction
            X = preprocess_input(data)
            prediction = model.predict(X)[0]
            probability = None
            
            if hasattr(model, 'predict_proba'):
                probability = float(model.predict_proba(X)[0][1])
            
            return jsonify({
                'prediction': int(prediction),
                'probability': probability,
                'class': 'Purchase' if prediction == 1 else 'No Purchase'
            }), 200
        
        elif isinstance(data, list):
            # Batch prediction
            if len(data) == 0:
                return jsonify({
                    'error': 'Empty list provided. Please send at least one record.'
                }), 400
            
            X = preprocess_input(data)
            predictions = model.predict(X).tolist()
            probabilities = None
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1].tolist()
            
            results = []
            for i, pred in enumerate(predictions):
                result = {
                    'prediction': int(pred),
                    'class': 'Purchase' if pred == 1 else 'No Purchase'
                }
                if probabilities is not None:
                    result['probability'] = float(probabilities[i])
                results.append(result)
            
            return jsonify({
                'predictions': results,
                'count': len(results)
            }), 200
        
        else:
            return jsonify({
                'error': 'Invalid data format. Expected object or array.'
            }), 400
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    return predict()

@app.route('/features', methods=['GET'])
def get_features():
    """Get list of required features"""
    if feature_columns is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'features': feature_columns,
        'count': len(feature_columns)
    }), 200

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    info = {
        'model_type': type(model).__name__,
        'features_count': len(feature_columns) if feature_columns else 0,
        'has_probabilities': hasattr(model, 'predict_proba')
    }
    
    return jsonify(info), 200

if __name__ == '__main__':
    print("=" * 80)
    print("ML MODEL API SERVER")
    print("=" * 80)
    
    # Load model on startup
    if load_model_artifacts():
        print("\n✓ Server starting...")
        print("✓ API endpoints available:")
        print("  - GET / - API information")
        print("  - POST /predict - Make predictions (single or batch)")
        print("  - GET /health - Health check")
        print("  - GET /features - Get required features")
        print("  - GET /model/info - Get model information")
        print("\n🚀 Starting Flask server on http://localhost:5000")
        print("=" * 80)
        
        # Run the app
        # Use PORT environment variable for Railway/Heroku deployment, default to 5000 for local
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("\n❌ Failed to load model. Please ensure you've trained the model first.")
        print("   Run: python train_model.py")

