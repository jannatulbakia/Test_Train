"""
Prediction script using the trained model
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, label_encoders, feature_columns
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_model.py first to train the model.")
        return None, None, None, None

def preprocess_data(df, scaler, label_encoders, feature_columns):
    """Preprocess new data using saved transformers"""
    # Make a copy
    X = df.copy()
    
    # Handle missing values
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in numerical_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    for col in categorical_cols:
        if col in X.columns:
            if col in label_encoders:
                # Use saved encoder
                le = label_encoders[col]
                unique_values = set(X[col].astype(str).unique())
                known_values = set(le.classes_)
                
                # Handle unknown categories
                for val in unique_values:
                    if val not in known_values:
                        X.loc[X[col] == val, col] = le.classes_[0]  # Replace with first known class
                
                X[col] = le.transform(X[col].astype(str))
            else:
                X[col] = X[col].fillna('Unknown')
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    
    # Select only feature columns and scale
    X = X[feature_columns]
    X_scaled = scaler.transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_columns)
    
    return X

def predict(df, return_probabilities=False):
    """
    Make predictions on new data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with the same features as training data
    return_probabilities : bool
        If True, return prediction probabilities
    
    Returns:
    --------
    predictions : numpy.ndarray
        Predicted classes (0 or 1)
    probabilities : numpy.ndarray (optional)
        Prediction probabilities if return_probabilities=True
    """
    model, scaler, label_encoders, feature_columns = load_model_and_preprocessors()
    
    if model is None:
        return None
    
    # Preprocess data
    X = preprocess_data(df, scaler, label_encoders, feature_columns)
    
    # Make predictions
    predictions = model.predict(X)
    
    if return_probabilities and hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
        return predictions, probabilities
    
    return predictions

if __name__ == "__main__":
    print("=" * 80)
    print("ONLINE SHOPPING MODEL - PREDICTION SCRIPT")
    print("=" * 80)
    
    # Example: Load data and make predictions
    try:
        # Try to load the dataset
        df_test = pd.read_csv('online_shopping_dataset.csv')
        
        # Remove target column if exists
        target_cols = ['purchased', 'Purchase']
        for col in target_cols:
            if col in df_test.columns:
                df_test = df_test.drop(columns=[col])
        
        # Make predictions on first 10 samples
        sample_data = df_test.head(10)
        predictions, probabilities = predict(sample_data, return_probabilities=True)
        
        print("\n📊 Sample Predictions:")
        print("-" * 80)
        results_df = sample_data.copy()
        results_df['Predicted_Purchase'] = predictions
        results_df['Purchase_Probability'] = probabilities
        
        print(results_df[['Predicted_Purchase', 'Purchase_Probability']].to_string())
        
        print(f"\n✅ Predictions completed!")
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Predicted purchases: {sum(predictions)}")
        
    except FileNotFoundError:
        print("\n⚠ Please ensure 'online_shopping_dataset.csv' exists or provide your own data.")
        print("\nExample usage:")
        print("  predictions = predict(your_dataframe)")
        print("  predictions, probs = predict(your_dataframe, return_probabilities=True)")


