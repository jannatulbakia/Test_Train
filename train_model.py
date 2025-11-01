"""
Online Shopping Dataset - ML Model Training Pipeline
Step-by-step machine learning model training for e-commerce data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("ONLINE SHOPPING ML MODEL TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n[STEP 1] Loading Dataset...")
print("-" * 80)

def load_data(file_path='online_shopping_dataset.csv'):
    """
    Load the online shopping dataset from CSV file
    If file doesn't exist, generate synthetic data for demonstration
    """
    if os.path.exists(file_path):
        print(f"✓ Loading data from {file_path}")
        df = pd.read_csv(file_path)
    else:
        print(f"⚠ File {file_path} not found. Generating synthetic dataset for demonstration...")
        np.random.seed(42)
        n_samples = 10000
        
        # Generate synthetic online shopping data
        df = pd.DataFrame({
            'user_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 65, n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
            'session_duration': np.random.normal(15, 8, n_samples).round(2),
            'pages_viewed': np.random.poisson(5, n_samples),
            'total_clicks': np.random.poisson(12, n_samples),
            'cart_value': np.random.normal(150, 75, n_samples).round(2),
            'discount_applied': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'COD'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_samples),
            'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples),
            'month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], n_samples),
            'purchased': np.random.choice([0, 1], n_samples, p=[0.45, 0.55])  # Target variable
        })
        
        # Create some realistic correlations
        df.loc[df['session_duration'] > 20, 'purchased'] = np.random.choice([0, 1], size=df.loc[df['session_duration'] > 20].shape[0], p=[0.3, 0.7])
        df.loc[df['cart_value'] > 200, 'purchased'] = np.random.choice([0, 1], size=df.loc[df['cart_value'] > 200].shape[0], p=[0.25, 0.75])
        
        # Save synthetic data
        df.to_csv('online_shopping_dataset.csv', index=False)
        print("✓ Synthetic dataset generated and saved as 'online_shopping_dataset.csv'")
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

df = load_data()

# ============================================================================
# STEP 2: DATA EXPLORATION
# ============================================================================
print("\n[STEP 2] Data Exploration...")
print("-" * 80)

print("\n📊 Dataset Overview:")
print(df.head())

print("\n📈 Dataset Info:")
print(df.info())

print("\n📉 Statistical Summary:")
print(df.describe())

print("\n🔍 Missing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("✓ No missing values found!")

# Check target variable distribution
if 'purchased' in df.columns:
    target_col = 'purchased'
elif 'Purchase' in df.columns:
    target_col = 'Purchase'
else:
    # Find binary column that might be target
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    target_col = binary_cols[0] if binary_cols else None

if target_col:
    print(f"\n🎯 Target Variable Distribution ({target_col}):")
    print(df[target_col].value_counts())
    print(f"\n  Class Distribution: {df[target_col].value_counts(normalize=True)}")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 3] Data Preprocessing...")
print("-" * 80)

# Identify feature columns (exclude user_id and target)
feature_cols = [col for col in df.columns if col not in [target_col, 'user_id', 'User ID', 'Unnamed: 0']]
print(f"✓ Identified {len(feature_cols)} feature columns")

# Separate features and target
X = df[feature_cols].copy()
y = df[target_col].copy() if target_col else None

# Handle missing values
print("\n🧹 Handling missing values...")
if X.isnull().sum().sum() > 0:
    # Fill numerical columns with median
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    
    # Fill categorical columns with mode
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    print("✓ Missing values handled")

# Encode categorical variables
print("\n🔤 Encoding categorical variables...")
label_encoders = {}
X_encoded = X.copy()

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded: {col} ({X[col].nunique()} unique values)")

X = X_encoded

# Feature scaling
print("\n📏 Scaling numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
print("✓ Features scaled using StandardScaler")

# Train-test split
print("\n✂️ Splitting data into train and test sets...")
if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Training set class distribution: {y_train.value_counts().to_dict()}")
else:
    print("⚠ No target variable found. Creating dummy target for demonstration.")
    y = np.random.choice([0, 1], size=len(X), p=[0.5, 0.5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================
print("\n[STEP 4] Model Training...")
print("-" * 80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🤖 Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    print(f"  ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
print("\n[STEP 5] Model Evaluation...")
print("-" * 80)

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"  Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"  CV Score: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std'] * 2:.4f})")

# Detailed evaluation of best model
print(f"\n📋 Detailed Classification Report for {best_model_name}:")
y_pred_best = results[best_model_name]['predictions']
print(classification_report(y_test, y_pred_best, target_names=['No Purchase', 'Purchase']))

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# ROC Curve (if probabilities available)
if results[best_model_name]['probabilities'] is not None:
    fpr, tpr, _ = roc_curve(y_test, results[best_model_name]['probabilities'])
    auc_score = roc_auc_score(y_test, results[best_model_name]['probabilities'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {best_model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC Curve saved as 'roc_curve.png' (AUC = {auc_score:.4f})")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print("\n🔍 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved as 'feature_importance.png'")

# ============================================================================
# STEP 6: MODEL SAVING
# ============================================================================
print("\n[STEP 6] Saving Model and Preprocessing Objects...")
print("-" * 80)

# Save best model
model_path = 'best_model.pkl'
joblib.dump(best_model, model_path)
print(f"✓ Best model saved as '{model_path}'")

# Save scaler
scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved as '{scaler_path}'")

# Save label encoders
encoders_path = 'label_encoders.pkl'
joblib.dump(label_encoders, encoders_path)
print(f"✓ Label encoders saved as '{encoders_path}'")

# Save feature columns
features_path = 'feature_columns.pkl'
joblib.dump(list(X.columns), features_path)
print(f"✓ Feature columns saved as '{features_path}'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE! 🎉")
print("=" * 80)
print(f"\n✅ Best Model: {best_model_name}")
print(f"✅ Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"✅ Cross-Validation Score: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std'] * 2:.4f})")
print(f"\n📁 Saved Files:")
print(f"  - {model_path}")
print(f"  - {scaler_path}")
print(f"  - {encoders_path}")
print(f"  - {features_path}")
print(f"  - confusion_matrix.png")
if results[best_model_name]['probabilities'] is not None:
    print(f"  - roc_curve.png")
if hasattr(best_model, 'feature_importances_'):
    print(f"  - feature_importance.png")
print("\n" + "=" * 80)

