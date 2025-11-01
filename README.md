# Online Shopping ML Model Training Pipeline

A comprehensive step-by-step machine learning pipeline for training predictive models on online shopping datasets.

## 🎯 Features

- **Automated Data Loading**: Handles CSV files or generates synthetic data for demonstration
- **Complete Preprocessing**: Missing value handling, categorical encoding, feature scaling
- **Multiple Models**: Trains Logistic Regression, Random Forest, and Gradient Boosting
- **Model Evaluation**: Accuracy, cross-validation, confusion matrix, ROC curve
- **Feature Importance**: Visualizes important features for tree-based models
- **Model Persistence**: Saves trained models and preprocessing objects for future use

## 📋 Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Step 1: Prepare Your Dataset

Place your dataset as `online_shopping_dataset.csv` in the project directory. The script will automatically:
- Detect if the file exists
- Generate synthetic data if the file is not found (for demonstration)

**Expected Dataset Format:**
- CSV file with features (numerical and categorical)
- Target variable named `purchased` or `Purchase` (binary: 0/1)
- Common e-commerce features: age, gender, session duration, cart value, etc.

### Step 2: Train the Model

Run the training script:

```bash
python train_model.py
```

### Step 3: Review Results

The script will:
1. Load and explore the data
2. Preprocess features
3. Train multiple models
4. Evaluate and select the best model
5. Save the model and generate visualizations

**Output Files:**
- `best_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoders.pkl` - Categorical encoders
- `feature_columns.pkl` - Feature column names
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve (if applicable)
- `feature_importance.png` - Feature importance plot (for tree models)

### Step 4: Make Predictions

Use the prediction script:

```python
from predict import predict
import pandas as pd

# Load your new data
new_data = pd.read_csv('new_customers.csv')

# Make predictions
predictions, probabilities = predict(new_data, return_probabilities=True)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

Or run the prediction script directly:

```bash
python predict.py
```

## 📊 Training Pipeline Steps

### Step 1: Data Loading
- Loads dataset from CSV or generates synthetic data
- Displays dataset shape and basic information

### Step 2: Data Exploration
- Shows data overview, info, and statistical summary
- Identifies missing values
- Analyzes target variable distribution

### Step 3: Data Preprocessing
- Handles missing values (median for numerical, mode for categorical)
- Encodes categorical variables using Label Encoding
- Scales features using StandardScaler
- Splits data into train/test sets (80/20)

### Step 4: Model Training
Trains three models:
- **Logistic Regression**: Linear model for binary classification
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Boosting ensemble method

Each model is evaluated using 5-fold cross-validation.

### Step 5: Model Evaluation
- Compares all models and selects the best one
- Generates classification report
- Creates confusion matrix
- Plots ROC curve (for binary classification)
- Visualizes feature importance (for tree models)

### Step 6: Model Saving
- Saves the best model and all preprocessing objects
- Enables easy model deployment and reuse

## 📈 Expected Results

The model predicts purchase behavior (0 = No Purchase, 1 = Purchase) based on:
- Customer demographics (age, gender, city)
- Session behavior (duration, pages viewed, clicks)
- Shopping cart information (cart value, discounts)
- Transaction details (payment method, product category, device type)
- Temporal features (month)

## 🔧 Customization

### Using Your Own Dataset

1. **Rename target column**: If your target column has a different name, the script will auto-detect binary columns or you can modify line 77 in `train_model.py`.

2. **Adjust features**: The script automatically excludes common non-feature columns like `user_id`, `User ID`, and `Unnamed: 0`.

3. **Change models**: Modify the `models` dictionary in Step 4 to use different algorithms.

### Hyperparameter Tuning

To add hyperparameter tuning, uncomment and modify this section in `train_model.py`:

```python
# Example for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## 📝 Notes

- The script automatically handles missing values and categorical encoding
- All preprocessing steps are saved for consistent prediction preprocessing
- Synthetic data is generated if no dataset file is found (for testing)
- Random seed is set to 42 for reproducibility

## 🐛 Troubleshooting

**Issue**: `FileNotFoundError` when loading dataset
- **Solution**: Ensure your CSV file is named `online_shopping_dataset.csv` or modify the filename in the script

**Issue**: Missing dependencies
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Model accuracy is low
- **Solution**: 
  - Check data quality and feature engineering
  - Try hyperparameter tuning
  - Consider feature selection
  - Verify target variable distribution (imbalanced data may need resampling)

## 📄 License

This is a demonstration project for educational purposes.

## 🌐 API Deployment

After training the model, you can deploy it as a REST API for use with your React.js application or any other frontend.

### Quick Start

1. **Start the API server**:
   ```bash
   python app.py
   ```
   The API will run on `http://localhost:5000`

2. **Test the API**:
   ```bash
   python test_api.py
   ```

3. **Use from React.js**: See `DEPLOYMENT.md` for detailed integration instructions.

For detailed deployment instructions (Heroku, Railway, Docker, etc.), see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 🤝 Contributing

Feel free to customize and extend this pipeline for your specific use case!


