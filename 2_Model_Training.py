import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

# --- Load the Processed Data ---
try:
    processed_data = joblib.load('processed_data.joblib')
    X_train_processed = processed_data['X_train_processed']
    X_test_processed = processed_data['X_test_processed']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    print("Successfully loaded processed data.")
except FileNotFoundError:
    print("Error: 'processed_data.joblib' not found.")
    print("Please run the '1_Data_Preprocessing.py' script first.")
    exit() # Exit the script if the data isn't available

print(f"Training data shape: {X_train_processed.shape}")
print(f"Test data shape: {X_test_processed.shape}")


# --- Model Training ---

# Initialize the LightGBM Regressor model
# LGBM is often a good choice for tabular data like this due to its performance and speed.
# We use some baseline parameters here. These can be tuned for better performance.
lgbm = lgb.LGBMRegressor(random_state=42)

print("\nTraining the LightGBM model...")

# Train the model on the processed training data
lgbm.fit(X_train_processed, y_train)

print("Model training complete.")


# --- Model Evaluation ---

print("\nEvaluating the model...")

# Make predictions on the test set
y_pred = lgbm.predict(X_test_processed)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance on Test Set:")
print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"  R-squared (R²): {r2:.4f}")


# --- Save the Trained Model ---

# Save the trained model to a file so we can use it in our Streamlit app
joblib.dump(lgbm, 'vehicle_price_model.joblib')

print(f"\nModel saved successfully as 'vehicle_price_model.joblib'")

# --- Optional: Feature Importance ---
# This helps understand which features the model found most predictive.
try:
    # Load the preprocessor to get feature names
    preprocessor = processed_data['preprocessor']
    model_columns = joblib.load('model_columns.joblib')

    # Get feature names from the preprocessor
    numerical_features = model_columns['numerical_features']
    categorical_features = model_columns['categorical_features']
    
    # Access the fitted OneHotEncoder to get the generated categories
    ohe_categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    # Combine all feature names in the correct order
    all_feature_names = numerical_features + list(ohe_categories)

    feature_importances = pd.DataFrame({
        'feature': all_feature_names,
        'importance': lgbm.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importances.head(15))

except Exception as e:
    print(f"\nCould not display feature importances due to an error: {e}")

