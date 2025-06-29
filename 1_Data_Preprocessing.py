import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib # For saving the preprocessor

# Load the dataset
try:
    df = pd.read_csv('dataset.csv')
except Exception as e:
    print(f"Error loading dataset.csv: {e}")
    # Create a dummy dataframe for demonstration if file not found
    data = {'name': ['Honda Civic', 'Toyota Camry', 'Ford F-150'],
            'year': [2018, 2020, 2019],
            'price': [15000, 25000, 35000],
            'mileage': [50000, 30000, 40000],
            'fuel': ['Gasoline', 'Gasoline', 'Gasoline'],
            'transmission': ['Automatic', 'Automatic', 'Automatic'],
            'make': ['Honda', 'Toyota', 'Ford'],
            'model': ['Civic', 'Camry', 'F-150'],
            'body': ['Sedan', 'Sedan', 'Pickup Truck'],
            'cylinders': [4, 4, 6],
            'drivetrain': ['Front-wheel Drive', 'Front-wheel Drive', 'Four-wheel Drive'],
            'description': ['','',''], 'trim': ['LX','LE','XLT'], 'doors': [4,4,4],
            'exterior_color': ['Black','White','Red'], 'interior_color': ['Black','Beige','Gray'],
            'engine':['4-Cyl','4-Cyl','V6']}
    df = pd.DataFrame(data)
    print("Loaded a dummy dataset for demonstration purposes.")


print("Initial Dataset Shape:", df.shape)
print("Initial Columns:", df.columns.tolist())

# --- Feature Engineering and Cleaning ---

# 1. Extract 'make' and 'model' from 'name' if they are missing
df['make'] = df['make'].fillna(df['name'].apply(lambda x: str(x).split(' ')[0]))

# 2. Clean 'mileage'
if df['mileage'].dtype == 'object':
    df['mileage'] = df['mileage'].str.replace('[^\d.]', '', regex=True)
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
# We will let the imputer handle NaNs now

# 3. Clean 'price' (our target variable)
if df['price'].dtype == 'object':
    df['price'] = df['price'].str.replace('[^\d.]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
# Drop rows where price is missing as it's our target
df.dropna(subset=['price'], inplace=True)


# 4. Clean 'cylinders' - Extract number from string
def clean_cylinders(value):
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        if match:
            return int(match.group(0))
    if pd.api.types.is_numeric_dtype(value):
        return int(value)
    return np.nan

df['cylinders'] = df['cylinders'].apply(clean_cylinders)
# We will let the imputer handle NaNs now

# 5. We will let the imputer handle 'doors' as well.

# --- Define Features and Target ---
features = ['year', 'mileage', 'make', 'model', 'body', 'cylinders', 'fuel', 'transmission', 'drivetrain', 'doors']
target = 'price'

existing_features = [f for f in features if f in df.columns]
print(f"Using features: {existing_features}")

df_model = df[existing_features + [target]].copy()

# ==================================================================
# THE FIX: REMOVE THE LINE BELOW
# This was causing the error because it was deleting all your data.
# The preprocessor pipeline will handle missing values correctly.
# df_model.dropna(inplace=True) 
# ==================================================================

print("Shape before splitting (after removing price NaNs):", df_model.shape)


# --- Preprocessing Pipelines ---

categorical_features = ['make', 'model', 'body', 'fuel', 'transmission', 'drivetrain']
numerical_features = ['year', 'mileage', 'cylinders', 'doors']

categorical_features = [f for f in categorical_features if f in df_model.columns]
numerical_features = [f for f in numerical_features if f in df_model.columns]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- Split Data and Preprocess ---
X = df_model[existing_features]
y = df_model[target]

# This check prevents the error from happening if the dataframe is empty for other reasons
if X.empty:
    print("\nERROR: The dataframe is empty before splitting. Cannot proceed.")
    print("Please check the initial dataset and cleaning steps.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on the training data and transform both training and testing data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"X_train shape after processing: {X_train_processed.shape}")
    print(f"X_test shape after processing: {X_test_processed.shape}")

    # --- Save the Preprocessor and the Cleaned Data ---
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("Preprocessor saved to 'preprocessor.joblib'")

    processed_data = {
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor
    }
    joblib.dump(processed_data, 'processed_data.joblib')
    print("Processed data (train/test splits) saved to 'processed_data.joblib'")
    
    # We also need to get unique values from the dataframe *before* imputation for the dropdowns
    model_columns = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'feature_order': existing_features,
        'unique_values': {col: df_model[col].dropna().unique().tolist() for col in categorical_features}
    }
    joblib.dump(model_columns, 'model_columns.joblib')
    print("Model columns and unique values saved to 'model_columns.joblib'")

    print("\nData preprocessing complete!")