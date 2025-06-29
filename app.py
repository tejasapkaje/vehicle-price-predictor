import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# --- Load Saved Artifacts ---
# Use a try-except block to handle missing files gracefully.
try:
    model = joblib.load('vehicle_price_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("Successfully loaded model and preprocessor.")
except FileNotFoundError:
    st.error("Error: Model or preprocessor files not found.")
    st.info("Please ensure 'vehicle_price_model.joblib', 'preprocessor.joblib', and 'model_columns.joblib' are in the same directory.")
    # Stop the app from running further if files are missing.
    st.stop()


# Extract the unique values for dropdowns from the loaded 'model_columns'
unique_values = model_columns.get('unique_values', {})
feature_order = model_columns.get('feature_order', [])

# --- Page Configuration ---
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# --- UI Layout and Styling ---
st.title("🚗 Vehicle Price Prediction")
st.markdown("Enter the vehicle's details below to get an estimated market price.")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

# --- User Input Section (using the sidebar for a clean look) ---
st.sidebar.header("Enter Vehicle Features")

# Function to create input fields
def user_input_features():
    # Using current year for the slider's max value
    current_year = datetime.now().year
    year = st.sidebar.slider("Year", 1990, current_year, current_year - 5)
    
    mileage = st.sidebar.number_input("Mileage (in miles)", min_value=0, max_value=500000, value=50000, step=1000)
    
    # Dropdowns for categorical features, using sorted unique values
    make = st.sidebar.selectbox("Make", sorted(unique_values.get('make', ['N/A'])))
    model_options = sorted(unique_values.get('model', ['N/A']))
    model_val = st.sidebar.selectbox("Model", model_options)
    
    body = st.sidebar.selectbox("Body Style", sorted(unique_values.get('body', ['N/A'])))
    
    cylinders = st.sidebar.selectbox("Cylinders", sorted(unique_values.get('cylinders', [4, 6, 8])))

    fuel = st.sidebar.selectbox("Fuel Type", sorted(unique_values.get('fuel', ['Gasoline'])))
    
    transmission = st.sidebar.selectbox("Transmission", sorted(unique_values.get('transmission', ['Automatic'])))

    drivetrain = st.sidebar.selectbox("Drivetrain", sorted(unique_values.get('drivetrain', ['Front-wheel Drive'])))
    
    doors = st.sidebar.selectbox("Doors", sorted(unique_values.get('doors', [2,4])))


    data = {
        'year': year,
        'mileage': mileage,
        'make': make,
        'model': model_val,
        'body': body,
        'cylinders': cylinders,
        'fuel': fuel,
        'transmission': transmission,
        'drivetrain': drivetrain,
        'doors': doors
    }
    
    # Create a DataFrame from the user input
    # Important: The column order must match the 'feature_order' list
    features = pd.DataFrame(data, index=[0])
    return features[feature_order] # Enforce column order

# Get user input
input_df = user_input_features()


# --- Display User Input ---
with col1:
    st.subheader("Your Selections")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))


# --- Prediction Logic ---
# Create a button to trigger the prediction
if st.sidebar.button("Predict Price", use_container_width=True):
    try:
        # The input_df is already in the correct format and order
        
        # Transform the input data using the loaded preprocessor
        input_processed = preprocessor.transform(input_df)
        
        # Make a prediction
        prediction = model.predict(input_processed)
        
        # Display the prediction
        with col2:
            st.subheader("Predicted Price")
            # Format the prediction as currency
            price_str = f"${prediction[0]:,.2f}"
            st.markdown(f"""
            <div style="background-color:#DFF2BF; padding:20px; border-radius:10px;">
                <h2 style="color:#4F8A10; text-align:center;">Estimated Price:</h2>
                <h1 style="color:#4F8A10; text-align:center;">{price_str}</h1>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
     with col2:
        st.info("Click the 'Predict Price' button in the sidebar to see the result.")

# Add a footer
st.markdown("---")
st.markdown("Developed for the Vehicle Price Prediction Project.")

