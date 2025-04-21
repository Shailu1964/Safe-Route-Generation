import joblib
import pandas as pd

# Load the trained model
model = joblib.load('crime_severity_model.pkl')

# Define the feature columns used during training
FEATURE_COLS = ['Date', 'Time', 'latitude', 'longitude', 'Day']

# Define the severity mapping
SEVERITY_MAPPING = {"Very Low": 0.1, "Low": 0.2, "Moderate": 0.5, "High": 0.7, "Very High": 1.0}

# Function to predict crime severity
def predict_crime_severity(lat, lon, date, time, day):
    input_data = {
        'Date': [date],
        'Time': [time],
        'latitude': [lat],
        'longitude': [lon],
        'Day': [day]
    }
    input_df = pd.DataFrame(input_data)[FEATURE_COLS]  # Ensure correct column order
    severity_category = model.predict(input_df)[0]  # Get the categorical prediction
    severity = SEVERITY_MAPPING.get(severity_category, 0.1)  # Map to numeric value, default to 0.1
    return severity
