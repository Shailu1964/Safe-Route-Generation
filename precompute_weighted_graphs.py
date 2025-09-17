import joblib
import copy
import os
from datetime import datetime
import pandas as pd
from route_calculations import adjust_weights_for_crime
from utils import predict_crime_severity

print("Starting pre-computation of all required data...")

# --- Load Base Data ---
print("Loading base graph and crime data...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
G = joblib.load(os.path.join(BASE_DIR, 'pune_graph.pkl'))
crime_data = joblib.load(os.path.join(BASE_DIR, 'preprocessed_Pune_crime_data.pkl'))
crime_data.columns = crime_data.columns.str.strip().str.lower() # Standardize columns
crime_spots = [(row['latitude'], row['longitude']) for index, row in crime_data.iterrows()]

# --- Predict Crime Severity for All Spots ---
print("Predicting current crime severity for all known spots...")
now = datetime.now()
date = now.timestamp()
time = now.hour
day = now.weekday()

dynamic_crime_spots_for_weighting = []
dynamic_crime_spots_for_heatmap = []
# NEW: A list to hold pre-calculated severities for our main dataframe
precomputed_severities = []

for index, row in crime_data.iterrows():
    lat, lon = row['latitude'], row['longitude']
    try:
        severity = predict_crime_severity(lat, lon, date, time, day)
        dynamic_crime_spots_for_weighting.append((lat, lon, severity))
        dynamic_crime_spots_for_heatmap.append([lat, lon, severity])
        precomputed_severities.append(severity)
    except Exception as e:
        print(f"Error predicting crime severity for spot ({lat}, {lon}): {e}")
        precomputed_severities.append(0.1) # Add a default low severity on error

# --- Add Pre-computed Severities to the DataFrame (NEW STEP) ---
print("Adding pre-computed severities to the crime dataframe...")
crime_data['precomputed_severity'] = precomputed_severities

# --- Save the Enriched Crime Data File ---
# We now overwrite the old file with this new, faster version
joblib.dump(crime_data, os.path.join(BASE_DIR, 'preprocessed_Pune_crime_data.pkl'))
print("'preprocessed_Pune_crime_data.pkl' has been updated with pre-computed severities.")

# --- Create and Save the Weighted Graphs ---
print("Creating the 'Safest' graph...")
safe_graph = adjust_weights_for_crime(copy.deepcopy(G), dynamic_crime_spots_for_weighting, radius=0.005)
joblib.dump(safe_graph, os.path.join(BASE_DIR, 'safe_graph.pkl'))

print("Creating the 'Optimized' graph...")
optimized_graph = adjust_weights_for_crime(copy.deepcopy(G), dynamic_crime_spots_for_weighting, radius=0.002)
joblib.dump(optimized_graph, os.path.join(BASE_DIR, 'optimized_graph.pkl'))

# --- Save the Pre-computed Heatmap Data ---
print("Saving pre-computed heatmap data...")
joblib.dump(dynamic_crime_spots_for_heatmap, os.path.join(BASE_DIR, 'heatmap_data.pkl'))

print("\nAll pre-computation complete!")