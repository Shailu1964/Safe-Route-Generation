import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from geopy.distance import geodesic
from datetime import datetime

# Load dataset
df = pd.read_csv("FINAL_CRIME_DATASET.csv") 

# Check the format of 'Date of Occurrence' and 'Time of Occurrence'
# print("Date format sample:", df['Date of Occurrence'].head())
# print("Time format sample:", df['Time of Occurrence'].head())

# Convert DateTime with proper error handling
try:
    # First convert Date of Occurrence to datetime if it isn't already
    df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'], format='%d/%m/%Y')
    
    # Combine date and time using proper string formatting
    df['DateTime'] = pd.to_datetime(
        df['Date of Occurrence'].dt.strftime('%Y-%m-%d') + ' ' + 
        df['Time of Occurrence'].astype(str)
    )
except Exception as e:
    print(f"Error in datetime conversion: {e}")
    # Alternative method using separate parsing
    try:
        df['DateTime'] = pd.to_datetime(
            pd.to_datetime(df['Date of Occurrence']).dt.strftime('%Y-%m-%d') + ' ' + 
            pd.to_datetime(df['Time of Occurrence']).dt.strftime('%H:%M:%S')
        )
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")
        # Last resort: parse components separately
        df['DateTime'] = pd.to_datetime(df['Date of Occurrence'])
        df['DateTime'] += pd.to_timedelta(df['Time of Occurrence'].astype(str))

# Verify the conversion worked
print("\nDateTime conversion result:")
print(df['DateTime'].head())

# Extract time features
df["hour"] = df['DateTime'].dt.hour
df["day_of_week"] = df['DateTime'].dt.dayofweek  # Monday=0, Sunday=6
df["is_night"] = df["hour"].apply(lambda x: 1 if x >= 18 or x < 6 else 0)
df["month"] = df['DateTime'].dt.month
df["year"] = df['DateTime'].dt.year

# Create time of day categories
def time_category(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['TimeOfDay'] = df['hour'].apply(time_category)

# Process Crime Types (since they're comma-separated in your sample)
def process_crime_types(df):
    # Get all unique crime types
    all_crimes = set()
    for crimes in df['Crime Type']:
        crime_list = [c.strip() for c in str(crimes).split(',')]
        all_crimes.update(crime_list)
    
    # Create binary columns for each crime type
    for crime in all_crimes:
        df[f'Crime_{crime}'] = df['Crime Type'].apply(lambda x: 1 if crime in str(x) else 0)
    
    return df

df = process_crime_types(df)

# Encoding categorical variables
categorical_cols = ["Station", "TimeOfDay", "Day of Occurrence"]
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Distance from police station is already in your dataset as 'Distance from Police Station'
# No need to calculate it again

# Define comprehensive crime weights dictionary
crime_weights = {
    # Violent Crimes (Highest severity)
    "Murder": 10,
    "Culpable Homicide": 9,
    "Rape": 10,
    "Sexual Assault": 9,
    "Kidnapping": 8,
    "Abduction": 8,
    "Trafficking": 9,
    "Grievous Hurt": 7,
    "Acid Attack": 9,
    "Dowry Death": 9,
    "Honor Killing": 10,
    "Gang Rape": 10,
    "Attempt to Murder": 8,
    
    # Property Crimes
    "Theft": 3,
    "Robbery": 6,
    "Dacoity": 7,
    "Extortion": 6,
    "Burglary": 5,
    "House-breaking": 5,
    "Criminal Breach of Trust": 4,
    "Cheating": 4,
    "Forgery": 4,
    "Counterfeiting": 5,
    "Mischief": 2,
    "Arson": 7,
    "Criminal Trespass": 3,
    
    # Public Safety Crimes
    "Rioting": 6,
    "Unlawful Assembly": 4,
    "Affray": 3,
    "Public Nuisance": 2,
    "Rash Driving": 5,
    "Negligent Driving": 4,
    "Drunk Driving": 6,
    "Food Adulteration": 5,
    "Drug Trafficking": 7,
    "Drug Possession": 4,
    
    # Crimes Against Public Authority
    "Sedition": 8,
    "Contempt of Court": 3,
    "Perjury": 4,
    "False Evidence": 4,
    "Bribery": 5,
    "Corruption": 6,
    "Obstruction of Justice": 5,
    
    # Cyber Crimes
    "Hacking": 6,
    "Data Theft": 6,
    "Online Fraud": 5,
    "Identity Theft": 6,
    "Cyberstalking": 5,
    "Child Pornography": 9,
    "Cyber Terrorism": 9,
    
    # Other Special Laws
    "Arms Act Violation": 6,
    "NDPS Act Violation": 7,
    "SC/ST Atrocities": 7,
    "Domestic Violence": 6,
    "Illegal Gambling": 2,
    "Illegal Liquor": 3,
    "Animal Cruelty": 3,
    "Environmental Violation": 4,
    
    # IPC Category Mappings
    "Offences against the State": 8,
    "Public Tranquillity": 5,
    "False Evidence & Justice": 4,
    "Public Health, Safety": 5,
    "Religious Offences": 4,
    "Offences Affecting Life": 9,
    "Sexual Offences": 9,
    "Hurt": 6,
    "Wrongful Restraint and Confinement": 5,
    "Criminal Force and Assault": 6,
    "Criminal Misappropriation of Property": 4,
    "Criminal Trespass": 3,
    "Offences Relating to Documents": 4,
    "Marriage-related Offences": 4,
    "Cruelty & Defamation": 5,
    "Intimidation & Annoyance": 4,
    
    # Generic fallbacks for IPC sections
    "IPC 302": 10,  # Murder
    "IPC 376": 10,  # Rape
    "IPC 307": 8,   # Attempt to murder
    "IPC 392": 6,   # Robbery
    "IPC 420": 4,   # Cheating
    "IPC 354": 7,   # Assault on woman
    "IPC 506": 4,   # Criminal intimidation
    "IPC 376D": 10, # Gang rape
    "IPC 304B": 9,  # Dowry death
    "IPC 498A": 6,  # Cruelty by husband or relatives
    
    # Non-IPC Acts
    "Arms Act": 6,
    "Motor Vehicles Act": 3,
    "Environmental Protection Act": 4,
    "Narcotic Drugs Act": 7,
    "Psychotropic Substances Act": 7,
    "Information Technology Act": 5,
    "Copyright Act": 3,
    "Bharatiya Nyaya Sanhita": 5,
    "Bharatiya Nagarik Suraksha Sanhita": 4,
    "Food Safety and Standards Act": 4,
    "Maharashtra Police Act": 3,
    "Maharashtra Prohibition Act": 3,
    "Maharashtra Gambling Prohibition Act": 2,
    "Maharashtra Motor Vehicle Rules": 3,
    "Prevention of Damage to Public Property Act": 4,
    "Juvenile Justice Act": 4,
    "Prevention of Cruelty to Animals Act": 3,
    "SC/ST Prevention of Atrocities Act": 7,
    "Rights of Persons with Disabilities Act": 4,
    "COTPA": 2,
    "NDPS": 7,
    "CrPC": 3,
    "BNS": 5,
    "BNSS": 4
}

# Function to get crime severity from IPC section number
def get_severity_from_ipc_section(section_number):
    # Convert to integer if it's a numeric section
    try:
        section = int(section_number.replace("IPC", "").strip())
        
        # Map IPC sections to categories and their severities
        if 299 <= section <= 311:  # Murder, culpable homicide
            return crime_weights.get("Offences Affecting Life", 9)
        elif 375 <= section <= 377:  # Sexual offenses
            return crime_weights.get("Sexual Offences", 9)
        elif 378 <= section <= 382:  # Theft
            return crime_weights.get("Theft", 3)
        elif 390 <= section <= 402:  # Robbery and dacoity
            return crime_weights.get("Robbery", 6)
        elif 425 <= section <= 440:  # Mischief
            return crime_weights.get("Mischief", 2)
        elif 319 <= section <= 338:  # Hurt
            return crime_weights.get("Hurt", 6)
        # Add more mappings as needed
        
        # Default severity for unmapped IPC sections
        return 5
    except:
        # If we can't parse the section number, return default severity
        return 5

# Function to estimate severity for any crime type string
def estimate_crime_severity(crime_string):
    # Clean the input string
    crime_string = crime_string.strip().lower()
    
    # Direct match in our dictionary (case-insensitive)
    for crime, weight in crime_weights.items():
        if crime.lower() in crime_string:
            return weight
    
    # Check for IPC section numbers
    if "ipc" in crime_string or "section" in crime_string:
        # Try to extract section numbers
        import re
        section_matches = re.findall(r'ipc\s*(\d+)', crime_string)
        if section_matches:
            return get_severity_from_ipc_section("IPC " + section_matches[0])
    
    # Keywords-based severity estimation
    keywords = {
        10: ["murder", "rape", "gang", "terrorism", "trafficking"], 
        9: ["homicide", "sexual assault", "abduction", "grievous"],
        8: ["attempt to murder", "kidnap", "grievous hurt", "sedition"],
        7: ["robbery", "dacoity", "drug", "arms", "weapon", "hurt", "assault"],
        6: ["extortion", "riot", "breach", "battery"],
        5: ["theft", "cheat", "fraud", "forgery"],
        4: ["trespass", "mischief", "defamation", "nuisance"],
        3: ["petty theft", "minor", "gambling", "liquor"],
        2: ["traffic", "minor offense", "public disturbance"]
    }
    
    # Check for keywords from highest to lowest severity
    for severity, words in sorted(keywords.items(), reverse=True):
        for word in words:
            if word in crime_string:
                return severity
    
    # Default severity if nothing matches
    return 3

# Define time of day weights
time_weights = {
    'Morning': 1,
    'Afternoon': 1.5,
    'Evening': 2,
    'Night': 3
}

#
def map_severity_to_words(severity):
    """
    Convert numeric severity to descriptive words.
    
    Args:
        severity (int/float): Numeric severity score (1-5)
    
    Returns:
        str: Descriptive severity level
    """
    if severity <= 1:
        return "Very Low"
    elif 1 < severity <= 2:
        return "Low"
    elif 2 < severity <= 3:
        return "Moderate"
    elif 3 < severity <= 4:
        return "High"
    else:
        return "Very High"

# Function to calculate severity score if not already in dataset
if 'CrimeSeverity' not in df.columns:
    # Function to calculate severity score
    def calculate_severity(row):
        try:
            # Base score from crime type
            crime_score = 0
            if pd.notna(row['Crime Type']):  # Check if crime type is not NaN
                for crime in str(row['Crime Type']).split(','):
                    crime = crime.strip()
                    if crime in crime_weights:
                        crime_score = max(crime_score, crime_weights[crime])
                    else:
                        # Use our advanced estimation function
                        crime_score = max(crime_score, estimate_crime_severity(crime))
            
            # Default crime score if no valid crimes found
            if crime_score == 0:
                crime_score = 3  # Default moderate severity
            
            # Adjust based on time of day (with null check)
            time_multiplier = time_weights.get(str(row['TimeOfDay']), 1.5)
            
            # Adjust based on distance from police (with null check)
            distance = row['Distance from Police Station']
            if pd.isna(distance):
                distance = 5  # Default moderate distance
            distance_factor = min(float(distance) / 10, 1) * 1.5 + 0.5
            
            # Combine factors
            severity = float(crime_score) * float(time_multiplier) * float(distance_factor)
            
            # Scale to 1-5 range with proper rounding
            severity = min(max(round(severity / 2), 1), 5)
            
            return map_severity_to_words(severity)
            
        except Exception as e:
            print(f"Error calculating severity: {e}")
            return 3  # Return default moderate severity on error

# Apply the severity calculation with error handling
if 'CrimeSeverity' not in df.columns:
    df['CrimeSeverity'] = df.apply(calculate_severity, axis=1)
    print("Crime severity calculation complete.")
    print(f"Severity distribution:\n{df['CrimeSeverity'].value_counts().sort_index()}")

    df['CrimeSeverity'] = df.apply(calculate_severity, axis=1)

# Normalizing features (create new columns for normalized values)
numeric_features = ['hour', 'day_of_week', 'Distance from Police Station']
scaler = StandardScaler()
normalized_features = scaler.fit_transform(df[numeric_features])

# Add normalized columns to the dataframe with a '_normalized' suffix
for i, feature in enumerate(numeric_features):
    df[f'{feature}_normalized'] = normalized_features[:, i]

# Prepare features for model
crime_type_cols = [col for col in df.columns if col.startswith('Crime_')]
feature_cols = [f'{feature}_normalized' for feature in numeric_features] + crime_type_cols + ['is_night', 'Station', 'TimeOfDay', 'Latitude', 'Longitude']
feature_cols = [col for col in feature_cols if col in df.columns]

# Save preprocessed data and model
df.to_csv("preprocessed_Pune_crime_data.csv", index=False)

