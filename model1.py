import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the dataset and feature columns
df = pd.read_csv('Pune_final_crime_dataset.csv')  # Replace 'crime_data.csv' with your actual dataset path

# Preprocessing non-numeric columns
df['Date'] = pd.to_datetime(df['Date']).astype('int64') // 10**9  # Convert to Unix timestamp
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour  # Extract hour from time
df['Day'] = df['Day'].astype('category').cat.codes  # Encode days as numeric categories

# Define feature columns based on the dataset
feature_cols = ['Date', 'Time', 'latitude', 'longitude', 'Day']  # Adjusted to match dataset column names

X = df[feature_cols]
y = df['CrimeSeverity']  # Using our calculated severity

# Map CrimeSeverity to numerical values
severity_mapping = {
    'Very Low': 0.1,
    'Low': 0.2,
    'Moderate': 0.5,
    'High': 0.7,
    'Very High': 0.9
}
df['CrimeSeverity'] = df['CrimeSeverity'].map(severity_mapping)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Feature importance
feature_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': clf.feature_importances_})
feature_imp = feature_imp.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_imp.head(10))

import joblib
joblib.dump(clf, 'model1.pkl')

print("Data preprocessing and model training complete. Preprocessed dataset and model saved.")