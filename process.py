import pandas as pd

# Load dataset
df = pd.read_csv("pune_final_dataset.csv")

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extract the day of the week and create a new column
df['Day'] = df['Date'].dt.day_name()

# Save the updated dataset
df.to_csv("pune_final_dataset_with_days.csv", index=False)

print(df.head())  # Display first few rows
