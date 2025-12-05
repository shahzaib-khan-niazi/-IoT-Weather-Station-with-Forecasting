import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- Part 1: Load and Prepare Data ---
input_filename = 'thingspeak_data_full.csv'
try:
    # Use the name of the file you have in the project directory
    df = pd.read_csv(input_filename, parse_dates=['created_at'])
    print(f"Loaded {len(df)} data points from {input_filename}")
except FileNotFoundError:
    print(f"Error: {input_filename} not found. Please ensure it's in the same directory.")
    exit()

# Filter out rows where essential sensor data is missing
# We need field1 (temp), field2 (humidity), field3 (pressure), field4 (water value)
original_rows = len(df)
df_cleaned = df.dropna(subset=['field1', 'field2', 'field3', 'field4']).copy()
cleaned_rows = len(df_cleaned)
print(f"Filtered out {original_rows - cleaned_rows} rows with missing sensor data.")
print(f"Remaining {cleaned_rows} rows after cleaning.")

if cleaned_rows == 0:
    print("WARNING: No complete data rows remaining after cleaning. Cannot train model.")
    exit()

# --- CRUCIAL CHANGE: AUTOMATICALLY GENERATE THE RAIN LABEL FOR TRAINING ---
# Since physical wetting is not possible, we will label the top N rows
# with the highest humidity as 'Rain' events for the purpose of training.
print("\nATTENTION: Automatically generating 'Rain' labels for training data.")
print("This simulates manual labeling based on the N highest humidity readings.")

# 1. Initialize the new target column to 0 (No Rain)
df_cleaned['is_rain_manual'] = 0

# 2. Sort by Humidity (field2) and take the top N rows
N_RAIN_SAMPLES = 10 # <--- You can adjust this number for more/fewer simulated rain events
rain_indices = df_cleaned.nlargest(N_RAIN_SAMPLES, 'field2').index

# 3. Set the 'is_rain_manual' column to 1 for those high-humidity rows
df_cleaned.loc[rain_indices, 'is_rain_manual'] = 1
print(f"Set {N_RAIN_SAMPLES} data points (highest humidity) as 'Rain events (1)' for training.")


# --- Set up Features and Target for Training ---
target = 'is_rain_manual'
# INCLUDING 'field4' (water sensor) as a feature
features = ['field1', 'field2', 'field3', 'field4']

X = df_cleaned[features]
y = df_cleaned[target]

print(f"\nFeatures selected for training: {features}")
print(f"Target selected: {target}")

# Check the distribution of our new label
rain_count = y.sum()
no_rain_count = len(y) - rain_count
print(f"\nRain_Actual distribution (simulated): Rain events={rain_count}, No Rain events={no_rain_count}")

if rain_count == 0 or no_rain_count == 0:
    print("FATAL ERROR: Failed to create Rain samples. Cannot train model.")
    print("This means either your dataset is too small, or N_RAIN_SAMPLES is 0.")
    exit()

# --- Part 2: Train the AI Model ---

# Use stratify=y to ensure both classes are represented in train and test sets
# This is important when one class is much smaller (like our simulated rain events)
if rain_count > 0 and no_rain_count > 0: # Check again before stratifying
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    # If only one class exists (which should be caught by the FATAL ERROR above), stratify would fail.
    # This branch is mostly a safeguard.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("WARNING: Stratification skipped due to only one class in target variable after rain labeling.")


print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

model = LogisticRegression(solver='liblinear', random_state=42, max_iter=200)

try:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0)) # zero_division to avoid warnings if a class has no predictions
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Part 3: Save the Trained Model ---
    model_filename = 'trained_rain_prediction_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\n✅ Trained model saved as '{model_filename}'.")

except ValueError as e:
    print(f"\n❌ ERROR during model training: {e}")
    print("This usually means only one class (Rain or No Rain) is present in your training data (X_train, y_train).")
    print("Ensure N_RAIN_SAMPLES is appropriate for your dataset size and produces enough 'rain' events.")
