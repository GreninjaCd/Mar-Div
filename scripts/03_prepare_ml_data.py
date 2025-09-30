import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Input file path for the cleaned data
CLEANED_DATA_PATH = os.path.join('..', 'data', 'processed', 'cleaned_obis_data.csv')
# Output file path for the machine learning-ready data
ML_DATA_DIR = os.path.join('..', 'data', 'processed')
ML_DATA_FILE = os.path.join(ML_DATA_DIR, 'ml_ready_data.csv')
# The number of pseudo-absence points to generate per presence point
PSEUDO_ABSENCE_RATIO = 1

def prepare_ml_data():
    """
    Prepares the cleaned data for machine learning by:
    1. Identifying the most common species to use as a target.
    2. Creating a balanced dataset with presence and pseudo-absence points.
    """
    print(f"Loading cleaned data from {CLEANED_DATA_PATH}...")

    if not os.path.exists(CLEANED_DATA_PATH):
        print("Error: Cleaned data file not found. Please run '02_process_data.py' first.")
        return

    df = pd.read_csv(CLEANED_DATA_PATH)

    # --- 1. Select Target Species ---
    # We'll choose the most frequently observed species as our target for this model.
    # A species with more data points is generally easier to model.
    target_species = df['species'].mode()[0]
    print(f"Selected target species for modeling: '{target_species}'")

    presence_df = df[df['species'] == target_species].copy()
    presence_df['presence'] = 1
    print(f"Found {len(presence_df)} presence records for the target species.")

    if len(presence_df) < 10:
        print("Warning: Very few records for the most common species. Model may be inaccurate.")

    # --- 2. Generate Pseudo-Absence Points ---
    # Machine learning classifiers need examples of both where the species IS (presence)
    # and where it IS NOT (absence). Since we don't have true absence data, we create it
    # by generating random points within the geographic bounds of our study area.
    
    num_absence_points = len(presence_df) * PSEUDO_ABSENCE_RATIO

    # Determine the geographic range from the full dataset
    min_lat, max_lat = df['decimalLatitude'].min(), df['decimalLatitude'].max()
    min_lon, max_lon = df['decimalLongitude'].min(), df['decimalLongitude'].max()

    # Generate random latitude and longitude points
    np.random.seed(42) # for reproducibility
    random_lat = np.random.uniform(min_lat, max_lat, num_absence_points)
    random_lon = np.random.uniform(min_lon, max_lon, num_absence_points)

    absence_df = pd.DataFrame({
        'decimalLatitude': random_lat,
        'decimalLongitude': random_lon,
        'presence': 0
    })
    print(f"Generated {len(absence_df)} pseudo-absence records.")
    
    # --- 3. Combine and Save the Dataset ---
    # We only need the location and the presence/absence columns for the model
    ml_df = pd.concat([
        presence_df[['decimalLatitude', 'decimalLongitude', 'presence']],
        absence_df
    ])

    # Shuffle the dataset to mix presence and absence records
    ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure the output directory exists
    os.makedirs(ML_DATA_DIR, exist_ok=True)
    
    ml_df.to_csv(ML_DATA_FILE, index=False)
    print(f"Successfully created ML-ready dataset with {len(ml_df)} records.")
    print(f"File saved to {ML_DATA_FILE}")

if __name__ == "__main__":
    prepare_ml_data()

