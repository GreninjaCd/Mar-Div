import pandas as pd
import numpy as np
import os
import argparse
import sys

# --- Configuration (Using Robust, Absolute Paths) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'combined', 'fused_species_ocean_data.csv')
ML_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ml_ready')
PSEUDO_ABSENCE_RATIO = 1
# This is now just for the warning, not for stopping the script
MINIMUM_RECORDS_THRESHOLD = 10

def prepare_ml_data_for_species(df, species_name):
    """Prepares ML-ready data for a single specified species."""
    print(f"Attempting to prepare data for target species: '{species_name}'")
    safe_species_name = "".join(c for c in species_name if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    ml_data_file = os.path.join(ML_DATA_DIR, f'sdm_training_data_{safe_species_name}.csv')

    # --- Robust Species Matching ---
    df['species_lower'] = df['species'].str.lower().str.strip()
    species_name_lower = species_name.lower().strip()

    print(f"Checking if '{species_name_lower}' exists in the dataset...")
    if species_name_lower not in df['species_lower'].unique():
        print(f"Error: Species '{species_name}' not found in the dataset.")
        top_species = df['species'].value_counts().nlargest(5).index.tolist()
        print("\nDid you mean one of these common species?")
        for sp in top_species:
            print(f"- {sp}")
        return False, None
    
    print("Species found. Proceeding with data preparation.")
    original_species_name = df[df['species_lower'] == species_name_lower]['species'].iloc[0]
    presence_df = df[df['species'] == original_species_name].copy()
    
    # --- MODIFIED: Changed from an error to a warning ---
    # The script will no longer stop if records are below the threshold.
    if len(presence_df) < MINIMUM_RECORDS_THRESHOLD:
        print(f"Warning: Only found {len(presence_df)} records for '{species_name}'.")
        print(f"The resulting model will likely be inaccurate due to the small number of data points.")
    
    # If there are zero records, we must stop.
    if len(presence_df) == 0:
        print(f"Error: Found 0 records for '{species_name}'. Cannot create a data file.")
        return False, None

    presence_df['presence'] = 1
    print(f"Found {len(presence_df)} presence records for the target species.")

    # Generate Pseudo-Absence Points
    print("Generating pseudo-absence points...")
    num_absence_points = len(presence_df) * PSEUDO_ABSENCE_RATIO
        
    min_lat, max_lat = df['decimalLatitude'].min(), df['decimalLatitude'].max()
    min_lon, max_lon = df['decimalLongitude'].min(), df['decimalLongitude'].max()

    np.random.seed(42)
    random_lat = np.random.uniform(min_lat, max_lat, num_absence_points)
    random_lon = np.random.uniform(min_lon, max_lon, num_absence_points)

    absence_df = pd.DataFrame({
        'decimalLatitude': random_lat,
        'decimalLongitude': random_lon,
        'presence': 0
    })
    print(f"Generated {len(absence_df)} pseudo-absence records.")
    
    # Combine and Save
    print("Combining presence and absence data...")
    ml_df = pd.concat([
        presence_df[['decimalLatitude', 'decimalLongitude', 'presence']],
        absence_df
    ])
    ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Saving ML-ready data to {ml_data_file}...")
    os.makedirs(ML_DATA_DIR, exist_ok=True)
    ml_df.to_csv(ml_data_file, index=False)
    print(f"Successfully created ML-ready dataset with {len(ml_df)} records.")
    return True, ml_data_file

def main():
    """Main function to handle command-line arguments and run the data prep."""
    parser = argparse.ArgumentParser(description="Prepare data for Species Distribution Modeling for a specific species.")
    parser.add_argument("--species_name", type=str, required=True, help="The scientific name of the species to model (e.g., 'Thunnus albacares').")
    args = parser.parse_args()

    print("--- Starting On-Demand Data Preparation ---")
    
    try:
        if not os.path.exists(CLEANED_DATA_PATH):
            print(f"Error: Cleaned data not found at {CLEANED_DATA_PATH}.")
            sys.exit(1)

        df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
        
        success, output_file_path = prepare_ml_data_for_species(df, args.species_name)
        
        # Final Self-Verification Step
        if success and output_file_path and os.path.exists(output_file_path):
             print("--- On-Demand Data Preparation Finished Successfully ---")
             sys.exit(0)
        else:
             print("\n--- On-Demand Data Preparation FAILED ---")
             sys.exit(1)

    except Exception as e:
        print(f"\nAn unexpected error occurred in 03_prepare_sdm_data.py: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

