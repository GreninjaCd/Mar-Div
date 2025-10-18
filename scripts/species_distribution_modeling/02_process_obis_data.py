import pandas as pd
import os

# --- Configuration (Updated Paths) ---
# The script is now two levels deep (in scripts/species_distribution_modeling), so we go up twice ('../../')
RAW_DATA_PATH = os.path.join('..', '..', 'data', 'raw', 'biodiversity', 'obis_indian_ocean_raw.csv')

# Output directory for the cleaned data
PROCESSED_DATA_DIR = os.path.join('..', '..', 'data', 'processed', 'biodiversity')
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'cleaned_obis_data.csv')

def process_data():
    """
    Cleans the raw OBIS data by selecting relevant columns and removing records
    with missing essential information (species name, coordinates).
    """
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data file not found at '{RAW_DATA_PATH}'. Please run '01_fetch_obis_data.py' first.")
        return
        
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} records.")

    # --- 1. Select a Subset of Useful Columns ---
    columns_to_keep = [
        'species', 'decimalLatitude', 'decimalLongitude', 'eventDate',
        'scientificName', 'phylum', 'class', 'order', 'family', 'genus'
    ]
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]
    print(f"Selected {len(existing_columns)} relevant columns.")
    
    # --- 2. Clean the Data ---
    initial_rows = len(df)
    df.dropna(subset=['species', 'decimalLatitude', 'decimalLongitude'], inplace=True)
    rows_after_cleaning = len(df)
    
    print(f"Removed {initial_rows - rows_after_cleaning} records with missing species name or coordinates.")
    
    if df.empty:
        print("No data left after cleaning. Exiting.")
        return
        
    # --- 3. Save the Processed Data ---
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Successfully cleaned data and saved {len(df)} records to {PROCESSED_DATA_FILE}")

if __name__ == "__main__":
    process_data()
