import pandas as pd
import os

# --- Configuration ---
# Input file path for the raw data
RAW_DATA_PATH = os.path.join('..', 'data', 'raw', 'obis_indian_ocean_raw.csv')
# Output file path for the cleaned data
PROCESSED_DATA_DIR = os.path.join('..', 'data', 'processed')
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'cleaned_obis_data.csv')

def process_data():
    """
    Cleans the raw OBIS data by selecting relevant columns and removing records
    with missing essential information (species name, coordinates).
    """
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    
    if not os.path.exists(RAW_DATA_PATH):
        print("Error: Raw data file not found. Please run '01_fetch_data.py' first.")
        return
        
    # Load the dataset
    # Using low_memory=False can help prevent type inference issues with large, mixed-type columns
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} records.")

    # --- 1. Select a Subset of Useful Columns ---
    # The raw data has many columns; we only need a few for our analysis
    columns_to_keep = [
        'species', 'decimalLatitude', 'decimalLongitude', 'eventDate',
        'scientificName', 'phylum', 'class', 'order', 'family', 'genus'
    ]
    
    # Filter for columns that actually exist in the dataframe to avoid errors
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]
    print(f"Selected {len(existing_columns)} relevant columns.")
    
    # --- 2. Clean the Data ---
    # The most critical step is to remove rows where we don't have a species name
    # or the geographic location, as these are useless for our model.
    initial_rows = len(df)
    df.dropna(subset=['species', 'decimalLatitude', 'decimalLongitude'], inplace=True)
    rows_after_cleaning = len(df)
    
    print(f"Removed {initial_rows - rows_after_cleaning} records with missing species name or coordinates.")
    
    if df.empty:
        print("No data left after cleaning. Exiting.")
        return
        
    # --- 3. Save the Processed Data ---
    # Ensure the output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Successfully cleaned data and saved {len(df)} records to {PROCESSED_DATA_FILE}")

if __name__ == "__main__":
    process_data()

# Note: This script assumes that '01_fetch_data.py' has been run successfully
# and that the raw data file exists at the specified path.