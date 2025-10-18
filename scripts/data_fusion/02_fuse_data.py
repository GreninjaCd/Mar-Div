import pandas as pd
import xarray as xr
import os
import sys

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Input files
CLEANED_BIO_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'biodiversity', 'cleaned_obis_data.csv')
OCEAN_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'oceanography', 'indian_ocean_enviro_data.nc')

# Output file
FUSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'combined')
FUSED_DATA_FILE = os.path.join(FUSED_DATA_DIR, 'fused_species_ocean_data.csv')

def fuse_data():
    """
    Fuses biodiversity occurrence data with oceanographic data by matching
    the closest time and location.
    """
    print("--- Starting Data Fusion Process ---")

    # --- Load Datasets ---
    print(f"Loading biodiversity data from: {CLEANED_BIO_PATH}")
    if not os.path.exists(CLEANED_BIO_PATH):
        print("Error: Cleaned biodiversity data not found. Please run the processing script first.")
        sys.exit(1)
    df_bio = pd.read_csv(CLEANED_BIO_PATH, parse_dates=['eventDate'])

    print(f"Loading oceanographic data from: {OCEAN_DATA_PATH}")
    if not os.path.exists(OCEAN_DATA_PATH):
        print("Error: Oceanographic data not found. Please run '01_fetch_ocean_data.py' first.")
        sys.exit(1)
    ds_ocean = xr.open_dataset(OCEAN_DATA_PATH)
    # Rename for easier access
    ds_ocean = ds_ocean.rename({'thetao': 'temperature', 'so': 'salinity', 'latitude': 'lat', 'longitude': 'lon', 'time': 'time'})
    print("Oceanographic data loaded.")

    # --- Match and Extract Environmental Data ---
    print("\nExtracting environmental data for each species observation...")
    print("This step can be slow for large datasets...")

    # Prepare lists to hold the new data
    temperatures = []
    salinities = []
    
    # Convert pandas Series to xarray DataArrays for efficient lookup
    lats = xr.DataArray(df_bio['decimalLatitude'], dims="occurrence")
    lons = xr.DataArray(df_bio['decimalLongitude'], dims="occurrence")
    times = xr.DataArray(df_bio['eventDate'], dims="occurrence")

    # Use xarray's advanced selection to find the nearest data point for all occurrences at once
    extracted_temp = ds_ocean['temperature'].sel(lat=lats, lon=lons, time=times, method='nearest').to_pandas()
    extracted_sal = ds_ocean['salinity'].sel(lat=lats, lon=lons, time=times, method='nearest').to_pandas()

    df_bio['temperature'] = extracted_temp.values
    df_bio['salinity'] = extracted_sal.values

    print("Environmental data extraction complete.")

    # --- Clean and Save Fused Data ---
    print("\nCleaning the fused dataset (removing rows with no matching env data)...")
    initial_rows = len(df_bio)
    df_bio.dropna(subset=['temperature', 'salinity'], inplace=True)
    final_rows = len(df_bio)
    print(f"Removed {initial_rows - final_rows} records that were outside the environmental data's range.")
    
    os.makedirs(FUSED_DATA_DIR, exist_ok=True)
    df_bio.to_csv(FUSED_DATA_FILE, index=False)
    print(f"\nFusion complete! Saved {final_rows} enriched records to {FUSED_DATA_FILE}")

if __name__ == "__main__":
    fuse_data()