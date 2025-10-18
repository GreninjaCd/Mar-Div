import copernicusmarine as cm
import os
import sys

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'oceanography')
OUTPUT_FILENAME = "indian_ocean_enviro_data.nc"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# --- Use stable, hardcoded Dataset ID ---
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

def fetch_copernicus_data():
    """
    Downloads environmental data (temperature, salinity) for the Indian Ocean region
    from the Copernicus Marine Service.
    """
    print("--- Starting Oceanographic Data Fetch from Copernicus ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        print(f"Data file '{OUTPUT_FILENAME}' already exists. Skipping download.")
        print("To re-download, please delete the existing file.")
        return

    print(f"Using Dataset ID: {DATASET_ID}")
    print(f"Downloading data to: {OUTPUT_FILE}")
    print("This may take several minutes depending on the date range...")

    try:
        cm.subset(
            dataset_id=DATASET_ID,
            variables=["thetao", "so"],
            minimum_longitude=68,
            maximum_longitude=97,
            minimum_latitude=8,
            maximum_latitude=37,
            start_datetime="2021-01-01T00:00:00",
            end_datetime="2021-12-31T23:59:59",
            output_filename=OUTPUT_FILENAME,
            output_directory=OUTPUT_DIR
        )
        print("\nDownload complete!")
        print(f"Data saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print("Copernicus download failed. This can happen if:")
        print("1. Your Copernicus Marine credentials are not configured correctly.")
        print("2. The Copernicus service is temporarily unavailable.")
        print(f"3. The Dataset ID '{DATASET_ID}' has been updated by the provider.")
        print(f"Original error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_copernicus_data()
