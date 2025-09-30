import requests
import pandas as pd
import os

# --- Configuration ---
# API endpoint for OBIS
API_URL = "https://api.obis.org/v3/occurrence"
# Geographic area for the Indian Exclusive Economic Zone (EEZ)
GEOMETRY_POLYGON = 'POLYGON((68 8, 97 8, 97 37, 68 37, 68 8))'
# Maximum records per API request (OBIS has a limit of 10,000)
BATCH_SIZE = 10000
# Development limit for number of records
DEV_RECORD_LIMIT = 30000
# Output file path
OUTPUT_DIR = os.path.join('..', 'data', 'raw')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'obis_indian_ocean_raw.csv')

def fetch_all_data():
    """
    Fetches all occurrence data from the OBIS API for the specified region.
    Handles pagination to retrieve more than the BATCH_SIZE limit.
    """
    print("Starting data fetch from OBIS...")
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_records = []
    offset = 0
    
    while True:
        params = {
            'geometry': GEOMETRY_POLYGON,
            'size': BATCH_SIZE,
            'offset': offset
        }
        
        try:
            response = requests.get(API_URL, params=params)
            response.raise_for_status() # Raise an exception for bad status codes
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print("No more records found. Fetch complete.")
                break
                
            all_records.extend(results)
            print(f"Fetched {len(results)} records. Total fetched: {len(all_records)}")
            
            # Stop fetching after reaching the development limit
            if len(all_records) >= DEV_RECORD_LIMIT:
                print(f"\nReached development limit of {DEV_RECORD_LIMIT} records. Stopping fetch.")
                break
            
            if len(results) < BATCH_SIZE:
                print("All records have been fetched.")
                break

            offset += BATCH_SIZE

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break
            
    if not all_records:
        print("No data was fetched. Exiting.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved {len(df)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_all_data()

