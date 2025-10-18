import requests
import pandas as pd
import os

# --- Configuration (Updated Paths) ---
API_URL = "https://api.obis.org/v3/occurrence"
GEOMETRY_POLYGON = 'POLYGON((68 8, 97 8, 97 37, 68 37, 68 8))'  # Indian EEZ
BATCH_SIZE = 10000
DEV_RECORD_LIMIT = 30000

# Output file path (Updated)
OUTPUT_DIR = os.path.join('..', '..', 'data', 'raw', 'biodiversity')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'obis_indian_ocean_raw.csv')


def fetch_all_data():
    """
    Fetches all occurrence data from the OBIS API for the specified region.
    Automatically creates folders and saves the data, even if the file doesn't exist yet.
    """
    print(" Starting data fetch from OBIS...")

    # Ensure output directory exists
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
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])

            if not results:
                print(" No more records found. Fetch complete.")
                break

            all_records.extend(results)
            print(f"Fetched {len(results)} records. Total fetched: {len(all_records)}")

            # Stop fetching after reaching development limit
            if 0 < DEV_RECORD_LIMIT <= len(all_records):
                print(f"\n Reached development limit of {DEV_RECORD_LIMIT} records. Stopping fetch.")
                break

            if len(results) < BATCH_SIZE:
                print(" All available records have been fetched.")
                break

            offset += BATCH_SIZE

        except requests.exceptions.RequestException as e:
            print(f" An error occurred: {e}")
            break

    if not all_records:
        print(" No data was fetched. Exiting.")
        return

    # Trim extra records (if any)
    if DEV_RECORD_LIMIT > 0 and len(all_records) > DEV_RECORD_LIMIT:
        all_records = all_records[:DEV_RECORD_LIMIT]

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Save file safely (create if not exists)
    try:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"\n Successfully saved {len(df)} records to '{os.path.abspath(OUTPUT_FILE)}'")
    except Exception as e:
        print(f" Failed to save data: {e}")


if __name__ == "__main__":
    fetch_all_data()
