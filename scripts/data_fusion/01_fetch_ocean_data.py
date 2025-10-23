import copernicusmarine as cm
import os
import sys
import xarray as xr
import pandas as pd

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'oceanography')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Dataset ID ---
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# --- Target years ---
YEARS = [2020, 2021, 2022, 2023]


def fetch_and_convert_year(year: int):
    """
    Downloads Copernicus Marine oceanographic data for a given year and converts it safely to CSV.
    """
    nc_filename = f"indian_ocean_enviro_data_{year}.nc"
    csv_filename = f"indian_ocean_enviro_data_{year}.csv"

    nc_path = os.path.join(OUTPUT_DIR, nc_filename)
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)

    print(f"\n=== Fetching and processing data for {year} ===")

    # --- Step 1: Download data if not already present ---
    if not os.path.exists(nc_path):
        try:
            print(f"➡️ Downloading dataset: {DATASET_ID}")
            cm.subset(
                dataset_id=DATASET_ID,
                variables=["thetao", "so"],
                minimum_longitude=68,
                maximum_longitude=97,
                minimum_latitude=8,
                maximum_latitude=37,
                start_datetime=f"{year}-01-01T00:00:00",
                end_datetime=f"{year}-12-31T23:59:59",
                output_filename=nc_filename,
                output_directory=OUTPUT_DIR
            )
            print(f"✅ Download complete: {nc_filename}")
        except Exception as e:
            print(f"❌ Download failed for {year}: {e}")
            return
    else:
        print(f"✔️ Found existing NetCDF file: {nc_filename}")

    # --- Step 2: Convert .nc → .csv safely ---
    try:
        print(f"➡️ Converting {nc_filename} to CSV (chunked + cleaned)...")

        ds = xr.open_dataset(nc_path, chunks={"time": 1})

        print("📦 Variables in dataset:", list(ds.data_vars))

        # Surface only (depth = 0)
        if "depth" in ds.dims:
            ds = ds.sel(depth=ds.depth[0])

        # Extract only relevant variables (temperature & salinity)
        if "thetao" in ds and "so" in ds:
            ds = ds[["thetao", "so"]]
        elif "temperature" in ds and "salinity" in ds:
            ds = ds[["temperature", "salinity"]]
        else:
            raise ValueError("❌ Could not find temperature/salinity variables in dataset")

        # Get all time steps (lazy loading)
        times = ds.time.values

        # Remove old CSV if exists
        if os.path.exists(csv_path):
            os.remove(csv_path)

        for i, t in enumerate(times):
            subset = ds.sel(time=t)
            df = subset.to_dataframe().reset_index()

            # Rename to standardized names
            df = df.rename(columns={
                "thetao": "temperature_C",
                "so": "salinity_PSU",
                "temperature": "temperature_C",
                "salinity": "salinity_PSU"
            })

            # Drop missing values
            df = df.dropna(subset=["temperature_C", "salinity_PSU"])

            # Append to CSV incrementally
            df.to_csv(csv_path, mode='a', index=False, header=(i == 0))

            print(f"   🕒 Processed {i + 1}/{len(times)} timesteps for {year}")

        print(f"✅ CSV conversion complete: {csv_filename}")

    except Exception as e:
        print(f"\n❌ ERROR converting {nc_filename} to CSV.")
        print(f"Original error: {e}")
        sys.exit(1)


def fetch_copernicus_data():
    """
    Fetches and processes Copernicus Marine data for all specified years.
    """
    print("--- Starting Multi-Year Oceanographic Data Fetch ---")
    for year in YEARS:
        fetch_and_convert_year(year)

    print("\n🎉 All downloads and conversions complete!")
    print(f"Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    fetch_copernicus_data()
