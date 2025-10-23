import xarray as xr
import pandas as pd
import os

INPUT_FILE = os.path.join("data", "raw", "oceanography", "indian_ocean_enviro_data.nc")
OUTPUT_FILE = os.path.join("data", "raw", "oceanography", "indian_ocean_surface.csv")

print("--- Loading NetCDF file ---")
ds = xr.open_dataset(INPUT_FILE, chunks={"time": 1})

# Check variable names
print("Variables in dataset:", list(ds.data_vars))

# Extract only surface layer if depth exists
if "depth" in ds.dims:
    ds = ds.sel(depth=ds.depth[0])

# Extract relevant variables (temperature + salinity)
if "thetao" in ds and "so" in ds:
    df = ds[["thetao", "so"]].to_dataframe().reset_index()
elif "temperature" in ds and "salinity" in ds:
    df = ds[["temperature", "salinity"]].to_dataframe().reset_index()
else:
    raise ValueError("❌ Could not find temperature/salinity variables in dataset")

# Rename to clear column names
df = df.rename(columns={
    "thetao": "temperature_C",
    "so": "salinity_PSU",
    "temperature": "temperature_C",
    "salinity": "salinity_PSU"
})

# Drop missing values
df = df.dropna(subset=["temperature_C", "salinity_PSU"])

print("--- Saving as CSV ---")
df.to_csv(OUTPUT_FILE, index=False)
print(f" Saved surface ocean data with temperature and salinity: {OUTPUT_FILE}")
