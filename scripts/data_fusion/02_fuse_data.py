# scripts/data_fusion/02_fuse_data.py

import pandas as pd
import glob
from scipy.spatial import cKDTree
import os

print("--- Starting Data Fusion Process ---")

# Paths
obis_file = "data/processed/biodiversity/cleaned_obis_data.csv"
ocean_files = glob.glob("data/raw/oceanography/indian_ocean_enviro_data_*.csv")
output_file = "data/processed/ml_ready/fused_obis_ocean.csv"

# 1️⃣ Load OBIS biodiversity data
print(f"Loading biodiversity data from: {obis_file}")
biodiv = pd.read_csv(obis_file)
print(f"Loaded {len(biodiv)} biodiversity records.")

# Convert eventDate to datetime (UTC)
biodiv['eventDate'] = pd.to_datetime(biodiv['eventDate'], errors='coerce', utc=True)

# Drop records without coordinates
biodiv = biodiv.dropna(subset=['decimalLatitude', 'decimalLongitude'])
print(f"{len(biodiv)} records remaining after dropping missing coordinates.")

# 2️⃣ Load oceanographic data
ocean_list = []
for f in ocean_files:
    print(f"Loading ocean data from: {f}")
    df = pd.read_csv(f)
    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    df = df.dropna(subset=['latitude', 'longitude'])
    df['year'] = df['time'].dt.year
    ocean_list.append(df)

ocean = pd.concat(ocean_list, ignore_index=True)
print(f"Total oceanographic records: {len(ocean)}")

# 3️⃣ Spatial nearest neighbor fusion (fast)
print("Building global spatial KD-Tree for ocean data...")
ocean_coords = ocean[['latitude', 'longitude']].values
tree = cKDTree(ocean_coords)

species_coords = biodiv[['decimalLatitude', 'decimalLongitude']].values
dist, idx = tree.query(species_coords, k=1)

biodiv['temperature_C'] = ocean.iloc[idx]['temperature_C'].values
biodiv['salinity_PSU'] = ocean.iloc[idx]['salinity_PSU'].values
biodiv['depth'] = ocean.iloc[idx]['depth'].values
biodiv['ocean_time'] = ocean.iloc[idx]['time'].values

# Ensure ocean_time is datetime UTC
biodiv['ocean_time'] = pd.to_datetime(biodiv['ocean_time'], errors='coerce', utc=True)

# 4️⃣ Fill missing eventDate with nearest ocean_time
missing_dates = biodiv['eventDate'].isna().sum()
if missing_dates > 0:
    print(f"Filling {missing_dates} missing eventDate(s) with nearest ocean time")
    biodiv['eventDate'] = biodiv['eventDate'].fillna(biodiv['ocean_time'])

# Extract year
biodiv['eventYear'] = biodiv['eventDate'].dt.year

# 5️⃣ Optional: Temporal nearest neighbor by year (efficient)
print("Applying temporal nearest neighbor matching per year...")
fused_groups = []
for year, group in biodiv.groupby('eventYear'):
    subset = ocean[ocean['year'] == year]
    if subset.empty:
        fused_groups.append(group)  # fallback: keep original nearest neighbor
        continue
    # Build KD-Tree once per year
    tree_year = cKDTree(subset[['latitude', 'longitude']].values)
    coords = group[['decimalLatitude', 'decimalLongitude']].values
    _, idx_year = tree_year.query(coords, k=1)
    # Assign year-matched ocean data
    group['temperature_C'] = subset.iloc[idx_year]['temperature_C'].values
    group['salinity_PSU'] = subset.iloc[idx_year]['salinity_PSU'].values
    group['depth'] = subset.iloc[idx_year]['depth'].values
    group['ocean_time'] = subset.iloc[idx_year]['time'].values
    fused_groups.append(group)

biodiv = pd.concat(fused_groups, ignore_index=True)

# 6️⃣ Save fused dataset
os.makedirs(os.path.dirname(output_file), exist_ok=True)
biodiv.to_csv(output_file, index=False)
print(f"Fused dataset saved to: {output_file}")
print(f"Total records in fused dataset: {len(biodiv)}")
