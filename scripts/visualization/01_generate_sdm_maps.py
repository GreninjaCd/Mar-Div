import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Input file path for the cleaned data
CLEANED_DATA_PATH = os.path.join('..', 'data', 'processed', 'cleaned_obis_data.csv')
# Input file path for the trained model
MODEL_PATH = os.path.join('..', 'outputs', 'models', 'species_distribution_model.joblib')
# Output directory for plots
PLOTS_DIR = os.path.join('..', 'outputs', 'plots')
# Output file for the general species distribution plot
DISTRIBUTION_PLOT_FILE = os.path.join(PLOTS_DIR, 'species_distribution.png')
# Output file for the prediction map
PREDICTION_MAP_FILE = os.path.join(PLOTS_DIR, 'prediction_map.png')


def generate_distribution_plot():
    """
    Creates and saves a scatter plot showing the geographic distribution
    of all species records in the cleaned dataset.
    """
    print("Generating general species distribution plot...")
    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"Error: Cleaned data not found at {CLEANED_DATA_PATH}. Please run previous scripts.")
        return

    df = pd.read_csv(CLEANED_DATA_PATH)
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='decimalLongitude', y='decimalLatitude', s=5, alpha=0.3)
    plt.title('Geographic Distribution of All Species Records')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(DISTRIBUTION_PLOT_FILE)
    print(f"Distribution plot saved to {DISTRIBUTION_PLOT_FILE}")
    plt.close()


def generate_prediction_map():
    """
    Loads the trained model and generates a prediction map showing the
    probability of species presence across the study area.
    """
    print("\nGenerating prediction map...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLEANED_DATA_PATH):
        print("Error: Model or cleaned data not found. Please run previous scripts.")
        return

    # Load the trained model
    model = joblib.load(MODEL_PATH)
    # Load the data to get geographic bounds and actual presence points
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # --- Create a grid of points covering the entire region ---
    # This grid represents the "pixels" of our map.
    min_lon, max_lon = df['decimalLongitude'].min(), df['decimalLongitude'].max()
    min_lat, max_lat = df['decimalLatitude'].min(), df['decimalLatitude'].max()
    
    # Create a meshgrid (a grid of coordinate points)
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(min_lon, max_lon, 200), # 200 points for longitude
        np.linspace(min_lat, max_lat, 200)  # 200 points for latitude
    )
    
    # Flatten the grid to create a list of points to predict on
    grid_points = np.c_[grid_lat.ravel(), grid_lon.ravel()]
    
    # --- Make Predictions on the Grid ---
    # Use predict_proba to get the probability of presence (class 1)
    print("Making predictions across the geographic grid...")
    probabilities = model.predict_proba(grid_points)[:, 1]
    
    # Reshape the probabilities back into a 2D grid for plotting
    Z = probabilities.reshape(grid_lon.shape)

    # --- Plot the Prediction Map ---
    print("Plotting the prediction map...")
    plt.figure(figsize=(12, 10))
    
    # Create a contour plot (heatmap) of the probabilities
    contour = plt.contourf(grid_lon, grid_lat, Z, cmap='viridis', levels=20)
    plt.colorbar(contour, label='Probability of Presence')

    # Overlay the actual presence points of the target species for context
    target_species = df['species'].mode()[0]
    presence_points = df[df['species'] == target_species]
    plt.scatter(presence_points['decimalLongitude'], presence_points['decimalLatitude'], 
                s=10, c='red', edgecolor='white', alpha=0.75, label=f'Actual Presence: {target_species}')

    plt.title('Predicted Species Distribution Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    plt.savefig(PREDICTION_MAP_FILE)
    print(f"Prediction map saved to {PREDICTION_MAP_FILE}")
    plt.close()


if __name__ == "__main__":
    generate_distribution_plot()
    generate_prediction_map()

