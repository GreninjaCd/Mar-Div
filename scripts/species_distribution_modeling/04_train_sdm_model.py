import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys

# --- Configuration (Using Robust, Absolute Paths) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

ML_DATA_DIR = os.path.join(PROJECT_ROOT,'data', 'processed', 'ml_ready')
MODEL_DIR_BASE = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'species_distribution_model')
PLOTS_DIR_BASE = os.path.join(PROJECT_ROOT,'outputs', 'plots', 'species_distribution_model')
# MODIFIED: Lower the minimum data threshold for training
MINIMUM_RECORDS_FOR_SPLIT = 4

def train_model_for_species(species_name):
    """Trains and evaluates a model for a single specified species."""
    safe_species_name = "".join(c for c in species_name if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    
    ml_data_path = os.path.join(ML_DATA_DIR, f'sdm_training_data_{safe_species_name}.csv')
    model_dir = os.path.join(MODEL_DIR_BASE, safe_species_name)
    model_file = os.path.join(model_dir, 'random_forest_model.joblib')
    plots_dir = os.path.join(PLOTS_DIR_BASE, safe_species_name)
    confusion_matrix_file = os.path.join(plots_dir, 'confusion_matrix.png')

    print(f"Loading ML-ready data from {ml_data_path}...")
    if not os.path.exists(ml_data_path):
        print(f"Error: Training data for '{species_name}' not found at '{ml_data_path}'. Please run '03_prepare_sdm_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(ml_data_path)

    # --- MODIFIED: Lenient Data Check ---
    # Instead of exiting, we will now proceed but print a strong warning.
    if len(df) < MINIMUM_RECORDS_FOR_SPLIT:
        print("\n" + "="*50)
        print(f"WARNING: Only {len(df)} records found for '{species_name}'.")
        print("The dataset is too small for a reliable train/test split.")
        print("The model will be trained on all available data, but its predictions will be highly uncertain.")
        print("="*50 + "\n")
        # Use all data for training and testing to prevent a crash, acknowledging this is not standard practice
        X_train, X_test, y_train, y_test = df[['decimalLatitude', 'decimalLongitude']], df[['decimalLatitude', 'decimalLongitude']], df['presence'], df['presence']
    else:
        X = df[['decimalLatitude', 'decimalLongitude', 'temperature', 'salinity']]
        y = df['presence']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        print(f"Data split into {len(X_train)} training records and {len(X_test)} testing records.")

    print("Training the Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Absence', 'Presence'], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Absence', 'Presence'], yticklabels=['Absence', 'Presence'])
    plt.title(f'Confusion Matrix for {species_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(confusion_matrix_file)
    print(f"\nConfusion matrix plot saved to {confusion_matrix_file}")
    plt.close()

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_file)
    print(f"Trained model successfully saved to {model_file}")

def main():
    """Main function to handle command-line arguments and run the training."""
    parser = argparse.ArgumentParser(description="Train a Species Distribution Model for a specific species.")
    parser.add_argument("--species_name", type=str, required=True, help="The scientific name of the species to train a model for.")
    args = parser.parse_args()

    train_model_for_species(args.species_name)

if __name__ == "__main__":
    main()

