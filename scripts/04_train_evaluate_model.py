import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
# Input file path for the ML-ready data
ML_DATA_PATH = os.path.join('..', 'data', 'processed', 'ml_ready_data.csv')
# Output directory for the trained model
MODEL_DIR = os.path.join('..', 'outputs', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'species_distribution_model.joblib')
# Output directory for the confusion matrix plot
PLOTS_DIR = os.path.join('..', 'outputs', 'plots')
CONFUSION_MATRIX_FILE = os.path.join(PLOTS_DIR, 'confusion_matrix.png')

def train_model():
    """
    Trains a Random Forest Classifier to predict species presence,
    evaluates its performance, and saves the trained model.
    """
    print(f"Loading ML-ready data from {ML_DATA_PATH}...")

    if not os.path.exists(ML_DATA_PATH):
        print("Error: ML-ready data file not found. Please run '03_prepare_ml_data.py' first.")
        return

    df = pd.read_csv(ML_DATA_PATH)

    # --- 1. Prepare Data for Training ---
    # X contains the features (our inputs), and y contains the target (what we want to predict)
    X = df[['decimalLatitude', 'decimalLongitude']]
    y = df['presence']

    # Split the data into a training set (to teach the model) and a testing set (to evaluate it)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Data split into {len(X_train)} training records and {len(X_test)} testing records.")

    # --- 2. Train the Random Forest Model ---
    print("Training the Random Forest Classifier...")
    # n_estimators is the number of trees in the forest.
    # random_state ensures the results are reproducible.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 3. Evaluate the Model ---
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    
    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification Report (shows precision, recall, f1-score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Absence', 'Presence']))

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualize the Confusion Matrix
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Absence', 'Presence'], yticklabels=['Absence', 'Presence'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CONFUSION_MATRIX_FILE)
    print(f"\nConfusion matrix plot saved to {CONFUSION_MATRIX_FILE}")

    # --- 4. Save the Trained Model ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Trained model successfully saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()

