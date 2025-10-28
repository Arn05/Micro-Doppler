import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# --- Global Constants (Must be consistent with training) ---
WINDOW_SIZE = 128
STRIDE = 64
class_names = ["Bird", "Drone"] # 0 = Bird, 1 = Drone

# --- Model Loading ---
try:
    # Ensure 'accuracy.h5' is in the same directory as this script
    model = load_model('accuracy.h5')
    print("Model 'accuracy.h5' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if the model can't be loaded
    exit()

# ----------------------------------------------------------------------
# --- Data Preprocessing Function ---
# ----------------------------------------------------------------------

def load_and_process_test_file(file_path):
    """Loads a CSV file, computes FFT segments, and standardizes the data."""
    df = pd.read_csv(file_path)
    if 'V' not in df.columns:
        raise ValueError("CSV missing 'V' column")
    
    # Extract the signal and drop non-numeric/missing values
    v = pd.to_numeric(df['V'], errors='coerce').dropna().values

    segments = []
    # Loop through the signal, creating overlapping windows
    for i in range(0, len(v) - WINDOW_SIZE + 1, STRIDE):
        window = v[i:i+WINDOW_SIZE]
        # Compute the magnitude of the FFT (half the window size for single-sided spectrum)
        fft_window = np.abs(np.fft.fft(window))[:WINDOW_SIZE//2]
        segments.append(fft_window)

    if not segments:
        raise ValueError("File is too short to create segments.")

    X = np.array(segments)
    
    # Standardize the segments (important for neural networks)
    # Avoid division by zero if a segment has zero standard deviation
    std_dev = X.std(axis=1, keepdims=True)
    std_dev[std_dev == 0] = 1e-8 # Small epsilon for stable division
    
    X = (X - X.mean(axis=1, keepdims=True)) / std_dev
    
    # Add a channel dimension for Keras/TensorFlow (e.g., for 1D CNN)
    X = X[..., np.newaxis]
    return X

# ----------------------------------------------------------------------
# --- Evaluation Helper Functions ---
# ----------------------------------------------------------------------

def get_test_files_and_labels(test_data_dir):
    """Gathers file paths and their true labels from structured directories."""
    file_paths = []
    true_labels = [] # 0 for Bird, 1 for Drone
    label_map = {"Bird": 0, "Drone": 1}

    for class_name in class_names:
        class_dir = os.path.join(test_data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found for class '{class_name}': {class_dir}")
            continue

        for filename in os.listdir(class_dir):
            if filename.endswith(".csv"):
                file_paths.append(os.path.join(class_dir, filename))
                true_labels.append(label_map[class_name])
                
    return file_paths, true_labels

# ----------------------------------------------------------------------
# --- Main Evaluation Function ---
# ----------------------------------------------------------------------

def evaluate_model_on_dataset(test_data_dir, model):
    """Runs the model on the full test dataset and calculates statistics."""
    file_paths, y_true_all = get_test_files_and_labels(test_data_dir)
    y_pred_all = [] 

    if not file_paths:
        print("No test files found. Check your directory path and structure.")
        return

    print(f"\nProcessing {len(file_paths)} test files...")
    
    for i, file_path in enumerate(file_paths):
        try:
            # Process the file
            X_test = load_and_process_test_file(file_path)
            
            # Predict segment probabilities
            probs = model.predict(X_test, verbose=0)
            predictions = (probs > 0.5).astype(int).flatten()

            # Determine the majority prediction for the entire file
            num_bird = np.sum(predictions == 0)
            num_drone = np.sum(predictions == 1)
            final_pred = 0 if num_bird >= num_drone else 1
            y_pred_all.append(final_pred)
            
            print(f"[{i+1}/{len(file_paths)}] Classified: {os.path.basename(file_path)} as {class_names[final_pred]}")


        except Exception as e:
            # If an error occurs, print it and remove the corresponding true label
            print(f"Skipping file {file_path} due to error: {e}")
            del y_true_all[len(y_pred_all)] # Remove the true label that corresponds to the skipped file
            continue # Continue to the next file

    # --- Calculate Statistics ---
    
    if len(y_true_all) == 0 or len(y_true_all) != len(y_pred_all):
         print("\nError: Evaluation failed or no files were successfully processed.")
         return

    # Calculate the Confusion Matrix (Drone=1 is the Positive Class)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # Calculate Metrics
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)

    print("\n" + "="*50)
    print("--- Model Evaluation Results (Drone is Positive Class) ---")
    print(f"Total Test Files Processed: {len(y_true_all)}")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\n--- Confusion Matrix ---")
    print(f"True Negatives (True Bird, Predicted Bird): {TN}")
    print(f"False Positives (True Bird, Predicted Drone): {FP}")
    print(f"False Negatives (True Drone, Predicted Bird): {FN}")
    print(f"True Positives (True Drone, Predicted Drone): {TP}")
    print("="*50)


# --- Execution Block ---
if __name__ == "__main__":
    # !!! CHANGE THIS PATH to the root directory of your test data !!!
    TEST_DATA_DIRECTORY = r"C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\Micro-Doppler\\test_dataset" 
    
    evaluate_model_on_dataset(TEST_DATA_DIRECTORY, model)