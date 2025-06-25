from tensorflow.keras.models import load_model
model = load_model('microdoppler_fft_cnn.h5')

import numpy as np
import pandas as pd

WINDOW_SIZE = 128
STRIDE = 64

def load_and_process_test_file(file_path):
    df = pd.read_csv(file_path, sep=',')
    if 'V' not in df.columns:
        raise ValueError("CSV missing 'V' column")
    v = pd.to_numeric(df['V'], errors='coerce').dropna().values

    segments = []
    for i in range(0, len(v) - WINDOW_SIZE + 1, STRIDE):
        window = v[i:i+WINDOW_SIZE]
        fft_window = np.abs(np.fft.fft(window))[:WINDOW_SIZE//2]
        segments.append(fft_window)

    X = np.array(segments)
    # Normalize per segment
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    X = X[..., np.newaxis]  # add channel dimension
    return X

# Load your saved model
from tensorflow.keras.models import load_model
model = load_model('microdoppler_fft_cnn.h5')

# Load and process test data
test_file = 'C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\Micro-Doppler\\bird_scaled\\bird205.csv'  # change this to your test file path
X_test = load_and_process_test_file(test_file)

# Predict probabilities
probs = model.predict(X_test)

# Convert probabilities to labels (0 or 1)
predictions = (probs > 0.5).astype(int).flatten()

print("Predictions for each segment:")
print(predictions)

# Optional: summarize results
num_bird = np.sum(predictions == 0)
num_drone = np.sum(predictions == 1)
print(f"Segments predicted as Bird: {num_bird}")
print(f"Segments predicted as Drone: {num_drone}")
