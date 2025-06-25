import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# ================================
# CONFIGURATION
# ================================
BIRD_DIR = 'C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\Micro-Doppler\\birdcsv'
DRONE_DIR = 'C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\Micro-Doppler\\dronecsv'
WINDOW_SIZE = 128
STRIDE = 64
EPOCHS = 4
BATCH_SIZE = 64

# ================================
# DATA LOADING & SEGMENTATION
# ================================

def load_csv_sequences(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv") or file.endswith(".tsv"):
            path = os.path.join(folder_path, file)
            df = pd.read_csv(path, sep=',')
            if 'V' in df.columns:
                df['V'] = pd.to_numeric(df['V'], errors='coerce')
                df = df.dropna(subset=['V'])
                data.append((df['V'].values, label))
            else:
                print(f"⚠️ Skipped: Missing 'V' column in {file}")
    return data


def create_segments(sequence, label, window_size=128, stride=64):
    segments = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[i:i+window_size]
        segments.append((window, label))
    return segments

# Load data
bird_data = load_csv_sequences(BIRD_DIR, 0)
drone_data = load_csv_sequences(DRONE_DIR, 1)
all_data = bird_data + drone_data

# Segment data
X, y = [], []
for seq, label in all_data:
    if len(seq) >= WINDOW_SIZE:
        segments = create_segments(seq, label, WINDOW_SIZE, STRIDE)
        for segment, lbl in segments:
            X.append(segment)
            y.append(lbl)
    else:
        print(f"⚠️ Skipped sequence (too short): length = {len(seq)}")


X = np.array(X)
y = np.array(y)

print(f"Loaded {X.shape[0]} segments with shape: {X.shape}")


# Normalize per segment
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# Reshape for 1D CNN input
X = X[..., np.newaxis]  # shape: (samples, window_size, 1)

print(f"Prepared {X.shape[0]} samples with shape {X.shape[1:]}")

# ================================
# MODEL DEFINITION
# ================================

model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(WINDOW_SIZE, 1)),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ================================
# MODEL TRAINING
# ================================
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# ================================
# EVALUATION
# ================================
y_pred = (model.predict(X) > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

# ================================
# SAVE MODEL
# ================================
model.save('microdoppler_cnn.h5')
print("\nModel saved as 'microdoppler_cnn.h5'")
