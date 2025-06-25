import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt


# Config
BIRD_DIR = 'C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\Micro-Doppler\\birdcsv'
DRONE_DIR = 'C:\\Users\\arnav\\OneDrive\\Desktop\\Dev\\Micro-Doppler\\dronecsv'
WINDOW_SIZE = 128
STRIDE = 64
EPOCHS = 3
BATCH_SIZE = 64

# Load CSV data
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

# Segment and apply FFT
def create_fft_segments(sequence, label, window_size=128, stride=64):
    segments = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[i:i+window_size]
        fft_window = np.abs(np.fft.fft(window))[:window_size // 2]  # Keep only positive frequencies
        segments.append((fft_window, label))
    return segments

# Load all data
bird_data = load_csv_sequences(BIRD_DIR, 0)
drone_data = load_csv_sequences(DRONE_DIR, 1)
all_data = bird_data + drone_data

X, y = [], []
for seq, label in all_data:
    if len(seq) >= WINDOW_SIZE:
        segments = create_fft_segments(seq, label, WINDOW_SIZE, STRIDE)
        for segment, lbl in segments:
            X.append(segment)
            y.append(lbl)
    else:
        print(f"⚠️ Skipped short sequence: length = {len(seq)}")

X = np.array(X)
y = np.array(y)

# Normalize FFT features (per segment)
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# Reshape for CNN: (samples, features, 1)
X = X[..., np.newaxis]

print(f"Prepared {X.shape[0]} FFT samples with shape {X.shape[1:]}")

# Build CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE//2, 1)),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# Evaluate
y_pred = (model.predict(X) > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

# Save model
model.save('microdoppler_fft_cnn.h5')
print("✅ FFT-based model saved as 'microdoppler_fft_cnn.h5'")




plt.plot(np.abs(np.fft.fft(X[0].flatten())))
plt.title("FFT of a sample segment")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.show()
