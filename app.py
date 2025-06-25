import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

WINDOW_SIZE = 128
STRIDE = 64
class_names = ["Bird", "Drone"]

# Load your model once
model = load_model('microdoppler_fft_cnn.h5')

def load_and_process_test_file(file_path):
    df = pd.read_csv(file_path)
    if 'V' not in df.columns:
        raise ValueError("CSV missing 'V' column")
    v = pd.to_numeric(df['V'], errors='coerce').dropna().values

    segments = []
    for i in range(0, len(v) - WINDOW_SIZE + 1, STRIDE):
        window = v[i:i+WINDOW_SIZE]
        fft_window = np.abs(np.fft.fft(window))[:WINDOW_SIZE//2]
        segments.append(fft_window)

    X = np.array(segments)
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    X = X[..., np.newaxis]
    return X

def classify_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return
    try:
        X_test = load_and_process_test_file(file_path)
        probs = model.predict(X_test)
        predictions = (probs > 0.5).astype(int).flatten()

        # Count predictions
        num_bird = np.sum(predictions == 0)
        num_drone = np.sum(predictions == 1)

        # Decide majority class
        final_label = "Bird" if num_bird >= num_drone else "Drone"

        # Clear previous results in table
        for i in tree.get_children():
            tree.delete(i)

        # Show just the final result in the table
        tree.insert("", "end", values=(1, final_label))

        summary_label.config(text=f"Final Prediction: {final_label}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# --- Tkinter UI setup ---
root = tk.Tk()
root.title("Micro-Doppler Object Classifier")

btn_load = tk.Button(root, text="Load CSV and Classify", command=classify_file)
btn_load.pack(pady=10)

cols = ('Segment', 'Prediction')
tree = ttk.Treeview(root, columns=cols, show='headings', height=15)
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, anchor='center')
tree.pack(expand=True, fill='both')

summary_label = tk.Label(root, text="No file loaded.")
summary_label.pack(pady=10)

root.mainloop()
