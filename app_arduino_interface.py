import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

import serial

def send_to_arduino(label):
    try:
        # Replace 'COM3' with your Nano RP2040 port
        with serial.Serial('COM4', 9600, timeout=1) as arduino:
            arduino.write(f"{label}\n".encode())
            print(f"Sent to Arduino: {label}")
    except Exception as e:
        print(f"Arduino communication error: {e}")


model = load_model('accuracy.h5')
WINDOW_SIZE = 128
STRIDE = 64
class_names = ["Bird", "Drone"]

def load_and_process_test_file(file_path):
    df = pd.read_csv(file_path)
    if 'V' not in df.columns:
        raise ValueError("CSV must contain 'V' column")

    v = pd.to_numeric(df['V'], errors='coerce').dropna().values
    segments = []

    for i in range(0, len(v) - WINDOW_SIZE + 1, STRIDE):
        window = v[i:i + WINDOW_SIZE]
        fft_window = np.abs(np.fft.fft(window))[:WINDOW_SIZE // 2]
        segments.append(fft_window)

    X = np.array(segments)
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    X = X[..., np.newaxis]
    return X

# ---------------- GUI Setup ----------------
root = tk.Tk()
root.title("Micro-Doppler Classifier")
root.geometry("500x300")
root.resizable(False, False)

style = ttk.Style(root)
style.theme_use('clam')
style.configure("TButton", font=("Segoe UI", 12), padding=10)
style.configure("TLabel", font=("Segoe UI", 12))

main_frame = ttk.Frame(root, padding=30)
main_frame.pack(expand=True, fill='both')

title = ttk.Label(main_frame, text="Micro-Doppler Object Classifier", font=("Segoe UI", 18, "bold"))
title.pack(pady=(0, 20))

result_label = ttk.Label(main_frame, text="No file loaded.", font=("Segoe UI", 16, "bold"), foreground="#555")
result_label.pack(pady=10)

def classify_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    try:
        X = load_and_process_test_file(file_path)
        probs = model.predict(X)
        preds = (probs > 0.5).astype(int).flatten()

        
        final_class = 0 if np.sum(preds == 0) >= np.sum(preds == 1) else 1
        prediction_text = f"Prediction: {class_names[final_class]}"

        result_label.config(text=prediction_text, foreground="green" if final_class == 1 else "blue")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        result_label.config(text="Failed to classify.")
    
    final_class = 0 if np.sum(preds == 0) >= np.sum(preds == 1) else 1
    prediction_text = f"Prediction: {class_names[final_class]}"
    result_label.config(text=prediction_text, foreground="green" if final_class == 1 else "blue")

   
    send_to_arduino(class_names[final_class])

btn = ttk.Button(main_frame, text="Load CSV and Predict", command=classify_file)
btn.pack(pady=10)

root.mainloop()