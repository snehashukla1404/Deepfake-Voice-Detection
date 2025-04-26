import os
import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load model and scaler
model = load_model("deepfake_voice_detection.h5")
scaler = joblib.load("scaler.save")

# Extract 26 features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs = np.mean(mfcc, axis=1)

        features = np.hstack([chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfccs])
        return features
    except Exception as e:
        print("Feature extraction error:", e)
        return None

# Make prediction
def predict_audio(file_path):
    features = extract_features(file_path)
    if features is None or features.shape[0] != 26:
        result_label.config(text="Error: Could not extract features properly.")
        return

    features_scaled = scaler.transform([features])
    features_reshaped = features_scaled.reshape((1, 26, 1))
    prediction = model.predict(features_reshaped)[0][0]

    if prediction >= 0.5:
        label = "REAL"
        confidence = prediction
    else:
        label = "FAKE"
        confidence = 1 - prediction

    result_label.config(
        text=f"🎤 Prediction: {label}\n🔍 Confidence: {confidence:.2%}"
    )

# Browse button action
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        predict_audio(file_path)

# GUI Design
root = tk.Tk()
root.title("🔊 Deepfake Voice Detector")
root.geometry("500x350")
root.configure(bg="#f2f2f2")

title = tk.Label(
    root, text="Deepfake Voice Detector",
    font=("Helvetica", 20, "bold"), fg="#333", bg="#f2f2f2"
)
title.pack(pady=20)

desc = tk.Label(
    root,
    text="Upload a WAV file to analyze if the voice is FAKE or REAL.",
    font=("Helvetica", 12), bg="#f2f2f2"
)
desc.pack(pady=5)

upload_btn = tk.Button(
    root, text="📂 Upload WAV File",
    command=browse_file,
    font=("Helvetica", 14),
    bg="#4CAF50", fg="white",
    padx=20, pady=10, relief="flat"
)
upload_btn.pack(pady=20)

result_label = tk.Label(
    root,
    text="Result will be displayed here.",
    font=("Helvetica", 14), fg="#333", bg="#f2f2f2", wraplength=400, justify="center"
)
result_label.pack(pady=30)

root.mainloop()
