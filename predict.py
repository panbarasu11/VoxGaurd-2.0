"""
predict.py
----------
Use the trained VoxGuard model to check if a voice recording is REAL or FAKE.

Usage:
    python predict.py path/to/audio.wav
    python predict.py path/to/folder_of_audios/
"""

import sys
import os
import numpy as np
import tensorflow as tf

import config
from extract_features import extract_mfcc


def predict_single(model, file_path):
    """
    Predict whether a single audio file is REAL or FAKE.
    Returns: label string and confidence percentage.
    """
    mfcc = extract_mfcc(file_path)

    if mfcc is None:
        return None, None

    # Model expects shape: (1, time_steps, n_mfcc)
    # We may need to resize to match what the model was trained on
    expected_timesteps = model.input_shape[1]
    if mfcc.shape[0] != expected_timesteps:
        # Pad or trim to match expected length
        if mfcc.shape[0] < expected_timesteps:
            pad = expected_timesteps - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
        else:
            mfcc = mfcc[:expected_timesteps]

    X = np.expand_dims(mfcc, axis=0)   # shape: (1, time_steps, n_mfcc)

    prob = model.predict(X, verbose=0)[0][0]   # probability of being FAKE

    if prob > 0.5:
        label      = "🚨 FAKE (Deepfake Detected)"
        confidence = prob * 100
    else:
        label      = "✅ REAL (Genuine Voice)"
        confidence = (1 - prob) * 100

    return label, confidence


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file_or_folder>")
        sys.exit(1)

    input_path = sys.argv[1]

    # ---- Load model ----
    if not os.path.exists(config.MODEL_PATH):
        print(f"[ERROR] Model not found at {config.MODEL_PATH}")
        print("  → Run: python train.py first")
        sys.exit(1)

    model = tf.keras.models.load_model(config.MODEL_PATH)
    print(f"✅ Model loaded: {config.MODEL_PATH}")
    print("=" * 50)

    # ---- Single file or folder ----
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith((".wav", ".flac", ".mp3"))
        ]
        print(f"Found {len(files)} audio files in: {input_path}\n")
    else:
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)

    # ---- Predict each file ----
    for file_path in files:
        filename = os.path.basename(file_path)
        label, confidence = predict_single(model, file_path)

        if label:
            print(f"  {filename}")
            print(f"    → {label}  ({confidence:.1f}% confident)\n")
        else:
            print(f"  {filename}  → [Could not process]\n")


if __name__ == "__main__":
    main()
