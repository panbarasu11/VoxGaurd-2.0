"""
extract_features.py
-------------------
Reads audio files from data/real and data/fake folders,
extracts MFCC features using librosa with silence trimming and audio tile-repeat padding,
and saves them to a .npz file.
"""

import os
import sys
import numpy as np
import librosa
from tqdm import tqdm
import config

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass


def extract_mfcc(file_path):
    """
    Load an audio file, trim silence, pad/tile to fixed duration, and extract MFCC features.

    Returns:
        numpy array of shape (time_steps, N_MFCC) or None.
    """
    try:
        audio, sr = librosa.load(
            file_path,
            sr=config.SAMPLE_RATE,
            mono=True
        )

        # Trim unvoiced dead silence so mic recordings aren't filled with zero-padding
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
        if len(audio_trimmed) > 0:
            audio = audio_trimmed

        target_length = config.SAMPLE_RATE * config.DURATION
        if len(audio) < target_length:
            repeats = int(np.ceil(target_length / len(audio)))
            audio = np.tile(audio, repeats)[:target_length]
        else:
            audio = audio[:target_length]

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=config.N_MFCC,
            hop_length=config.HOP_LENGTH,
            n_fft=config.N_FFT
        )

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        feat = np.vstack([mfcc, delta, delta2])  # Shape: (60, time_steps)
        return feat.T  # Shape: (time_steps, 60)

    except Exception as e:
        print(f"  [Warning] Could not process {file_path}: {e}")
        return None


def load_all_files(folder, label):
    """Go through all audio files in a folder, extract features, and assign given label."""
    features = []
    labels   = []

    audio_files = [
        f for f in os.listdir(folder)
        if f.endswith((".wav", ".flac", ".mp3"))
    ]

    print(f"\nProcessing {len(audio_files)} files from: {folder}")

    for filename in tqdm(audio_files):
        file_path = os.path.join(folder, filename)
        mfcc = extract_mfcc(file_path)

        if mfcc is not None:
            features.append(mfcc)
            labels.append(label)

    return features, labels


def main():
    print("=" * 50)
    print("  VoxGuard — Feature Extraction (Silence Trimmed & Tile-Padded)")
    print("=" * 50)

    real_features, real_labels = load_all_files(
        config.REAL_AUDIO_DIR,
        label=config.LABEL_REAL
    )

    fake_features, fake_labels = load_all_files(
        config.FAKE_AUDIO_DIR,
        label=config.LABEL_FAKE
    )

    all_features = real_features + fake_features
    all_labels   = real_labels   + fake_labels

    if len(all_features) == 0:
        print("\n[ERROR] No audio files found! Check your data/ folder.")
        return

    min_len = min(f.shape[0] for f in all_features)
    all_features = [f[:min_len] for f in all_features]

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n[OK] Feature extraction complete!")
    print(f"   Total samples : {len(X)}")
    print(f"   Real samples  : {real_labels.count(config.LABEL_REAL)}")
    print(f"   Fake samples  : {fake_labels.count(config.LABEL_FAKE)}")
    print(f"   Feature shape : {X.shape}  (samples, time_steps, mfcc_coeffs)")

    os.makedirs(os.path.dirname(config.FEATURES_PATH), exist_ok=True)
    np.savez(config.FEATURES_PATH, X=X, y=y)
    print(f"\n[SAVE] Saved features to: {config.FEATURES_PATH}")


if __name__ == "__main__":
    main()
