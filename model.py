"""
model.py
--------
Defines the VoxGuard CNN-LSTM model architecture for Audio Deepfake Detection.

Architecture:
- 1D Convolutional Blocks: Scans MFCC spectrogram frames to extract spectral artifacts.
- Batch Normalization & Dropout: Accelerates convergence and prevents overfitting.
- Bidirectional LSTM / LSTM: Captures temporal speech dynamics and voice continuity over time.
- Dense Output Layer: Binary Sigmoid output representing AI Fake Probability (>0.5 = FAKE, <0.5 = REAL).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import config


def build_model(input_shape=(126, config.N_FEAT)):
    """
    Build the high-accuracy VoxGuard CNN-LSTM model.

    Args:
        input_shape: tuple (time_steps, n_features), default (126, 60)

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential(name="VoxGuard_FedCNN_LSTM")

    # --- Input & Feature Normalization ---
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LayerNormalization(axis=-1))

    # --- Conv Block 1 ---
    model.add(layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu"
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(config.DROPOUT_RATE))

    # --- Conv Block 2 ---
    model.add(layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(config.DROPOUT_RATE))

    # --- Temporal Sequence Modeling (LSTM) ---
    model.add(layers.Bidirectional(layers.LSTM(
        units=config.LSTM_UNITS,
        return_sequences=False
    )))
    model.add(layers.Dropout(config.DROPOUT_RATE))

    # --- Classification Dense Output ---
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))

    # --- Compile ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


if __name__ == "__main__":
    dummy_shape = (126, config.N_FEAT)
    model = build_model(dummy_shape)
    model.summary()
