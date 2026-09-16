"""
train.py
--------
Loads the extracted features, trains the VoxGuard CNN-LSTM model,
and saves the best model to disk.

Run this after extract_features.py:
    python train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

import config
from model import build_model


def load_features():
    """Load the pre-extracted features from the .npz file."""
    if not os.path.exists(config.FEATURES_PATH):
        print("[ERROR] Features file not found!")
        print("  → Please run: python extract_features.py first")
        exit(1)

    data = np.load(config.FEATURES_PATH)
    X, y = data["X"], data["y"]
    print(f"✅ Loaded features: X={X.shape}, y={y.shape}")
    print(f"   Real samples : {np.sum(y == 0)}")
    print(f"   Fake samples : {np.sum(y == 1)}")
    return X, y


def plot_training_curves(history):
    """Save accuracy and loss plots to the results/ folder."""
    os.makedirs("results", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="royalblue")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="tomato")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"],     label="Train Loss", color="royalblue")
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="tomato")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.PLOT_PATH, dpi=150)
    plt.close()
    print(f"📊 Training curves saved to: {config.PLOT_PATH}")


def main():
    print("=" * 50)
    print("  VoxGuard — Training")
    print("=" * 50)

    # ---- Load & Shuffle Data ----
    X, y = load_features()
    X, y = shuffle(X, y, random_state=42)

    # ---- Train / Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y          # Keep class balance in both splits
    )

    print(f"\n   Train samples : {len(X_train)}")
    print(f"   Test  samples : {len(X_test)}")

    # ---- Build Model ----
    input_shape = X_train.shape[1:]   # (time_steps, n_mfcc)
    model = build_model(input_shape)
    model.summary()

    # ---- Callbacks ----
    callbacks = [
        # Stop early if validation loss doesn't improve for 5 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Save the best model during training
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate if stuck in a plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    # ---- Train ----
    print(f"\n🚀 Starting training for up to {config.EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # ---- Evaluate on Test Set ----
    print("\n📊 Evaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n   Test Loss     : {loss:.4f}")
    print(f"   Test Accuracy : {accuracy * 100:.2f}%")

    # ---- Save Training Curves ----
    plot_training_curves(history)

    print(f"\n✅ Training complete! Model saved to: {config.MODEL_PATH}")
    print("   Next step → run: python evaluate.py")


if __name__ == "__main__":
    main()
