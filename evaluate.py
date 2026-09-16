"""
evaluate.py
-----------
Loads the trained model and test features,
prints classification report and saves a confusion matrix.

Run after training:
    python evaluate.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.utils import shuffle
import tensorflow as tf

import config


def plot_confusion_matrix(cm):
    """Save a nicely styled confusion matrix."""
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        linewidths=0.5
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("VoxGuard — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(config.CM_PATH, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved to: {config.CM_PATH}")


def main():
    print("=" * 50)
    print("  VoxGuard — Evaluation")
    print("=" * 50)

    # ---- Load features ----
    if not os.path.exists(config.FEATURES_PATH):
        print("[ERROR] Run extract_features.py first!")
        exit(1)

    data = np.load(config.FEATURES_PATH)
    X, y = data["X"], data["y"]
    X, y = shuffle(X, y, random_state=42)

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---- Load trained model ----
    if not os.path.exists(config.MODEL_PATH):
        print("[ERROR] No trained model found! Run train.py first.")
        exit(1)

    model = tf.keras.models.load_model(config.MODEL_PATH)
    print(f"✅ Loaded model from: {config.MODEL_PATH}")

    # ---- Predict ----
    y_prob = model.predict(X_test).flatten()   # probabilities (0 to 1)
    y_pred = (y_prob > 0.5).astype(int)        # threshold at 0.5

    # ---- Metrics ----
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    auc = roc_auc_score(y_test, y_prob)
    print(f"   ROC-AUC Score : {auc:.4f}")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    print("\n   Confusion Matrix:")
    print(f"   True Real → predicted Real : {cm[0][0]}")
    print(f"   True Real → predicted Fake : {cm[0][1]}  (False Positives)")
    print(f"   True Fake → predicted Real : {cm[1][0]}  (False Negatives)")
    print(f"   True Fake → predicted Fake : {cm[1][1]}")

    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()
