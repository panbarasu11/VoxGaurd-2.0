"""
evaluate.py
-----------
Loads the trained model and test features,
prints classification report and saves a confusion matrix.

Run after training:
    python evaluate.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import config

if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def shuffle(X, y, seed=42, random_state=42):
    s = seed if seed is not None else random_state
    np.random.seed(s)
    p = np.random.permutation(len(X))
    return X[p], y[p]

def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    np.random.seed(random_state)
    n = len(X)
    p = np.random.permutation(n)
    split_idx = max(1, int(n * (1 - test_size)))
    X_shuffled, y_shuffled = X[p], y[p]
    return X_shuffled[:split_idx], X_shuffled[split_idx:], y_shuffled[:split_idx], y_shuffled[split_idx:]

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm

def roc_auc_score(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    tprs, fprs = [], []
    for th in thresholds:
        yp = (y_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        fn = np.sum((y_true == 1) & (yp == 0))
        tn = np.sum((y_true == 0) & (yp == 0))
        tprs.append(tp / (tp + fn + 1e-8))
        fprs.append(fp / (fp + tn + 1e-8))
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(np.abs(trapz_func(tprs, fprs)))

def classification_report(y_true, y_pred, target_names=["Real", "Fake"]):
    cm = confusion_matrix(y_true, y_pred)
    res = "              precision    recall  f1-score   support\n\n"
    for i, name in enumerate(target_names):
        tp = cm[i][i]
        fp = cm[1-i][i]
        fn = cm[1-i][i]
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        sup = np.sum(y_true == i)
        res += f"{name:>12}       {prec:.2f}      {rec:.2f}      {f1:.2f}        {sup}\n"
    return res

def plot_confusion_matrix(cm):
    """Save a nicely styled confusion matrix using matplotlib."""
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("VoxGuard — Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Real", "Fake"])
    plt.yticks(tick_marks, ["Real", "Fake"])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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
