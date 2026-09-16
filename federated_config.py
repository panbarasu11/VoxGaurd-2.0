# ============================================================
# VoxGuard — Federated Learning Configuration
# FL-specific hyperparameter settings and configuration
# ============================================================

import config

# --- Federated Learning Hyperparameters ---
NUM_CLIENTS      = 5      # Total number of simulated edge clients
NUM_ROUNDS       = 10     # Number of global communication/aggregation rounds
LOCAL_EPOCHS     = 2      # Number of local training epochs per client per round
CLIENT_FRACTION  = 1.0    # Fraction of clients selected per round (1.0 = all clients participate)
BATCH_SIZE       = 64     # Batch size for local client training (64 speeds up CPU training)

# --- Data Distribution Settings ---
# True  = IID (Independent and Identically Distributed): random equal split across clients
# False = Non-IID: heterogeneous class distribution across clients (simulates real-world user data skew)
IID_SPLIT       = True

# Dirichlet distribution concentration parameter for Non-IID split (smaller alpha = higher data heterogeneity/skew)
DIRICHLET_ALPHA = 0.5

# --- Output & Artifact Paths ---
FL_MODEL_PATH   = "voxguard_fl_model.h5"
FL_PLOT_PATH    = "results/fl_vs_centralized_curves.png"
FL_METRICS_PATH = "results/fl_metrics.json"

# --- Re-export standard config parameters for convenience ---
FEATURES_PATH    = config.FEATURES_PATH
LABEL_REAL       = config.LABEL_REAL
LABEL_FAKE       = config.LABEL_FAKE
LEARNING_RATE    = config.LEARNING_RATE
