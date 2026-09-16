# ============================================================
# VoxGuard — Configuration
# ============================================================

# --- Audio Settings ---
SAMPLE_RATE = 16000        # Standard sample rate for speech (16kHz)
DURATION    = 4            # Clip all audio to 4 seconds
N_MFCC      = 20           # Static MFCC coefficients
N_FEAT      = 60           # 20 MFCC + 20 Delta + 20 Delta-Delta = 60 total acoustic features
HOP_LENGTH  = 512          # Frames between each MFCC window
N_FFT       = 1024         # FFT window size

# --- Dataset Paths ---
REAL_AUDIO_DIR  = "data/real"
FAKE_AUDIO_DIR  = "data/fake"
FEATURES_PATH   = "data/features.npz"

# --- Model Settings ---
CNN_FILTERS     = [64, 128]
LSTM_UNITS      = 64
DROPOUT_RATE    = 0.3

# --- Training Settings ---
EPOCHS          = 10
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
VALIDATION_SPLIT = 0.2

# --- Save Paths ---
MODEL_PATH      = "voxguard_model.h5"
PLOT_PATH       = "results/training_curves.png"
CM_PATH         = "results/confusion_matrix.png"

# --- Labels ---
LABEL_REAL = 0    # Genuine/real voice
LABEL_FAKE = 1    # Fake/spoofed voice
