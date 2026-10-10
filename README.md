# 🛡️ VoxGuard — Deepfake Voice Detection with Federated Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Federated_Learning-FedAvg-purple" />
  <img src="https://img.shields.io/badge/Privacy-Zero_Raw_Data_Transmission-green" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

> **VoxGuard** is a privacy-preserving audio deepfake detection system built on **Federated Learning (FedAvg)**. It detects AI-generated synthetic speech without ever transmitting raw audio data — only encrypted model weight gradients are shared across edge clients.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Federated Learning Pipeline](#federated-learning-pipeline)
- [Web Application](#web-application)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Model Details](#model-details)

---

## Overview

With the rapid rise of AI voice cloning tools (ElevenLabs, VALL-E, etc.), synthetic speech has become indistinguishable to the human ear. **VoxGuard** addresses this threat by:

- Classifying audio as **Genuine Human Voice** ✅ or **AI-Generated Deepfake** 🚨
- Training a **CNN-LSTM model across 5 simulated edge clients** using Federated Averaging (FedAvg)
- Ensuring **zero raw data transmission** — only model weight deltas are aggregated at the server
- Providing a **real-time web interface** with spectrogram visualization and explainable AI metrics

---

## Features

| Feature | Details |
|---|---|
| 🔊 Audio Formats | WAV, FLAC, MP3, OGG, WebM (browser mic) |
| 🤖 Model | Federated CNN-LSTM (Bidirectional) |
| 🔐 Privacy | FedAvg — gradient weight aggregation only |
| 📊 Explainability | MFCC spectrogram, ZCR, spectral centroid, RMS energy |
| 🌐 Interface | Flask web app with real-time prediction |
| 🧪 FL Simulation | IID & Non-IID data partitioning (Dirichlet) |
| 📈 Benchmarking | FL vs Centralized model accuracy curves |
| 🎙️ Live Recording | In-browser microphone recording support |

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         VoxGuard System Overview         │
                    └─────────────────────────────────────────┘

  Audio Input (WAV/FLAC/MP3/OGG)
         │
         ▼
  ┌─────────────────┐
  │  extract_mfcc() │  ← 60D MFCC (20 static + 20 Δ + 20 ΔΔ)
  │  shape: (126,60)│     Silence trimming, tile-repeat padding
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────┐
  │    VoxGuard CNN-LSTM Model      │
  │  LayerNorm → Conv1D(64) →       │
  │  BN → Pool → Dropout →          │
  │  Conv1D(128) → BN → Pool →      │
  │  BiLSTM(64) → Dense(64) →       │
  │  Sigmoid Output (0=Real,1=Fake) │
  └────────┬────────────────────────┘
           │
           ▼
  Prediction: P(FAKE) > 0.5 → 🚨 AI Deepfake
              P(FAKE) ≤ 0.5 → ✅ Genuine Voice


  ┌─────────────────────────────────────────┐
  │      Federated Learning (FedAvg)        │
  │                                         │
  │  Global Server ────────────────────┐    │
  │       ↓ distribute weights         │    │
  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │    │
  │  │ C0  │ │ C1  │ │ C2  │ │ C3  │  │    │
  │  │Edge │ │Smart│ │VoIP │ │Desk │  │    │
  │  │Phone│ │Spkr │ │Gate │ │ top │  │    │
  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘  │    │
  │     └───────┴───────┴───────┘      │    │
  │         ↑ Upload encrypted         │    │
  │           weight gradients only    │    │
  │  ← FedAvg Aggregation ─────────────┘    │
  └─────────────────────────────────────────┘
```

---

## Project Structure

```
VoxGuard-deepfake-voice-detection-fl-main/
│
├── app.py                    # Flask web server (REST API + UI backend)
├── model.py                  # CNN-LSTM model architecture definition
├── train.py                  # Centralized baseline training script
├── federated_train.py        # Federated Learning (FedAvg) training pipeline
├── extract_features.py       # 60D MFCC feature extraction (NumPy/SciPy, Numba-free)
├── evaluate.py               # Model evaluation and metrics reporting
├── predict.py                # Standalone CLI prediction script
├── fl_utils.py               # FL utilities: FedAvg, data partitioning, normalization
├── generate_sample_data.py   # Generate synthetic sample audio for testing
│
├── config.py                 # Global audio & model hyperparameters
├── federated_config.py       # FL-specific hyperparameters (clients, rounds, etc.)
│
├── data/
│   ├── real/                 # Genuine human voice audio files (.wav/.flac/.mp3)
│   ├── fake/                 # AI-generated / spoofed audio files
│   └── features.npz          # Pre-extracted MFCC features (auto-generated)
│
├── results/
│   ├── training_curves.png          # Centralized training accuracy/loss curves
│   ├── fl_vs_centralized_curves.png # FL vs Centralized comparison plots
│   └── fl_metrics.json              # Federated training metrics summary (JSON)
│
├── templates/                # Jinja2 HTML templates for the Flask web UI
├── static/                   # CSS, JS, and frontend assets
├── temp_uploads/             # Temporary storage for uploaded audio (auto-cleaned)
│
├── voxguard_model.h5         # Saved centralized baseline model
├── voxguard_fl_model.h5      # Saved federated learning model (used for inference)
└── requirements.txt          # Python dependencies
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/VoxGuard-deepfake-voice-detection-fl.git
cd VoxGuard-deepfake-voice-detection-fl-main
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Requirements:** `numpy`, `librosa`, `tensorflow>=2.12`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `soundfile`, `flask`

### 3. Prepare Your Dataset

Place audio files in:
- `data/real/` — genuine human speech (`.wav`, `.flac`, `.mp3`)
- `data/fake/` — AI-generated/deepfake speech

> **Recommended datasets:** [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336), [FakeAVCeleb](https://github.com/shahroztariq/FakeAVCeleb), [WaveFake](https://github.com/RUB-SysSec/WaveFake)

Or generate sample data for testing:
```bash
python generate_sample_data.py
```

### 4. Extract MFCC Features

```bash
python extract_features.py
```

Outputs `data/features.npz` with shape `(N_samples, 126, 60)`.

### 5. Train the Federated Model

```bash
# Default: 5 clients, 10 FL rounds, IID partitioning
python federated_train.py

# Custom options
python federated_train.py --num-clients 8 --num-rounds 15 --non-iid
```

### 6. Launch the Web Application

```bash
python app.py
```

Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## Federated Learning Pipeline

VoxGuard simulates a real-world FL system on a single machine. The pipeline follows these steps:

### Step-by-Step

| Step | Script | Description |
|------|--------|-------------|
| 1 | `extract_features.py` | Extract 60D MFCC features from audio dataset |
| 2 | `federated_train.py` | Run FedAvg training across N simulated clients |
| 3 | `app.py` | Serve the trained FL model via Flask API |

### FedAvg Algorithm

```
For each communication round r = 1 ... R:
  1. Server broadcasts global model weights W_r to all clients
  2. Each client k trains locally for E epochs on its private data D_k
  3. Each client uploads trained weights W_k (NO raw data transmitted)
  4. Server aggregates: W_{r+1} = Σ(|D_k| / |D|) × W_k  [weighted average]
  5. Server evaluates global model on held-out test set
```

### Data Partitioning Modes

| Mode | Flag | Description |
|------|------|-------------|
| **IID** | *(default)* | Random equal split across clients (uniform distribution) |
| **Non-IID** | `--non-iid` | Dirichlet distribution (α=0.5) — simulates real-world data skew |

### FL Configuration (`federated_config.py`)

```python
NUM_CLIENTS     = 5      # Edge clients simulated
NUM_ROUNDS      = 10     # Global communication rounds
LOCAL_EPOCHS    = 2      # Local training epochs per client per round
CLIENT_FRACTION = 1.0    # All clients participate each round
BATCH_SIZE      = 64     # Local client batch size
IID_SPLIT       = True   # IID or Non-IID data split
DIRICHLET_ALPHA = 0.5    # Non-IID heterogeneity (lower = more skewed)
```

---

## Web Application

The Flask web app provides:

- 📁 **Upload Mode** — Upload any audio file for instant detection
- 🎙️ **Live Recording** — Record directly from browser microphone
- 🔊 **Preset Samples** — Test with pre-loaded real and fake audio samples
- 📊 **MFCC Spectrogram** — Visual heatmap of audio features (Magma colormap)
- 🧠 **Acoustic Stats** — Duration, sample rate, RMS energy, ZCR, spectral centroid
- 📡 **FL Dashboard** — View connected edge clients, FL metrics, and simulate rounds

### Running the Server

```bash
python app.py
# Server starts at http://127.0.0.1:5000
# Override port: PORT=8080 python app.py
```

---

## Configuration

### `config.py` — Audio & Model Settings

```python
SAMPLE_RATE   = 16000    # 16 kHz speech processing standard
DURATION      = 4        # Clip all audio to 4 seconds
N_MFCC        = 20       # Static MFCC coefficients
N_FEAT        = 60       # 20 MFCC + 20 Δ (velocity) + 20 ΔΔ (acceleration)
HOP_LENGTH    = 512      # FFT hop length between frames
N_FFT         = 1024     # FFT window size

LSTM_UNITS    = 64
DROPOUT_RATE  = 0.3
LEARNING_RATE = 0.001
EPOCHS        = 10
BATCH_SIZE    = 32
```

---

## API Reference

All endpoints return JSON responses.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve web UI (index.html) |
| `GET` | `/api/presets` | List available preset real/fake audio samples |
| `POST` | `/api/predict_upload` | Upload audio file → get deepfake prediction |
| `POST` | `/api/predict_preset` | Predict on a preset sample by filename |
| `GET` | `/api/fl_metrics` | Get FL training metrics & client node info |
| `POST` | `/api/simulate_fl_round` | Simulate a FL round and get live updates |
| `POST` | `/api/reload_model` | Hot-reload the FL model from disk |
| `GET` | `/api/sample/<category>/<filename>` | Stream a preset audio file |

### Example: Upload Prediction Response

```json
{
  "success": true,
  "filename": "sample_voice.wav",
  "is_fake": true,
  "label": "🚨 AI-Generated Deepfake",
  "confidence": 87.3,
  "prob_fake": 87.3,
  "prob_real": 12.7,
  "spectrogram": "<base64 PNG>",
  "acoustic_stats": {
    "duration": 3.84,
    "sample_rate": 16000,
    "rms_energy": 0.0412,
    "zero_crossing_rate": 0.0631,
    "spectral_centroid": 1843.2,
    "mfcc_timesteps": 126,
    "mfcc_coefficients": 60,
    "mfcc_variance": 14.2,
    "mfcc_mean": -2.31
  },
  "fl_details": {
    "model_type": "Federated CNN-LSTM (FedAvg)",
    "edge_clients": 5,
    "privacy_mode": "Zero Raw Data Transmission (Gradient Weight Aggregation Only)",
    "fake_probability": 0.873,
    "real_probability": 0.127
  }
}
```

---

## Model Details

### Architecture: `VoxGuard_FedCNN_LSTM`

```
Input:  (126, 60)        — 126 time frames × 60 MFCC features

LayerNormalization        — per-coefficient feature normalization

Conv1D(64, kernel=5)      — local spectral artifact detection
BatchNormalization
MaxPooling1D(2)
Dropout(0.3)

Conv1D(128, kernel=3)     — higher-level spectral pattern extraction
BatchNormalization
MaxPooling1D(2)
Dropout(0.3)

Bidirectional LSTM(64)    — temporal speech dynamics modeling
Dropout(0.3)

Dense(64, relu)           — classification head
Dropout(0.2)
Dense(1, sigmoid)         — output: P(FAKE) ∈ [0, 1]
```

### Feature Extraction: 60D MFCC

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Static MFCC | 20 | Raw mel-frequency cepstral coefficients |
| Delta (Δ) | 20 | 1st-order temporal velocity |
| Delta-Delta (ΔΔ) | 20 | 2nd-order temporal acceleration |
| **Total** | **60** | **Stacked feature vector per time frame** |

- **Silence trimming**: Energy threshold at 0.02 (peak-normalized)
- **Padding**: Tile-repeat padding to reach fixed 4-second (64,000 sample) duration
- **Normalization**: Per-coefficient Z-score normalization using training statistics (`data/norm_stats.npz`)
- **Numba-free**: Custom SciPy/NumPy MFCC engine — no Numba/DLL dependency issues on Windows

---

## Privacy & Security

VoxGuard is designed with **privacy-by-design** principles:

- ✅ **Raw audio never leaves client devices** — only encrypted model weights are transmitted
- ✅ **FedAvg aggregation** — server only sees weighted averages of client weight updates
- ✅ **No user identification** — edge clients are anonymous (`Edge Device #0–4`)
- ✅ **Local inference** — all audio processing happens on the server-side (not cloud APIs)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ | VoxGuard — Protecting Voice Integrity with Federated AI
</p>