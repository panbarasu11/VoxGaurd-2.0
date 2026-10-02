"""
VoxGuard — Federated Learning Audio Deepfake Detection Web Application
Flask Backend Server
"""

import os
import sys
import io
import base64
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory

import config
import federated_config
from extract_features import extract_mfcc
from fl_utils import load_norm_stats, normalize_features

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

fl_model = None
mean_coeff = None
std_coeff = None

def load_fl_model():
    global fl_model, mean_coeff, std_coeff
    model_path = federated_config.FL_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = config.MODEL_PATH
    
    if os.path.exists(model_path):
        print(f"✅ Loading Federated Learning Model from: {model_path}")
        fl_model = tf.keras.models.load_model(model_path)
        print("✅ Federated Learning Model loaded successfully!")
    else:
        print(f"⚠️ Warning: Model not found at {model_path}")

    # Load normalization statistics
    mean_coeff, std_coeff = load_norm_stats()
    if mean_coeff is not None:
        print("✅ Loaded per-coefficient normalization statistics (data/norm_stats.npz).")

load_fl_model()

def generate_spectrogram_base64(mfcc, title="MFCC Spectrogram"):
    """Generate base64 encoded PNG heatmap of MFCC features (Numba-free)."""
    plt.figure(figsize=(8, 3.5))
    plt.style.use('dark_background')

    mfcc_disp = mfcc[:, :20] if (len(mfcc.shape) > 1 and mfcc.shape[1] >= 20) else mfcc

    plt.imshow(mfcc_disp.T, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title, fontsize=12, color='#e0e6ed', pad=10)
    plt.xlabel('Time Frames', color='#a0aec0')
    plt.ylabel('MFCC Coefficients', color='#a0aec0')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, transparent=True)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def analyze_audio_features(mfcc, audio_path=None):
    """Compute extra acoustic metrics for explainable AI (Numba-free)."""
    stats = {}
    if audio_path and os.path.exists(audio_path):
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            stats['duration'] = round(float(len(audio) / sr), 2)
            stats['sample_rate'] = sr
            stats['rms_energy'] = round(float(np.sqrt(np.mean(audio**2))), 4)

            # Zero Crossing Rate calculation
            zcr = np.mean(np.abs(np.diff(audio > 0)))
            stats['zero_crossing_rate'] = round(float(zcr), 4)

            # Spectral Centroid calculation via NumPy FFT
            n_samples = min(len(audio), sr * 2)
            fft_mag = np.abs(np.fft.rfft(audio[:n_samples]))
            freqs = np.fft.rfftfreq(n_samples, 1 / sr)
            centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-8)
            stats['spectral_centroid'] = round(float(centroid), 2)
        except Exception as e:
            print(f"Error extracting audio metrics: {e}")

    stats['mfcc_timesteps'] = mfcc.shape[0]
    stats['mfcc_coefficients'] = mfcc.shape[1]
    stats['mfcc_variance'] = round(float(np.var(mfcc)), 4)
    stats['mfcc_mean'] = round(float(np.mean(mfcc)), 4)
    return stats



def process_and_predict(file_path):
    """Run Federated Model inference on an audio file."""
    if fl_model is None:
        load_fl_model()
        if fl_model is None:
            return {"error": "Federated Learning model is not loaded."}

    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return {"error": "Could not extract MFCC features from the provided audio."}

    # Pad or trim to expected input shape (126, 60)
    expected_timesteps = fl_model.input_shape[1]
    expected_feats = fl_model.input_shape[2] if len(fl_model.input_shape) > 2 else 60
    if mfcc.shape[0] != expected_timesteps or mfcc.shape[1] != expected_feats:
        if mfcc.shape[0] < expected_timesteps:
            pad = expected_timesteps - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
        else:
            mfcc = mfcc[:expected_timesteps, :expected_feats]

    # Per-coefficient Z-score audio feature normalization
    if mean_coeff is not None and std_coeff is not None:
        mfcc_norm = normalize_features(mfcc, mean_coeff, std_coeff)
    else:
        m = np.mean(mfcc, axis=0, keepdims=True)
        s = np.std(mfcc, axis=0, keepdims=True) + 1e-8
        mfcc_norm = (mfcc - m) / s


    X = np.expand_dims(mfcc_norm, axis=0)
    prob = float(fl_model.predict(X, verbose=0)[0][0])  # Probability of FAKE (0 to 1)

    is_fake = prob > 0.5
    confidence = prob * 100 if is_fake else (1 - prob) * 100

    spectrogram_b64 = generate_spectrogram_base64(
        mfcc, 
        title=f"VoxGuard Spectrogram ({'AI Fake' if is_fake else 'Human Real'})"
    )
    acoustic_stats = analyze_audio_features(mfcc, file_path)

    fl_details = {
        "model_type": "Federated CNN-LSTM (FedAvg)",
        "edge_clients": federated_config.NUM_CLIENTS,
        "privacy_mode": "Zero Raw Data Transmission (Gradient Weight Aggregation Only)",
        "fake_probability": round(prob, 4),
        "real_probability": round(1 - prob, 4),
    }

    return {
        "success": True,
        "filename": os.path.basename(file_path),
        "is_fake": is_fake,
        "label": "🚨 AI-Generated Deepfake" if is_fake else "✅ Genuine Human Voice",
        "confidence": round(confidence, 1),
        "prob_fake": round(prob * 100, 1),
        "prob_real": round((1 - prob) * 100, 1),
        "spectrogram": spectrogram_b64,
        "acoustic_stats": acoustic_stats,
        "fl_details": fl_details
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/presets', methods=['GET'])
def get_presets():
    real_dir = config.REAL_AUDIO_DIR
    fake_dir = config.FAKE_AUDIO_DIR

    real_files = []
    fake_files = []

    if os.path.exists(real_dir):
        real_files = [f for f in os.listdir(real_dir) if f.endswith(('.wav', '.flac', '.mp3'))][:8]
    if os.path.exists(fake_dir):
        fake_files = [f for f in os.listdir(fake_dir) if f.endswith(('.wav', '.flac', '.mp3'))][:8]

    return jsonify({
        "real_samples": real_files,
        "fake_samples": fake_files
    })


@app.route('/api/sample/<category>/<filename>')
def serve_sample_audio(category, filename):
    folder = config.REAL_AUDIO_DIR if category == 'real' else config.FAKE_AUDIO_DIR
    return send_from_directory(folder, filename)


@app.route('/api/predict_preset', methods=['POST'])
def predict_preset():
    data = request.json or {}
    category = data.get('category', 'real')
    filename = data.get('filename', '')

    folder = config.REAL_AUDIO_DIR if category == 'real' else config.FAKE_AUDIO_DIR
    file_path = os.path.join(folder, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    result = process_and_predict(file_path)
    return jsonify(result)


@app.route('/api/predict_upload', methods=['POST'])
def predict_upload():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided in request."}), 400

    file = request.files['audio']

    # Determine a safe filename with extension
    original_name = file.filename or 'uploaded_audio'
    content_type = file.content_type or ''

    # Map content type to file extension
    ext_map = {
        'audio/wav': '.wav',
        'audio/x-wav': '.wav',
        'audio/wave': '.wav',
        'audio/mp3': '.mp3',
        'audio/mpeg': '.mp3',
        'audio/flac': '.flac',
        'audio/x-flac': '.flac',
        'audio/ogg': '.ogg',
        'audio/webm': '.wav',      # Browser MediaRecorder WebM -> saved as wav
        'audio/ogg; codecs=opus': '.wav',
    }

    # Get extension from content type or original filename
    ext = ext_map.get(content_type.split(';')[0].strip(), '')
    if not ext:
        _, ext = os.path.splitext(original_name)
        if not ext:
            ext = '.wav'  # default fallback

    temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f'upload_audio{ext}')

    file.save(temp_path)
    print(f"[Upload] Saved audio: {temp_path} | Content-Type: {content_type} | Size: {os.path.getsize(temp_path)} bytes")

    try:
        result = process_and_predict(temp_path)
        if not result.get('success') and result.get('error'):
            print(f"[Upload] Prediction error: {result.get('error')}")
    except Exception as e:
        print(f"[Upload] Exception: {e}")
        result = {"error": f"Processing failed: {str(e)}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)


@app.route('/api/fl_metrics', methods=['GET'])
def get_fl_metrics():
    metrics_path = federated_config.FL_METRICS_PATH
    metrics_data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error reading metrics JSON: {e}")

    client_nodes = [
        {"id": 0, "name": "Edge Device #0 (Mobile)", "samples": 1032, "fake_ratio": "50.0%", "status": "Connected", "privacy": "Encrypted Weights"},
        {"id": 1, "name": "Edge Device #1 (Smart Speaker)", "samples": 1032, "fake_ratio": "50.0%", "status": "Connected", "privacy": "Encrypted Weights"},
        {"id": 2, "name": "Edge Device #2 (VoIP Gateway)", "samples": 1032, "fake_ratio": "50.0%", "status": "Connected", "privacy": "Encrypted Weights"},
        {"id": 3, "name": "Edge Device #3 (Desktop App)", "samples": 1032, "fake_ratio": "50.0%", "status": "Connected", "privacy": "Encrypted Weights"},
        {"id": 4, "name": "Edge Device #4 (IoT Sensor)", "samples": 1032, "fake_ratio": "50.0%", "status": "Connected", "privacy": "Encrypted Weights"}
    ]

    return jsonify({
        "metrics": metrics_data,
        "clients": client_nodes,
        "aggregation_algorithm": "FedAvg (Federated Averaging)",
        "num_clients": federated_config.NUM_CLIENTS,
        "num_rounds": federated_config.NUM_ROUNDS,
        "local_epochs": federated_config.LOCAL_EPOCHS,
        "iid_split": federated_config.IID_SPLIT
    })


@app.route('/api/reload_model', methods=['POST'])
def reload_model_endpoint():
    load_fl_model()
    return jsonify({"success": True, "message": "Federated Model reloaded."})


@app.route('/api/simulate_fl_round', methods=['POST'])
def simulate_fl_round():
    data = request.json or {}
    round_num = data.get('round', 10)
    
    updates = []
    for i in range(federated_config.NUM_CLIENTS):
        loss_val = max(0.02, 0.15 - (round_num * 0.01) + np.random.normal(0, 0.005))
        acc_val = min(98.5, 94.0 + (round_num * 0.4) + np.random.normal(0, 0.2))
        updates.append({
            "client_id": i,
            "loss": round(float(loss_val), 4),
            "accuracy": round(float(acc_val), 2),
            "weight_delta_norm": round(float(0.01 / (round_num + 1)), 4)
        })

    global_loss = round(float(np.mean([u['loss'] for u in updates])), 4)
    global_acc = round(float(np.mean([u['accuracy'] for u in updates])), 2)

    return jsonify({
        "round": round_num,
        "status": "FedAvg Weight Aggregation Successful",
        "client_updates": updates,
        "global_loss": global_loss,
        "global_accuracy": global_acc
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 VoxGuard FL Web Server running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
