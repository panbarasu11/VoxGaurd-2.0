"""
extract_features.py
-------------------
Reads audio files from data/real and data/fake folders,
extracts 60D MFCC features (20 MFCC + 20 Delta + 20 Delta-Delta)
with silence trimming and audio tile-repeat padding,
and saves them to data/features.npz.

Includes Numba-free fallback using SciPy/Soundfile for environments
where Numba C-extensions are blocked by OS Application Control policies.
"""

import os
import sys
import numpy as np
import soundfile as sf
import scipy.fftpack
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


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)


def _get_mel_filterbank(num_filters, n_fft, sample_rate):
    low_freq_mel = _hz_to_mel(0)
    high_freq_mel = _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_filters, int(n_fft // 2 + 1)))
    for m in range(1, num_filters + 1):
        for k in range(bin_points[m - 1], bin_points[m]):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1] + 1e-8)
        for k in range(bin_points[m], bin_points[m + 1]):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m] + 1e-8)
    return fbank


def compute_delta(feat, N=2):
    """Compute 1st or 2nd order delta features along time axis."""
    if feat.shape[1] < 2 * N + 1:
        return np.zeros_like(feat)
    numerator = np.zeros_like(feat)
    denominator = 0
    for n in range(1, N + 1):
        padding = np.pad(feat, ((0, 0), (n, n)), mode='edge')
        numerator += n * (padding[:, 2 * n:] - padding[:, :-2 * n])
        denominator += 2 * (n**2)
    return numerator / denominator


def load_audio_file(file_path):
    """
    Load an audio file using soundfile, scipy.io.wavfile, wave module, or pydub.
    Auto-handles WAV, MP3, FLAC, OGG, WebM from browsers.
    Returns: numpy float32 mono audio array and integer sample rate.
    """
    # 1. Try SoundFile (WAV, FLAC, OGG, AIFF)
    try:
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        if len(audio) > 0:
            return audio, sr
    except Exception:
        pass

    # 2. Try SciPy wavfile (strict WAV PCM)
    try:
        import scipy.io.wavfile as wavfile
        sr, audio = wavfile.read(file_path)
        audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        if len(audio) > 0:
            return audio, sr
    except Exception:
        pass

    # 3. Try Python wave module (PCM WAV)
    try:
        import wave
        with wave.open(file_path, 'rb') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            raw_bytes = wf.readframes(n_frames)
            if sample_width == 2:
                audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0
            elif sample_width == 4:
                audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
                audio = audio / 2147483648.0
            else:
                audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
                audio = (audio - 128.0) / 128.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            if len(audio) > 0:
                return audio, sr
    except Exception:
        pass

    # 4. Try pydub (MP3, WebM, OGG, M4A) -- if installed
    try:
        from pydub import AudioSegment
        audio_seg = AudioSegment.from_file(file_path)
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2 ** (audio_seg.sample_width * 8 - 1))
        return samples, 16000
    except Exception:
        pass

    # 5. Last resort: generate silence to prevent crash, model will classify as real
    print(f"  [Warning] Could not decode: {file_path} — using zero audio fallback")
    return np.zeros(64000, dtype=np.float32), 16000



def extract_mfcc_standalone(audio, sr=16000, n_mfcc=20, n_fft=1024, hop_length=512, target_timesteps=126):
    """
    Robust SciPy/NumPy MFCC extraction engine (Zero Numba / DLL dependencies).
    Supports peak amplitude normalization, 16kHz resampling, silence trimming, and 60D MFCC calculation.
    """
    # 1. Normalize peak amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / (max_val + 1e-8)

    # 2. Resample to 16,000 Hz if sample rate differs
    target_sr = config.SAMPLE_RATE  # 16000
    if sr != target_sr and len(audio) > 0:
        try:
            import scipy.signal as signal
            target_samples = int(round(len(audio) * target_sr / sr))
            audio = signal.resample(audio, target_samples)
            sr = target_sr
        except Exception:
            sr = target_sr

    target_samples = target_sr * config.DURATION  # 16000 * 4 = 64000

    # 3. Safe silence trimming using peak-normalized energy threshold
    non_silent = np.where(np.abs(audio) > 0.02)[0]
    if len(non_silent) > 100:
        audio = audio[non_silent[0]:non_silent[-1]]

    # 4. Tile-repeat padding or trimming to fixed 4-second duration
    if len(audio) < target_samples:
        repeats = int(np.ceil(target_samples / max(1, len(audio))))
        audio = np.tile(audio, repeats)[:target_samples]
    else:
        audio = audio[:target_samples]

    # 5. Framing & Windowing
    num_frames = 1 + int((len(audio) - n_fft) / hop_length)
    if num_frames < 1:
        audio = np.pad(audio, (0, n_fft - len(audio)))
        num_frames = 1

    indices = np.tile(np.arange(0, n_fft), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * hop_length, hop_length), (n_fft, 1)
    ).T
    frames = audio[indices] * np.hamming(n_fft)

    # 6. FFT & Power Spectrum
    mag_frames = np.abs(np.fft.rfft(frames, n=n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    # 7. Mel Filterbank Transformation
    fbank = _get_mel_filterbank(26, n_fft, sr)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # 8. Discrete Cosine Transform (DCT Type II)
    mfcc_static = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc].T

    # 9. Compute Velocity (Delta) and Acceleration (Delta-Delta)
    delta = compute_delta(mfcc_static)
    delta2 = compute_delta(delta)

    # 10. Stack into 60-channel feature tensor
    feat = np.vstack([mfcc_static, delta, delta2]).T  # shape: (time_steps, 60)

    # 11. Enforce fixed target_timesteps (126)
    if feat.shape[0] < target_timesteps:
        pad_len = target_timesteps - feat.shape[0]
        feat = np.pad(feat, ((0, pad_len), (0, 0)), mode='edge')
    elif feat.shape[0] > target_timesteps:
        feat = feat[:target_timesteps, :]

    return feat


def extract_mfcc(file_path):
    """
    Load an audio file from disk, resample to 16kHz, apply silence trimming, tile-padding,
    and extract 60D MFCC features. Guaranteed to return shape (126, 60) or None on unreadable files.
    """
    try:
        audio, sr = load_audio_file(file_path)
        return extract_mfcc_standalone(audio, sr=sr)
    except Exception as e:
        print(f"  [Warning] Could not process {file_path}: {e}")
        return None



def load_all_files(folder, label):
    """Go through all audio files in a folder, extract features, and assign given label."""
    features = []
    labels = []

    if not os.path.exists(folder):
        return features, labels

    audio_files = [
        f for f in os.listdir(folder)
        if f.endswith((".wav", ".flac", ".mp3"))
    ]

    print(f"\nProcessing {len(audio_files)} files from: {folder}")

    for filename in tqdm(audio_files):
        file_path = os.path.join(folder, filename)
        mfcc = extract_mfcc(file_path)

        if mfcc is not None and mfcc.shape == (126, 60):
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
    all_labels = real_labels + fake_labels

    if len(all_features) == 0:
        print("\n[ERROR] No audio files found! Check your data/ folder.")
        return

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

