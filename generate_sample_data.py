"""
generate_sample_data.py
-----------------------
Generates realistic benchmark audio data for data/real (Genuine Human Speech)
and data/fake (AI Deepfake Voice Synthesis).

Simulates vocal tract formant dynamics for real speech vs. neural vocoder
phase jitter and spectral artifacts for AI deepfake voice detection.
"""

import sys
import os
import numpy as np
import soundfile as sf
import config

if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def generate_real_voice(sample_id, duration=4.0, sr=16000):
    """
    Simulate natural human speech with vocal tract formant resonances,
    prosodic fundamental frequency (F0) pitch sweeps, and organic breath noise.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    np.random.seed(100 + sample_id)

    # Base pitch (F0) with prosodic pitch modulation (110 Hz - 210 Hz human range)
    f0 = 120 + (sample_id % 15) * 6
    vibrato = 3.0 * np.sin(2 * np.pi * 5 * t)  # 5 Hz natural pitch vibrato
    pitch_contour = f0 + vibrato + 15.0 * np.sin(2 * np.pi * 0.8 * t)

    # Formant resonances (F1 ~ 500 Hz, F2 ~ 1500 Hz, F3 ~ 2500 Hz)
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
    f1 = 500 + 50 * np.sin(2 * np.pi * 1.2 * t)
    f2 = 1500 + 100 * np.cos(2 * np.pi * 0.9 * t)
    f3 = 2500 + 150 * np.sin(2 * np.pi * 0.5 * t)

    # Harmonic acoustic synthesis
    glottal_source = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    formant_filter = (
        0.5 * np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.15 * np.sin(2 * np.pi * f3 * t)
    )
    speech = glottal_source * (1.0 + 0.4 * formant_filter)

    # Natural speech syllabic envelope (cadence of spoken words)
    syllable_env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
    syllable_env = np.clip(syllable_env, 0.05, 1.0)
    speech *= syllable_env

    # Add subtle organic background room ambience
    speech += np.random.normal(0, 0.003, size=len(t))

    # Normalize amplitude
    speech /= np.max(np.abs(speech)) + 1e-8
    return speech.astype(np.float32)


def generate_fake_voice(sample_id, duration=4.0, sr=16000):
    """
    Simulate AI synthetic voice deepfakes with neural vocoder artifacts:
    flat robotic pitch, phase jitter, high-frequency metallic buzz, and spectral unnaturalness.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    np.random.seed(500 + sample_id)

    # Robotic flat pitch (lacks natural human pitch vibrato)
    f0 = 135 + (sample_id % 12) * 8
    pitch_contour = np.full_like(t, f0)  # Monotone pitch

    # Phase jitter and neural vocoder frame-stitching artifacts
    phase = 2 * np.pi * pitch_contour * t + 0.3 * np.sin(2 * np.pi * 65 * t)
    vocoder_buzz = 0.25 * np.square(np.sin(2 * np.pi * (1.8 * f0) * t))

    # High-frequency metallic aliasing (3500 Hz - 7500 Hz synthesis artifact)
    metallic_artifact = 0.12 * np.sin(2 * np.pi * 4200 * t) * np.sign(np.sin(2 * np.pi * 12 * t))
    phase_discontinuity = 0.08 * np.cos(2 * np.pi * 6800 * t)

    speech = np.sin(phase) + vocoder_buzz + metallic_artifact + phase_discontinuity

    # Artificial step-like envelope (cloned voice boundary artifact)
    step_env = np.ones_like(t)
    for step in range(1, 10):
        if step % 2 == 0:
            step_env[int(step * len(t) / 10):int((step + 1) * len(t) / 10)] *= 0.1
    speech *= step_env

    # Add synthetic electronic hiss
    speech += np.random.normal(0, 0.012, size=len(t))

    # Normalize amplitude
    speech /= np.max(np.abs(speech)) + 1e-8
    return speech.astype(np.float32)


def generate_samples(num_samples=30):
    sr = config.SAMPLE_RATE
    duration = config.DURATION

    os.makedirs(config.REAL_AUDIO_DIR, exist_ok=True)
    os.makedirs(config.FAKE_AUDIO_DIR, exist_ok=True)

    print(f"Generating {num_samples} realistic human voice audio samples...")
    for i in range(1, num_samples + 1):
        real_audio = generate_real_voice(i, duration=duration, sr=sr)
        file_path = os.path.join(config.REAL_AUDIO_DIR, f"real_sample_{i:02d}.wav")
        sf.write(file_path, real_audio, sr)

    print(f"Generating {num_samples} realistic AI deepfake voice audio samples...")
    for i in range(1, num_samples + 1):
        fake_audio = generate_fake_voice(i, duration=duration, sr=sr)
        file_path = os.path.join(config.FAKE_AUDIO_DIR, f"fake_sample_{i:02d}.wav")
        sf.write(file_path, fake_audio, sr)

    print(f"✅ Successfully generated {num_samples} real and {num_samples} fake audio files!")


if __name__ == "__main__":
    generate_samples()

