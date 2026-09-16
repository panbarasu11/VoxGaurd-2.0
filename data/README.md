# Dataset — VoxGuard

This folder holds the audio files for training VoxGuard.

## Folder Structure

```
data/
├── real/        ← Put genuine/real voice .wav files here
├── fake/        ← Put AI-generated/spoofed voice .wav files here
└── features.npz ← Auto-generated after running extract_features.py
```

## Recommended Dataset: ASVspoof 2019 (LA Partition)

The standard benchmark dataset for audio spoofing detection.

**Download:** https://datashare.ed.ac.uk/handle/10283/3336

1. Download the **LA (Logical Access)** partition
2. Copy real voice files to `data/real/`
3. Copy spoofed voice files to `data/fake/`
4. Run: `python extract_features.py`

## Other Free Datasets

- **FakeAVCeleb**: https://github.com/DASH-Lab/FakeAVCeleb
- **WaveFake**: https://github.com/RUB-SysSec/WaveFake
- **LibriSpeech** (real speech): https://www.openslr.org/12/

## Quick Test (No Dataset Needed)

You can generate synthetic test audio using Python's `soundfile` library
or record your own voice + use a TTS tool like ElevenLabs for fake samples.
