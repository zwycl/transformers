"""
Audio processing utilities for speech datasets.

Provides functions for resampling, chunking, and processing audio data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Optional torchaudio import for audio loading and resampling
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    TORCHAUDIO_AVAILABLE = False


# Constants for Whisper audio processing
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHUNK_LENGTH = 30  # seconds
WHISPER_CHUNK_SAMPLES = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH  # 480,000 samples
WHISPER_CHUNK_OVERLAP = WHISPER_SAMPLE_RATE * 2  # 2 seconds overlap


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Uses torchaudio if available, falls back to scipy.

    Args:
        audio: Audio array to resample
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    if TORCHAUDIO_AVAILABLE:
        audio_tensor = torch.tensor(audio).float()
        resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
        return resampled.numpy()
    else:
        # Fallback to scipy
        from scipy import signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)


def chunk_audio(
    audio: np.ndarray,
    chunk_length: int = WHISPER_CHUNK_SAMPLES,
    overlap: int = 0,
) -> list:
    """
    Split audio into chunks for processing.

    Args:
        audio: Audio array at 16kHz
        chunk_length: Length of each chunk in samples (default: 30 seconds)
        overlap: Number of samples to overlap between chunks

    Returns:
        List of (chunk_audio, start_sample, end_sample) tuples
    """
    total_samples = len(audio)

    if total_samples <= chunk_length:
        return [(audio, 0, total_samples)]

    chunks = []
    start = 0
    step = chunk_length - overlap

    while start < total_samples:
        end = min(start + chunk_length, total_samples)
        chunk = audio[start:end]
        chunks.append((chunk, start, end))

        if end >= total_samples:
            break
        start += step

    return chunks


def get_audio_duration(audio: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE) -> float:
    """Get audio duration in seconds."""
    return len(audio) / sample_rate


def load_audio_file(audio_path: Path) -> tuple:
    """
    Load an audio file and return the waveform and sample rate.

    Args:
        audio_path: Path to the audio file

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        ImportError: If torchaudio is not available
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("torchaudio is required for loading audio files")

    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != WHISPER_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, WHISPER_SAMPLE_RATE)
        sample_rate = WHISPER_SAMPLE_RATE

    audio_array = waveform.squeeze().numpy()
    return audio_array, sample_rate
