"""
Data loading utilities for uncertainty distillation.

Provides dataset loaders and audio processing utilities for speech datasets.
"""

from .audio import (
    TORCHAUDIO_AVAILABLE,
    WHISPER_CHUNK_LENGTH,
    WHISPER_CHUNK_OVERLAP,
    WHISPER_CHUNK_SAMPLES,
    WHISPER_SAMPLE_RATE,
    chunk_audio,
    get_audio_duration,
    load_audio_file,
    resample_audio,
)
from .datasets import (
    DATASET_CONFIGS,
    get_available_datasets,
    get_dataset_info,
    load_audio_samples,
    load_contextasr_bench,
    load_custom_audio,
)

__all__ = [
    # Audio utilities
    "TORCHAUDIO_AVAILABLE",
    "WHISPER_SAMPLE_RATE",
    "WHISPER_CHUNK_LENGTH",
    "WHISPER_CHUNK_SAMPLES",
    "WHISPER_CHUNK_OVERLAP",
    "resample_audio",
    "chunk_audio",
    "get_audio_duration",
    "load_audio_file",
    # Dataset utilities
    "DATASET_CONFIGS",
    "load_audio_samples",
    "load_custom_audio",
    "load_contextasr_bench",
    "get_available_datasets",
    "get_dataset_info",
]
