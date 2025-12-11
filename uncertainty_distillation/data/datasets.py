"""
Speech dataset loaders for uncertainty evaluation.

Supports various HuggingFace datasets and custom audio files.

Supported datasets:
- librispeech_dummy: Small test set (default, fast)
- librispeech_clean: LibriSpeech test-clean
- librispeech_other: LibriSpeech test-other
- peoples_speech: People's Speech (diverse American English)
- tedlium: TED-LIUM 3 (TED talks)
- voxpopuli: VoxPopuli English (European Parliament)
- gigaspeech: GigaSpeech XS (audiobooks, podcasts, YouTube)
- earnings22: Earnings-22 (financial earnings calls)
- afrispeech: AfriSpeech-200 (accented African English)
- speech_robust_bench: Speech Robust Bench (robustness with accents/noise)
- contextasr_dialogue: ContextASR-Bench dialogue (conversational speech)
- contextasr_speech: ContextASR-Bench speech (individual samples)
- custom: Custom audio files from a directory
"""

import io
import json
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from .audio import TORCHAUDIO_AVAILABLE, WHISPER_SAMPLE_RATE, load_audio_file, resample_audio

# Dataset configurations
# Note: Some datasets require authentication or use deprecated loading scripts.
DATASET_CONFIGS = {
    "librispeech_dummy": {
        "hf_path": "hf-internal-testing/librispeech_asr_dummy",
        "hf_name": "clean",
        "split": "validation",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "id",
        "description": "Small LibriSpeech dummy set (fast testing)",
        "streaming": True,
    },
    "librispeech_clean": {
        "hf_path": "openslr/librispeech_asr",
        "hf_name": "clean",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "id",
        "description": "LibriSpeech test-clean (read speech)",
        "streaming": True,
    },
    "librispeech_other": {
        "hf_path": "openslr/librispeech_asr",
        "hf_name": "other",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "id",
        "description": "LibriSpeech test-other (more challenging)",
        "streaming": True,
    },
    "peoples_speech": {
        "hf_path": "MLCommons/peoples_speech",
        "hf_name": "clean",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "id",
        "description": "People's Speech (diverse American English)",
        "streaming": True,
    },
    "tedlium": {
        "hf_path": "LIUM/tedlium",
        "hf_name": "release3",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "id",
        "description": "TED-LIUM 3 (TED talk transcriptions)",
        "streaming": True,
    },
    "voxpopuli": {
        "hf_path": "facebook/voxpopuli",
        "hf_name": "en",
        "split": "test",
        "audio_key": "audio",
        "text_key": "raw_text",
        "id_key": "audio_id",
        "description": "VoxPopuli English (European Parliament)",
        "streaming": True,
    },
    "gigaspeech": {
        "hf_path": "speechcolab/gigaspeech",
        "hf_name": "xs",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "segment_id",
        "description": "GigaSpeech XS (audiobooks, podcasts, YouTube)",
        "streaming": True,
    },
    "earnings22": {
        "hf_path": "revdotcom/earnings22",
        "hf_name": "default",
        "split": "test",
        "audio_key": "audio",
        "text_key": "sentence",
        "id_key": "id",
        "description": "Earnings-22 (financial earnings calls)",
        "streaming": True,
    },
    "afrispeech": {
        "hf_path": "tobiolatunji/afrispeech-200",
        "hf_name": "all",
        "split": "test",
        "audio_key": "audio",
        "text_key": "transcript",
        "id_key": "id",
        "description": "AfriSpeech-200 (accented African English speech)",
        "streaming": True,
    },
    "speech_robust_bench": {
        "hf_path": "mshah1/speech_robust_bench",
        "hf_name": "accented_cv",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "id",
        "description": "Speech Robust Bench (robustness evaluation with accents/noise)",
        "streaming": True,
    },
    "contextasr_dialogue": {
        "hf_path": "MrSupW/ContextASR-Bench",
        "hf_name": "ContextASR-Dialogue",
        "split": "english",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "uniq_id",
        "description": "ContextASR-Bench Dialogue (conversational speech with context)",
        "streaming": True,
    },
    "contextasr_speech": {
        "hf_path": "MrSupW/ContextASR-Bench",
        "hf_name": "ContextASR-Speech",
        "split": "english",
        "audio_key": "audio",
        "text_key": "text",
        "id_key": "uniq_id",
        "description": "ContextASR-Bench Speech (individual speech samples with domain labels)",
        "streaming": True,
    },
    "unsafe_kids": {
        "local_path": "~/UnsafeTranscriptionofKidsContent/audio",
        "description": "Local dataset: UnsafeTranscriptionofKidsContent",
    },
}


def _process_audio_item(item: dict, config: dict, index: int) -> dict:
    """Process a single audio item from a dataset."""
    # Handle different audio formats
    audio_data = item[config["audio_key"]]
    if isinstance(audio_data, dict):
        audio_array = audio_data["array"]
        sample_rate = audio_data["sampling_rate"]
    else:
        audio_array = np.array(audio_data)
        sample_rate = WHISPER_SAMPLE_RATE

    # Resample to 16kHz if needed
    if sample_rate != WHISPER_SAMPLE_RATE:
        audio_array = resample_audio(audio_array, sample_rate, WHISPER_SAMPLE_RATE)
        sample_rate = WHISPER_SAMPLE_RATE

    text = item.get(config["text_key"], "")
    sample_id = item.get(config["id_key"], f"sample_{index}")

    return {
        "audio": audio_array,
        "sr": sample_rate,
        "text": text,
        "id": str(sample_id),
    }


def load_audio_samples(
    dataset_name: str = "librispeech_dummy",
    num_samples: int = 5,
    custom_path: Optional[str] = None,
    random_sample: bool = False,
    seed: Optional[int] = None,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    domain_filter: Optional[str] = None,
) -> list:
    """
    Load audio samples from various datasets.

    Args:
        dataset_name: Name of the dataset (see DATASET_CONFIGS)
        num_samples: Number of samples to load
        custom_path: Path to custom audio files (for dataset_name="custom")
        random_sample: If True, randomly sample from the dataset
        seed: Random seed for reproducibility
        subset: Optional subset/configuration name to override default
        split: Optional split name to override default (e.g., 'farfield' for chime)
        domain_filter: Optional domain to filter by (e.g., "medical", "finance").
                       Only applies to ContextASR-Bench datasets.

    Returns:
        List of sample dictionaries with audio, sr, text, and id keys
    """
    import random

    if seed is not None:
        random.seed(seed)

    if dataset_name == "custom":
        return load_custom_audio(custom_path, num_samples)

    # Handle datasets with local_path (local directory datasets)
    if dataset_name in DATASET_CONFIGS and "local_path" in DATASET_CONFIGS[dataset_name]:
        local_path = Path(DATASET_CONFIGS[dataset_name]["local_path"]).expanduser()
        return load_custom_audio(str(local_path), num_samples)

    # Handle ContextASR-Bench with custom loader (uses JSONL + tar archives)
    if dataset_name in ("contextasr_dialogue", "contextasr_speech"):
        subset_name = "Dialogue" if dataset_name == "contextasr_dialogue" else "Speech"
        # Map split to language (default to English)
        language = "English"
        if split is not None:
            language = split.capitalize()
        return load_contextasr_bench(
            subset=subset_name,
            language=language,
            num_samples=num_samples,
            random_sample=random_sample,
            seed=seed,
            domain_filter=domain_filter,
        )

    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    config = DATASET_CONFIGS[dataset_name].copy()  # Copy to avoid modifying original

    # Override subset if provided
    if subset is not None:
        config["hf_name"] = subset

    # Override split if provided
    if split is not None:
        config["split"] = split

    print(f"Loading {num_samples} samples from {dataset_name}...")
    print(f"  Description: {config['description']}")
    print(f"  Subset: {config['hf_name']}, Split: {config['split']}")
    if random_sample:
        print(f"  Random sampling enabled (seed={seed})")

    use_streaming = config.get("streaming", False)

    try:
        dataset = load_dataset(
            config["hf_path"],
            config["hf_name"],
            split=config["split"],
            trust_remote_code=True,
            streaming=use_streaming,
        )
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        print("  Some datasets require authentication or manual download.")
        print("  For Common Voice, you may need to accept the terms on HuggingFace.")
        raise

    samples = []
    if use_streaming:
        if random_sample:
            # For streaming with random sampling, use reservoir sampling
            # Skip a random number of items first, then collect
            skip_count = random.randint(0, 500)  # Skip up to 500 items
            print(f"  Skipping first {skip_count} samples for randomization...")
            reservoir = []
            for i, item in enumerate(dataset):
                if i < skip_count:
                    continue
                if len(reservoir) < num_samples:
                    reservoir.append((i, item))
                else:
                    # Reservoir sampling: replace with decreasing probability
                    j = random.randint(0, i - skip_count)
                    if j < num_samples:
                        reservoir[j] = (i, item)
                # Stop after seeing enough items for good randomization
                if i >= skip_count + num_samples * 10:
                    break
            for idx, (orig_idx, item) in enumerate(reservoir):
                samples.append(_process_audio_item(item, config, orig_idx))
                print(f"  Loaded sample {idx+1}: {samples[-1]['id']}")
        else:
            # Sequential loading for streaming
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append(_process_audio_item(item, config, i))
                print(f"  Loaded sample {i+1}: {samples[-1]['id']}")
        return samples

    # Non-streaming dataset
    if random_sample:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    else:
        indices = list(range(min(num_samples, len(dataset))))

    for idx, i in enumerate(indices):
        item = dataset[i]
        samples.append(_process_audio_item(item, config, i))
        print(f"  Loaded sample {idx+1}: {samples[-1]['id']}")

    return samples


def load_custom_audio(audio_path: Optional[str], num_samples: int) -> list:
    """
    Load audio files from a custom directory.

    Args:
        audio_path: Path to directory containing audio files
        num_samples: Maximum number of samples to load

    Returns:
        List of sample dictionaries
    """
    if audio_path is None:
        raise ValueError("custom_path must be provided for custom dataset")

    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("torchaudio is required for loading custom audio files")

    audio_dir = Path(audio_path)
    if not audio_dir.exists():
        raise ValueError(f"Audio directory not found: {audio_path}")

    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = [
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_path}")

    print(f"Loading up to {num_samples} custom audio files...")

    samples = []
    for audio_file in audio_files:
        if len(samples) >= num_samples:
            break

        try:
            audio_array, sample_rate = load_audio_file(audio_file)
        except Exception as e:
            print(f"  Warning: Failed to load {audio_file.name}: {e}")
            continue

        # Check for transcript file
        transcript_file = audio_file.with_suffix(".txt")
        text = ""
        if transcript_file.exists():
            text = transcript_file.read_text().strip()

        samples.append({
            "audio": audio_array,
            "sr": sample_rate,
            "text": text,
            "id": audio_file.stem,
        })
        print(f"  Loaded: {audio_file.name}")

    return samples


def load_contextasr_bench(
    subset: str = "Speech",
    language: str = "English",
    num_samples: int = 5,
    random_sample: bool = False,
    seed: Optional[int] = None,
    domain_filter: Optional[str] = None,
) -> list:
    """
    Load audio samples from ContextASR-Bench dataset.

    This dataset uses a custom format with JSONL metadata and tar archives for audio.

    Args:
        subset: Either "Speech" or "Dialogue"
        language: Either "English" or "Mandarin"
        num_samples: Number of samples to load
        random_sample: If True, randomly sample from the dataset
        seed: Random seed for reproducibility
        domain_filter: Optional domain to filter by (e.g., "medical", "finance", "news").
                       Also accepts prefixes like "CMEEE", "IMCS21" to filter by source.

    Returns:
        List of sample dictionaries with audio, sr, text, and id keys
    """
    import random
    import soundfile as sf

    if seed is not None:
        random.seed(seed)

    repo_id = "MrSupW/ContextASR-Bench"
    jsonl_filename = f"ContextASR-{subset}_{language}.jsonl"

    print(f"Loading ContextASR-Bench ({subset}, {language})...")
    print(f"  Downloading metadata: {jsonl_filename}")

    # Download JSONL metadata
    jsonl_path = hf_hub_download(
        repo_id=repo_id,
        filename=jsonl_filename,
        repo_type="dataset",
    )

    # Load all metadata entries
    entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"  Found {len(entries)} entries in metadata")

    # Apply domain filter if specified
    if domain_filter:
        domain_lower = domain_filter.lower()
        filtered_entries = []
        for entry in entries:
            # Check domain_label field
            entry_domain = entry.get("domain_label", "").lower()
            entry_id = entry.get("uniq_id", "").upper()

            # Medical domain: match CMEEE, IMCS21, or domain containing "medical"
            if domain_lower == "medical":
                if "medical" in entry_domain or entry_id.startswith("CMEEE") or entry_id.startswith("IMCS21"):
                    filtered_entries.append(entry)
            # Allow filtering by source prefix (e.g., "CMEEE", "MSRA", "DLNER")
            elif entry_id.startswith(domain_filter.upper()):
                filtered_entries.append(entry)
            # General domain matching
            elif domain_lower in entry_domain:
                filtered_entries.append(entry)

        print(f"  Filtered to {len(filtered_entries)} entries matching domain '{domain_filter}'")
        entries = filtered_entries

    # Select samples
    if random_sample:
        selected_entries = random.sample(entries, min(num_samples, len(entries)))
    else:
        selected_entries = entries[:num_samples]

    # Group by tar archive (to minimize downloads)
    # Audio paths look like: audio/ContextASR-Speech/English/FNED-004539_EN.wav
    # Tar archives: audio/ContextASR-Speech/English/ContextASR-Speech_English_1.tar

    # Download and cache tar archives as needed, extract audio
    print(f"  Loading {len(selected_entries)} audio samples...")

    # Get list of tar files for this subset/language
    all_files = list_repo_files(repo_id, repo_type="dataset")
    tar_prefix = f"audio/ContextASR-{subset}/{language}/ContextASR-{subset}_{language}_"
    tar_files = sorted([f for f in all_files if f.startswith(tar_prefix) and f.endswith(".tar")])

    if not tar_files:
        raise ValueError(f"No tar archives found for {subset}/{language}")

    # Build an index of filename -> tar file by downloading and indexing all tars
    # This is more efficient than searching through tars for each file
    print("  Building audio file index from tar archives...")
    file_to_tar = {}  # filename -> (tar_file, member_name)

    for tar_file in tar_files:
        try:
            tar_path = hf_hub_download(
                repo_id=repo_id,
                filename=tar_file,
                repo_type="dataset",
            )
            with tarfile.open(tar_path, "r") as tf:
                for member_name in tf.getnames():
                    # Extract base filename (handles ./filename.wav format)
                    base_name = Path(member_name).name
                    file_to_tar[base_name] = (tar_file, member_name)
        except Exception as e:
            print(f"    Warning: Could not index {tar_file}: {e}")
            continue

    print(f"  Indexed {len(file_to_tar)} audio files across {len(tar_files)} tar archives")

    # Cache for opened tar files
    tar_cache = {}  # tar_path -> TarFile object

    samples = []
    for idx, entry in enumerate(selected_entries):
        audio_rel_path = entry["audio"]  # e.g., audio/ContextASR-Speech/English/FNED-004539_EN.wav
        audio_filename = Path(audio_rel_path).name  # e.g., FNED-004539_EN.wav

        # Look up the tar file containing this audio
        if audio_filename not in file_to_tar:
            print(f"    Warning: Could not find audio for {entry['uniq_id']}, skipping...")
            continue

        tar_file, member_name = file_to_tar[audio_filename]

        # Open tar file if not already cached
        if tar_file not in tar_cache:
            try:
                tar_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=tar_file,
                    repo_type="dataset",
                )
                tar_cache[tar_file] = tarfile.open(tar_path, "r")
            except Exception as e:
                print(f"    Warning: Could not open {tar_file}: {e}")
                continue

        tf = tar_cache[tar_file]

        # Extract the audio file
        audio_array = None
        sample_rate = None
        try:
            member = tf.getmember(member_name)
            f = tf.extractfile(member)
            if f is not None:
                audio_bytes = f.read()
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            print(f"    Warning: Error extracting {audio_filename}: {e}")
            continue

        if audio_array is None:
            print(f"    Warning: Could not find audio for {entry['uniq_id']}, skipping...")
            continue

        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != WHISPER_SAMPLE_RATE:
            audio_array = resample_audio(audio_array, sample_rate, WHISPER_SAMPLE_RATE)

        samples.append({
            "audio": audio_array.astype(np.float32),
            "sr": WHISPER_SAMPLE_RATE,
            "text": entry.get("text", ""),
            "id": entry["uniq_id"],
            "entity_list": entry.get("entity_list", []),
            "domain_label": entry.get("domain_label", ""),
        })
        print(f"  Loaded sample {idx + 1}: {entry['uniq_id']}")

        if len(samples) >= num_samples:
            break

    # Clean up tar file handles
    for tf in tar_cache.values():
        tf.close()

    return samples


def get_available_datasets() -> list:
    """Return list of available dataset names."""
    return list(DATASET_CONFIGS.keys()) + ["custom"]


def get_dataset_info(dataset_name: str) -> dict:
    """Get configuration info for a dataset."""
    if dataset_name == "custom":
        return {"description": "Custom audio files from a directory"}
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name].copy()
