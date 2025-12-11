#!/usr/bin/env python3
"""
Evaluate Whisper models on the UnsafeTranscriptionofKidsContent dataset.

Compares transcriptions from Whisper tiny, small, medium, and large-v3
against ground truth and shows potential mis-transcriptions.

Supports shallow fusion for suppressing slur hallucinations.
"""

import argparse
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd
import torch
import whisper
from whisper.decoding import DecodingTask, DecodingOptions

from uncertainty_distillation.data import load_audio_file, TORCHAUDIO_AVAILABLE
from shallow_fusion import BiasList, ShallowFusionProcessor, ShallowFusionConfig


def load_ground_truth(dataset_path: Path) -> dict:
    """Load ground truth from CSV files and create ID -> info mapping."""
    ground_truth = {}

    csv_files = [
        dataset_path / "YOUTUBE_FINAL_AAAI.csv",
        dataset_path / "AMAZON_FINAL_AAAI.csv",
    ]

    for csv_file in csv_files:
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            sample_id = str(row['Id'])
            ground_truth[sample_id] = {
                'slur': row.get('Slur', ''),
                'youtube_transcription': row.get('YouTube', ''),
                'amazon_transcription': row.get('Amazon', ''),
                'ground_truth': row.get('Original Transcription', ''),
                'actual_word': row.get('Actual Word', ''),
                'contains_slur': row.get('Does the audio contain a slur or not?', ''),
                'link': row.get('Link', ''),
            }

    return ground_truth


def create_slur_bias_list(
    slurs: Set[str],
    tokenizer,
    lambda_val: float = -10.0,
) -> BiasList:
    """
    Create a bias list for suppressing slur words.

    Args:
        slurs: Set of slur words to suppress
        tokenizer: Whisper tokenizer
        lambda_val: Suppression strength (negative value)

    Returns:
        BiasList configured for slur suppression
    """
    bias_list = BiasList(tokenizer)

    for slur in slurs:
        if not slur or not slur.strip():
            continue
        # Add word with variants (different cases, with/without leading space)
        bias_list.add_word(slur.strip(), lambda_val=lambda_val, include_variants=True)

    return bias_list


def transcribe_with_model(
    model,
    audio: np.ndarray,
    language: str = "en",
    shallow_fusion_processor: Optional[ShallowFusionProcessor] = None,
) -> str:
    """
    Transcribe audio using a Whisper model.

    Args:
        model: Whisper model
        audio: Audio array
        language: Language code
        shallow_fusion_processor: Optional processor for shallow fusion biasing

    Returns:
        Transcribed text
    """
    # Pad or trim to 30 seconds
    audio_tensor = torch.from_numpy(audio).float()
    audio_padded = whisper.pad_or_trim(audio_tensor)

    # Create mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_padded, n_mels=model.dims.n_mels).to(model.device)

    if shallow_fusion_processor is not None:
        # Use DecodingTask with shallow fusion filter
        options = DecodingOptions(language=language, without_timestamps=True)
        task = DecodingTask(model, options)

        # Add shallow fusion logit filter
        logit_filter = shallow_fusion_processor.get_logit_filter(
            sample_begin=task.sample_begin
        )
        task.logit_filters.append(logit_filter)

        # Run decoding
        result = task.run(mel.unsqueeze(0))[0]
        return result.text.strip()
    else:
        # Standard decoding
        options = whisper.DecodingOptions(language=language, without_timestamps=True)
        result = whisper.decode(model, mel, options)
        return result.text.strip()


def format_table_row(label: str, value, width: int = 80) -> str:
    """Format a single table row."""
    # Handle NaN/None values
    if pd.isna(value) or value is None:
        value = "N/A"
    value = str(value)
    # Wrap long text
    if len(value) > width:
        value = value[:width-3] + "..."
    return f"  | {label:<18} | {value:<{width}} |"


def print_sample_comparison(sample_id: str, transcriptions: dict, ground_truth: dict):
    """Print a formatted comparison table for a single sample."""
    info = ground_truth.get(sample_id, {})

    print(f"\n{'='*104}")
    print(f"  SAMPLE ID: {sample_id}")
    print(f"{'='*104}")

    # Header
    print(f"  |{'-'*20}|{'-'*81}|")

    # Whisper model transcriptions
    for model_name in ['tiny', 'small', 'medium', 'large-v3']:
        if model_name in transcriptions:
            print(format_table_row(f"Whisper {model_name}", transcriptions[model_name]))

    print(f"  |{'-'*20}|{'-'*81}|")

    # Ground truth info
    if info:
        gt_text = info.get('ground_truth', 'N/A')
        if gt_text:
            print(format_table_row("Ground Truth", gt_text))

        # Show what it was mis-transcribed as
        slur = info.get('slur', '')
        amazon = info.get('amazon_transcription', '')
        if slur and amazon:
            print(format_table_row("Mis-transcribed as", f'"{amazon}" (Amazon)'))

        actual = info.get('actual_word', '')
        if actual:
            print(format_table_row("Actual word", actual))

        contains = info.get('contains_slur', '')
        if contains:
            print(format_table_row("Contains slur?", contains))
    else:
        print(format_table_row("Ground Truth", "Not found in CSV"))

    print(f"  |{'-'*20}|{'-'*81}|")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper models on unsafe kids content dataset")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tiny", "small", "medium", "large-v3"],
        help="Whisper models to evaluate (default: tiny small medium large-v3)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(Path.home() / "UnsafeTranscriptionofKidsContent"),
        help="Path to the dataset",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--use-shallow-fusion",
        action="store_true",
        help="Enable shallow fusion to suppress slur words from ground truth",
    )
    parser.add_argument(
        "--shallow-fusion-lambda",
        type=float,
        default=-10.0,
        help="Lambda value for shallow fusion suppression (default: -10.0, more negative = stronger)",
    )
    parser.add_argument(
        "--extra-suppress-words",
        type=str,
        nargs="*",
        default=[],
        help="Additional words to suppress via shallow fusion",
    )
    args = parser.parse_args()

    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("torchaudio is required for this script")

    dataset_path = Path(args.dataset_path)
    audio_dir = dataset_path / "audio"

    if not audio_dir.exists():
        raise ValueError(f"Audio directory not found: {audio_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load ground truth
    print("\nLoading ground truth from CSV files...")
    ground_truth = load_ground_truth(dataset_path)
    print(f"  Loaded {len(ground_truth)} entries")

    # Setup shallow fusion if enabled
    shallow_fusion_processor = None
    if args.use_shallow_fusion:
        print("\nSetting up shallow fusion for slur suppression...")

        # Collect all unique slurs from ground truth
        slurs_to_suppress = set()
        for info in ground_truth.values():
            slur = info.get('slur', '')
            if slur and pd.notna(slur) and str(slur).strip():
                slurs_to_suppress.add(str(slur).strip())

        # Add extra suppress words
        for word in args.extra_suppress_words:
            if word.strip():
                slurs_to_suppress.add(word.strip())

        print(f"  Suppressing {len(slurs_to_suppress)} unique words")
        print(f"  Lambda value: {args.shallow_fusion_lambda}")

        if slurs_to_suppress:
            # Create tokenizer for bias list
            from whisper.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(multilingual=True, language=args.language)

            # Create bias list
            bias_list = create_slur_bias_list(
                slurs_to_suppress,
                tokenizer,
                lambda_val=args.shallow_fusion_lambda,
            )
            print(f"  Created {len(bias_list)} bias entries (including variants)")

            # Create processor
            config = ShallowFusionConfig(global_scale=1.0)
            shallow_fusion_processor = ShallowFusionProcessor.from_bias_list(
                bias_list, config
            )
            print(f"  Shallow fusion processor ready")

            # Show suppressed words
            if len(slurs_to_suppress) <= 10:
                print(f"  Words: {', '.join(sorted(slurs_to_suppress))}")
            else:
                sample = list(sorted(slurs_to_suppress))[:5]
                print(f"  Words (sample): {', '.join(sample)} ... and {len(slurs_to_suppress)-5} more")

    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted([
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in audio_extensions
    ], key=lambda x: int(x.stem) if x.stem.isdigit() else 0)

    print(f"  Found {len(audio_files)} audio files")

    # Filter to files that have ground truth with non-null Actual Word
    def has_actual_word(sample_id: str) -> bool:
        if sample_id not in ground_truth:
            return False
        actual_word = ground_truth[sample_id].get('actual_word', '')
        return pd.notna(actual_word) and str(actual_word).strip() != ''

    audio_files_with_gt = [f for f in audio_files if has_actual_word(f.stem)]
    print(f"  {len(audio_files_with_gt)} files have matching ground truth with Actual Word")

    # Load models
    models = {}
    for model_name in args.models:
        print(f"\nLoading Whisper {model_name}...")
        models[model_name] = whisper.load_model(model_name, device=device)

    # Process samples
    print(f"\n{'='*104}")
    fusion_status = " [SHALLOW FUSION ENABLED]" if shallow_fusion_processor else ""
    print(f"  WHISPER MODEL COMPARISON - {args.num_samples} samples{fusion_status}")
    print(f"{'='*104}")

    results = []
    samples_processed = 0

    for audio_file in audio_files_with_gt:
        if samples_processed >= args.num_samples:
            break

        sample_id = audio_file.stem

        # Load audio
        try:
            audio, sr = load_audio_file(audio_file)
        except Exception as e:
            print(f"\n  Skipping {audio_file.name}: {e}")
            continue

        # Transcribe with each model
        transcriptions = {}
        for model_name, model in models.items():
            transcriptions[model_name] = transcribe_with_model(
                model, audio, args.language, shallow_fusion_processor
            )

        # Print comparison
        print_sample_comparison(sample_id, transcriptions, ground_truth)

        # Store results
        info = ground_truth.get(sample_id, {})
        results.append({
            'sample_id': sample_id,
            'ground_truth': info.get('ground_truth', ''),
            'slur': info.get('slur', ''),
            'actual_word': info.get('actual_word', ''),
            'amazon_transcription': info.get('amazon_transcription', ''),
            **{f'whisper_{m}': t for m, t in transcriptions.items()}
        })

        samples_processed += 1

    print(f"\n{'='*104}")
    print(f"  EVALUATION COMPLETE - {samples_processed} samples processed")
    print(f"{'='*104}")

    # Summary: Check if any Whisper model produced the slur
    print(f"\n  SLUR DETECTION SUMMARY:")
    print(f"  {'-'*60}")

    def safe_str(val):
        return str(val).lower() if pd.notna(val) else ''

    for model_name in args.models:
        slur_count = 0
        for result in results:
            slur = safe_str(result.get('slur', ''))
            transcription = safe_str(result.get(f'whisper_{model_name}', ''))
            if slur and slur in transcription:
                slur_count += 1
        print(f"  Whisper {model_name:8}: {slur_count}/{len(results)} samples contained hallucinated slur")

    # Amazon baseline
    amazon_slur_count = sum(1 for r in results if safe_str(r.get('slur', '')) in safe_str(r.get('amazon_transcription', '')) and safe_str(r.get('slur', '')))
    print(f"  Amazon Transcribe: {amazon_slur_count}/{len(results)} samples contained hallucinated slur")

    # Save results if output specified
    if args.output:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
