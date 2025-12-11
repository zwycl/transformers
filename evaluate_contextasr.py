#!/usr/bin/env python3
"""
Evaluate Whisper models on ContextASR-Bench dataset with shallow fusion biasing.

Supports positive lambda biasing to boost sample-specific entities
from the dataset's entity_list for improved recognition accuracy.

ContextASR-Bench includes:
- Speech subset: Individual speech samples with domain labels
- Dialogue subset: Conversational speech with context

Each sample has an entity_list containing domain-specific terms to boost.
"""

import argparse
from typing import Dict, List, Optional

import jiwer
import numpy as np
import torch
import whisper
from whisper.decoding import DecodingTask, DecodingOptions
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer

from uncertainty_distillation.data import load_contextasr_bench
from shallow_fusion import BiasList, ShallowFusionProcessor, ShallowFusionConfig


def create_entity_bias_list(
    entities: List[str],
    tokenizer,
    lambda_val: float = 2.0,
) -> BiasList:
    """
    Create a bias list from sample-specific entities.

    Args:
        entities: List of entity strings to boost
        tokenizer: Whisper tokenizer
        lambda_val: Lambda value for boosting (positive)

    Returns:
        BiasList configured for entity boosting
    """
    bias_list = BiasList(tokenizer)

    for entity in entities:
        if entity and entity.strip():
            # Add with leading space (how tokens appear mid-sentence in Whisper)
            # This ensures first token matches what model actually predicts
            bias_list.add_text(" " + entity.strip(), lambda_val=lambda_val)

    return bias_list


def transcribe_with_model(
    model,
    audio: np.ndarray,
    language: str = "en",
    shallow_fusion_processor: Optional[ShallowFusionProcessor] = None,
    beam_size: Optional[int] = 10,
) -> str:
    """
    Transcribe audio using a Whisper model.

    Handles audio longer than 30 seconds by:
    - For baseline (no shallow fusion): uses whisper.transcribe() which processes in chunks
    - For shallow fusion: processes 30-second chunks with overlap and concatenates results

    Args:
        model: Whisper model
        audio: Audio array (16kHz)
        language: Language code
        shallow_fusion_processor: Optional processor for shallow fusion biasing
        beam_size: Beam size for beam search (None for greedy decoding)

    Returns:
        Transcribed text
    """
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 30 * SAMPLE_RATE  # 30 seconds

    if shallow_fusion_processor is None:
        # Baseline: use standard transcribe() which handles long audio properly
        result = whisper.transcribe(
            model,
            audio,
            language=language,
            beam_size=beam_size,
            without_timestamps=True,
            condition_on_previous_text=True,
        )
        return result["text"].strip()
    else:
        # Shallow fusion: need to process chunks manually since we inject custom logit filters
        audio_tensor = torch.from_numpy(audio).float()
        total_samples = audio_tensor.shape[0]

        all_texts = []
        seek = 0

        while seek < total_samples:
            # Extract chunk
            chunk_end = min(seek + CHUNK_SAMPLES, total_samples)
            chunk = audio_tensor[seek:chunk_end]

            # Pad to 30 seconds if needed
            chunk_padded = whisper.pad_or_trim(chunk)
            mel = whisper.log_mel_spectrogram(chunk_padded, n_mels=model.dims.n_mels).to(model.device)

            # Reset shallow fusion state for each chunk
            shallow_fusion_processor.reset()

            options = DecodingOptions(language=language, without_timestamps=True, beam_size=beam_size)
            task = DecodingTask(model, options)

            logit_filter = shallow_fusion_processor.get_logit_filter(
                sample_begin=task.sample_begin
            )
            task.logit_filters.append(logit_filter)

            result = task.run(mel.unsqueeze(0))[0]
            text = result.text.strip()

            if text:
                all_texts.append(text)

            seek += CHUNK_SAMPLES

        return " ".join(all_texts)


def compute_wer(reference: str, hypothesis: str, normalizer) -> float:
    """
    Compute Word Error Rate between reference and hypothesis using jiwer.

    Uses EnglishTextNormalizer for English or BasicTextNormalizer for other languages.

    Args:
        reference: Ground truth transcription
        hypothesis: Model's transcription
        normalizer: Text normalizer (EnglishTextNormalizer or BasicTextNormalizer)

    Returns:
        Word Error Rate as a float
    """
    ref_normalized = normalizer(reference)
    hyp_normalized = normalizer(hypothesis)

    if len(ref_normalized.split()) == 0:
        return 0.0 if len(hyp_normalized.split()) == 0 else 1.0

    return jiwer.wer(ref_normalized, hyp_normalized)


def print_sample_result(
    sample_id: str,
    reference: str,
    transcriptions: Dict[str, str],
    wer_scores: Dict[str, float],
    entities: List[str],
    domain: str,
):
    """Print formatted result for a single sample."""
    print(f"\n{'='*100}")
    print(f"  SAMPLE: {sample_id} | Domain: {domain}")
    print(f"  Entities: {', '.join(entities[:5])}{'...' if len(entities) > 5 else ''}")
    print(f"{'='*100}")
    print(f"  |{'-'*20}|{'-'*70}|{'-'*8}|")

    for model_name, text in transcriptions.items():
        wer = wer_scores.get(model_name, 0)
        display_text = text[:65] + "..." if len(text) > 65 else text
        print(f"  | {model_name:<18} | {display_text:<68} | {wer:>5.1%} |")

    print(f"  |{'-'*20}|{'-'*70}|{'-'*8}|")

    ref_display = reference[:65] + "..." if len(reference) > 65 else reference
    print(f"  | {'Reference':<18} | {ref_display:<68} |        |")
    print(f"  |{'-'*20}|{'-'*70}|{'-'*8}|")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper on ContextASR-Bench with entity-based shallow fusion"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tiny"],
        help="Whisper models to evaluate (default: tiny)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["Speech", "Dialogue"],
        default="Speech",
        help="ContextASR-Bench subset (default: Speech)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language: English or Mandarin (default: English)",
    )
    parser.add_argument(
        "--use-shallow-fusion",
        action="store_true",
        help="Enable shallow fusion with entity boosting from dataset",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=2.0,
        help="Lambda value for entity boosting (default: 2.0)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="Beam size for beam search (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample from dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load samples from ContextASR-Bench
    print(f"\nLoading ContextASR-Bench ({args.subset}, {args.language})...")
    samples = load_contextasr_bench(
        subset=args.subset,
        language=args.language,
        num_samples=args.num_samples,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    print(f"  Loaded {len(samples)} samples")

    # Get tokenizer for shallow fusion
    whisper_lang = "en" if args.language == "English" else "zh"
    tokenizer = None
    if args.use_shallow_fusion:
        tokenizer = get_tokenizer(multilingual=True, language=whisper_lang)
        print(f"\nShallow fusion enabled with lambda={args.lambda_val}")

    # Create text normalizer for WER computation
    if args.language == "English":
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    # Load models
    models = {}
    for model_name in args.models:
        print(f"\nLoading Whisper {model_name}...")
        models[model_name] = whisper.load_model(model_name, device=device)

    # Process samples
    fusion_status = f" [SHALLOW FUSION λ={args.lambda_val}]" if args.use_shallow_fusion else ""
    beam_status = f" [BEAM SIZE={args.beam_size}]" if args.beam_size else " [GREEDY]"
    print(f"\n{'='*100}")
    print(f"  CONTEXTASR-BENCH EVALUATION - {len(samples)} samples{fusion_status}{beam_status}")
    print(f"{'='*100}")

    results = []
    total_wer = {model_name: [] for model_name in args.models}

    for sample in samples:
        sample_id = sample["id"]
        audio = sample["audio"]
        reference = sample.get("text", "")
        entities = sample.get("entity_list", [])
        domain = sample.get("domain_label", "unknown")

        # Create per-sample shallow fusion processor if enabled
        shallow_fusion_processor = None
        if args.use_shallow_fusion and entities and tokenizer:
            bias_list = create_entity_bias_list(
                entities=entities,
                tokenizer=tokenizer,
                lambda_val=args.lambda_val,
            )
            if len(bias_list) > 0:
                config = ShallowFusionConfig(global_scale=1.0)
                shallow_fusion_processor = ShallowFusionProcessor.from_bias_list(
                    bias_list, config
                )

        # Transcribe with each model
        transcriptions = {}
        wer_scores = {}

        for model_name, model in models.items():
            text = transcribe_with_model(
                model, audio, whisper_lang, shallow_fusion_processor, args.beam_size
            )
            transcriptions[model_name] = text

            # Compute WER if reference is available
            if reference:
                wer = compute_wer(reference, text, normalizer)
                wer_scores[model_name] = wer
                total_wer[model_name].append(wer)

        # Print result
        print_sample_result(sample_id, reference, transcriptions, wer_scores, entities, domain)

        # Store results
        result = {
            "sample_id": sample_id,
            "domain": domain,
            "entities": "|".join(entities),
            "reference": reference,
        }
        for model_name, text in transcriptions.items():
            result[f"whisper_{model_name}"] = text
            result[f"wer_{model_name}"] = wer_scores.get(model_name, None)
        results.append(result)

    # Summary
    print(f"\n{'='*100}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*100}")

    print(f"\n  Average Word Error Rate (WER):")
    print(f"  {'-'*50}")

    for model_name in args.models:
        if total_wer[model_name]:
            avg_wer = np.mean(total_wer[model_name])
            std_wer = np.std(total_wer[model_name])
            print(f"  Whisper {model_name:<10}: {avg_wer:.1%} (±{std_wer:.1%})")

    print(f"  {'-'*50}")

    # Save results
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
