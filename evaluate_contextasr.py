#!/usr/bin/env python3
"""
Evaluate Whisper models on ContextASR-Bench dataset with shallow fusion biasing.

Uses HuggingFace transformers implementation with CostSubtractionLogitsProcessor
for beam search compatibility. Supports batch processing with combined entity lists.

ContextASR-Bench includes:
- Speech subset: Individual speech samples with domain labels
- Dialogue subset: Conversational speech with context

Each sample has an entity_list containing domain-specific terms to boost.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import jiwer
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from uncertainty_distillation.data import load_contextasr_bench
from shallow_fusion import BiasList, ShallowFusionProcessor, ShallowFusionConfig


@dataclass
class EvalResult:
    """Result for a single sample evaluation."""
    sample_id: str
    domain: str
    reference: str
    transcription: str
    wer: float
    entities: List[str]


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
            bias_list.add_text(" " + entity.strip(), lambda_val=lambda_val)

    return bias_list


def create_combined_entity_bias_list(
    batch_samples: List[dict],
    tokenizer,
    lambda_val: float = 2.0,
) -> BiasList:
    """
    Create a combined bias list from all entities in a batch.

    Args:
        batch_samples: List of sample dictionaries with entity_list field
        tokenizer: WhisperTokenizer
        lambda_val: Lambda value for boosting (positive)

    Returns:
        BiasList configured with all unique entities from the batch
    """
    bias_list = BiasList(tokenizer)

    # Collect all unique entities from the batch
    seen_entities = set()
    for sample in batch_samples:
        entities = sample.get("entity_list", [])
        for entity in entities:
            if entity and entity.strip():
                normalized = entity.strip()
                if normalized not in seen_entities:
                    seen_entities.add(normalized)
                    bias_list.add_text(" " + normalized, lambda_val=lambda_val)

    return bias_list


def prepare_batch_features(
    processor: WhisperProcessor,
    audio_list: List[np.ndarray],
    device: torch.device,
) -> tuple:
    """
    Prepare batched input features with attention mask for variable length audio.

    Args:
        processor: WhisperProcessor
        audio_list: List of audio arrays
        device: Target device

    Returns:
        Tuple of (batched_features, attention_mask, original_lengths)
    """
    SAMPLE_RATE = 16000

    # Process each audio to mel features
    feature_list = []
    original_lengths = []

    for audio in audio_list:
        # Ensure audio is numpy array
        if isinstance(audio, dict):
            audio = audio["array"]

        # Process to mel features
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        features = inputs.input_features.to(device)
        original_lengths.append(features.shape[-1])
        feature_list.append(features)

    # Find max length and pad all
    max_frames = max(original_lengths)

    padded_features = []
    for features in feature_list:
        if features.shape[-1] < max_frames:
            features = torch.nn.functional.pad(
                features,
                (0, max_frames - features.shape[-1]),
            )
        padded_features.append(features)

    # Stack into batch
    batched_features = torch.cat(padded_features, dim=0)

    # Create attention mask
    attention_mask = torch.zeros(len(feature_list), max_frames, device=device)
    for i, orig_len in enumerate(original_lengths):
        attention_mask[i, :orig_len] = 1

    return batched_features, attention_mask, original_lengths


def transcribe_batch(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    batch_samples: List[dict],
    language: str = "en",
    beam_size: int = 5,
    use_shallow_fusion: bool = False,
    lambda_val: float = 2.0,
) -> List[str]:
    """
    Batch transcribe audio samples.

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor
        batch_samples: List of sample dictionaries with audio and entity_list
        language: Language code
        beam_size: Beam size for beam search
        use_shallow_fusion: Whether to use shallow fusion
        lambda_val: Lambda value for entity boosting

    Returns:
        List of transcriptions
    """
    device = next(model.parameters()).device

    # Prepare audio list
    audio_list = []
    for sample in batch_samples:
        audio = sample["audio"]
        if isinstance(audio, dict):
            audio = audio["array"]
        audio_list.append(audio)

    # Prepare batched features
    batched_features, attention_mask, _ = prepare_batch_features(
        processor, audio_list, device
    )

    # Create shallow fusion processor if enabled
    logits_processor_list = []
    if use_shallow_fusion:
        combined_bias_list = create_combined_entity_bias_list(
            batch_samples,
            processor.tokenizer,
            lambda_val=lambda_val,
        )
        if len(combined_bias_list) > 0:
            config = ShallowFusionConfig(global_scale=1.0)
            sf_processor = ShallowFusionProcessor.from_bias_list(
                combined_bias_list, config
            )
            logits_processor = sf_processor.get_cost_subtraction_processor()
            logits_processor_list.append(logits_processor)

    # Generate
    generate_kwargs = {
        "attention_mask": attention_mask,
        "num_beams": beam_size,
        "return_timestamps": True,
        "language": language,
        "task": "transcribe",
    }

    if logits_processor_list:
        generate_kwargs["logits_processor"] = logits_processor_list

    outputs = model.generate(batched_features, **generate_kwargs)

    # Decode
    transcriptions = processor.batch_decode(outputs, skip_special_tokens=True)
    return [t.strip() for t in transcriptions]


def compute_wer(reference: str, hypothesis: str, normalizer) -> float:
    """
    Compute Word Error Rate between reference and hypothesis.

    Args:
        reference: Ground truth transcription
        hypothesis: Model's transcription
        normalizer: Text normalizer

    Returns:
        Word Error Rate as a float
    """
    ref_normalized = normalizer(reference)
    hyp_normalized = normalizer(hypothesis)

    if len(ref_normalized.split()) == 0:
        return 0.0 if len(hyp_normalized.split()) == 0 else 1.0

    return jiwer.wer(ref_normalized, hyp_normalized)


def batch_iterator(items, batch_size):
    """Yield batches of items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def print_sample_result(
    sample_id: str,
    reference: str,
    transcription: str,
    wer: float,
    entities: List[str],
    domain: str,
):
    """Print formatted result for a single sample."""
    print(f"\n  {sample_id} | {domain} | WER: {wer:.1%}")
    print(f"  Entities: {', '.join(entities[:3])}{'...' if len(entities) > 3 else ''}")

    ref_display = reference[:80] + "..." if len(reference) > 80 else reference
    hyp_display = transcription[:80] + "..." if len(transcription) > 80 else transcription

    print(f"  REF: {ref_display}")
    print(f"  HYP: {hyp_display}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper on ContextASR-Bench with entity-based shallow fusion"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="HuggingFace Whisper model (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["Speech", "Dialogue"],
        default="Speech",
        help="ContextASR-Bench subset (default: Speech)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter by domain (e.g., medical, legal, finance)",
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
        default=5,
        help="Beam size for beam search (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each sample",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print(f"\nLoading model: {args.model}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    processor = WhisperProcessor.from_pretrained(args.model)
    model.eval()

    # Load samples from ContextASR-Bench
    print(f"\nLoading ContextASR-Bench ({args.subset}, {args.language})...")
    samples = load_contextasr_bench(
        subset=args.subset,
        language=args.language,
        num_samples=args.num_samples,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    # Filter by domain if specified
    if args.domain:
        samples = [s for s in samples if args.domain.lower() in s.get("domain_label", "").lower()]
        print(f"  Filtered to {len(samples)} {args.domain} samples")
    else:
        print(f"  Loaded {len(samples)} samples")

    # Create text normalizer for WER computation
    whisper_lang = "en" if args.language == "English" else "zh"
    if args.language == "English":
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    # Run evaluation
    fusion_status = f" [SHALLOW FUSION λ={args.lambda_val}]" if args.use_shallow_fusion else " [BASELINE]"
    print(f"\n{'='*80}")
    print(f"EVALUATION: {args.model}{fusion_status}")
    print(f"Samples: {len(samples)} | Batch size: {args.batch_size} | Beam size: {args.beam_size}")
    print(f"{'='*80}")

    results = []
    all_wer = []

    # Process in batches with progress bar
    batches = list(batch_iterator(samples, args.batch_size))

    for batch in tqdm(batches, desc="Evaluating"):
        # Transcribe batch
        transcriptions = transcribe_batch(
            model=model,
            processor=processor,
            batch_samples=batch,
            language=whisper_lang,
            beam_size=args.beam_size,
            use_shallow_fusion=args.use_shallow_fusion,
            lambda_val=args.lambda_val,
        )

        # Compute WER for each sample
        for sample, transcription in zip(batch, transcriptions):
            sample_id = sample["id"]
            reference = sample.get("text", "")
            entities = sample.get("entity_list", [])
            domain = sample.get("domain_label", "unknown")

            wer = compute_wer(reference, transcription, normalizer)
            all_wer.append(wer)

            result = EvalResult(
                sample_id=sample_id,
                domain=domain,
                reference=reference,
                transcription=transcription,
                wer=wer,
                entities=entities,
            )
            results.append(result)

            if args.verbose:
                print_sample_result(
                    sample_id, reference, transcription, wer, entities, domain
                )

    # Summary
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")

    avg_wer = np.mean(all_wer)
    std_wer = np.std(all_wer)
    median_wer = np.median(all_wer)

    print(f"\n  Model: {args.model}")
    print(f"  Shallow Fusion: {'Yes (λ=' + str(args.lambda_val) + ')' if args.use_shallow_fusion else 'No'}")
    print(f"  Samples: {len(results)}")
    print(f"\n  Average WER: {avg_wer:.1%} (±{std_wer:.1%})")
    print(f"  Median WER:  {median_wer:.1%}")

    # Domain breakdown
    domain_wer = {}
    for r in results:
        if r.domain not in domain_wer:
            domain_wer[r.domain] = []
        domain_wer[r.domain].append(r.wer)

    if len(domain_wer) > 1:
        print(f"\n  WER by Domain:")
        print(f"  {'-'*40}")
        for domain, wers in sorted(domain_wer.items()):
            print(f"  {domain:<20}: {np.mean(wers):.1%} ({len(wers)} samples)")

    # Save results
    if args.output:
        import pandas as pd
        df = pd.DataFrame([
            {
                "sample_id": r.sample_id,
                "domain": r.domain,
                "reference": r.reference,
                "transcription": r.transcription,
                "wer": r.wer,
                "entities": "|".join(r.entities),
            }
            for r in results
        ])
        df.to_csv(args.output, index=False)
        print(f"\n  Results saved to: {args.output}")

    print(f"\n{'='*80}")

    return avg_wer


if __name__ == "__main__":
    main()
