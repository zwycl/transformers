"""
Test script for Stage 1: Training-Free Uncertainty Signals

Tests the SignalEnsemble on Whisper using various speech datasets from HuggingFace.
Shows token-level uncertainty signals aligned with transcription.
Supports shallow fusion with entity biasing for ContextASR-Bench datasets.

Uses Transformers WhisperForConditionalGeneration API.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional

import jiwer
import numpy as np
import torch

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from uncertainty_distillation import SignalEnsemble, UncertaintySignals
from uncertainty_distillation.data import (
    DATASET_CONFIGS,
    WHISPER_SAMPLE_RATE,
    get_audio_duration,
    load_audio_samples,
)
from uncertainty_distillation.signals.gradnorm import GradNormExtractor
from shallow_fusion import BiasList, ShallowFusionProcessor, ShallowFusionConfig


def prepare_input_features(processor: WhisperProcessor, audio: np.ndarray, device: str, truncation: bool = False):
    """Convert audio to input features using WhisperProcessor.

    Args:
        processor: WhisperProcessor
        audio: Audio numpy array
        device: Device to place tensors on
        truncation: Whether to truncate audio to 30 seconds (default False for long-form)

    Returns:
        Input features tensor
    """
    inputs = processor(
        audio,
        sampling_rate=WHISPER_SAMPLE_RATE,
        return_tensors="pt",
        truncation=truncation,
    )
    return inputs.input_features.to(device)


# Maximum frames the Whisper encoder can process at once (30 seconds at 100 fps)
MAX_ENCODER_FRAMES = 3000


def prepare_batch_input_features(processor: WhisperProcessor, audio_list: List[np.ndarray], device: str):
    """Prepare a batch of input features from audio arrays.

    Args:
        processor: WhisperProcessor
        audio_list: List of audio numpy arrays
        device: Device to place tensors on

    Returns:
        Batched input features tensor of shape [batch_size, n_mels, n_frames]
    """
    inputs = processor(
        audio_list,
        sampling_rate=WHISPER_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    return inputs.input_features.to(device)


def batch_iterator(samples, batch_size):
    """Iterate over samples in batches.

    Args:
        samples: List of sample dictionaries
        batch_size: Number of samples per batch

    Yields:
        List of sample dictionaries for each batch
    """
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


def compute_wer(reference: str, hypothesis: str, normalizer) -> float:
    """
    Compute Word Error Rate between reference and hypothesis using jiwer.

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


def create_entity_bias_list(
    entities: List[str],
    tokenizer,
    lambda_val: float = 2.0,
) -> BiasList:
    """
    Create a bias list from sample-specific entities.

    Args:
        entities: List of entity strings to boost
        tokenizer: WhisperTokenizer
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


@dataclass
class TranscriptionResult:
    """Result from transcription including segment information."""
    transcription: str
    hypotheses: List[str]
    segments: Optional[List[dict]] = None  # Segment boundaries for long-form audio


def transcribe_with_shallow_fusion(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    input_features: torch.Tensor,
    shallow_fusion_processor: ShallowFusionProcessor,
    n_beams: int = 10,
    language: str = "en",
) -> TranscriptionResult:
    """
    Transcribe audio using shallow fusion for entity biasing.

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor
        input_features: Input features from processor
        shallow_fusion_processor: Shallow fusion processor with entity biases
        n_beams: Number of beams for beam search
        language: Language code

    Returns:
        TranscriptionResult with transcription, hypotheses, and segment info
    """
    # Get logits processor for shallow fusion
    logits_processor = shallow_fusion_processor.get_logits_processor()

    # Generate with shallow fusion
    # Use return_timestamps=True for long-form audio transcription
    # Use return_segments=True to get segment boundaries
    outputs = model.generate(
        input_features,
        num_beams=n_beams,
        logits_processor=[logits_processor],
        return_timestamps=True,
        return_segments=True,
        language=language,
        task="transcribe",
    )

    # Handle dict output from return_segments=True
    if isinstance(outputs, dict):
        sequences = outputs["sequences"]
        segments = outputs.get("segments", [[]])[0]  # First batch item
    else:
        sequences = outputs
        segments = None

    # Decode transcription
    transcription = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()

    return TranscriptionResult(
        transcription=transcription,
        hypotheses=[transcription],
        segments=segments,
    )


def transcribe_standard(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    input_features: torch.Tensor,
    n_beams: int = 10,
    language: str = "en",
) -> TranscriptionResult:
    """
    Transcribe audio without shallow fusion.

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor
        input_features: Input features from processor
        n_beams: Number of beams for beam search
        language: Language code

    Returns:
        TranscriptionResult with transcription, hypotheses, and segment info
    """
    # Generate with return_timestamps for long-form audio
    outputs = model.generate(
        input_features,
        num_beams=n_beams,
        return_timestamps=True,
        return_segments=True,
        language=language,
        task="transcribe",
    )

    # Handle dict output from return_segments=True
    if isinstance(outputs, dict):
        sequences = outputs["sequences"]
        segments = outputs.get("segments", [[]])[0]  # First batch item
    else:
        sequences = outputs
        segments = None

    # Decode transcription
    transcription = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()

    return TranscriptionResult(
        transcription=transcription,
        hypotheses=[transcription],
        segments=segments,
    )


@dataclass
class SampleResult:
    """Results from processing a full audio sample."""
    transcription: str
    tokens: torch.Tensor
    sot_len: int
    signals: "UncertaintySignals"
    nbest_hypotheses: list
    gradnorm: Optional[float] = None
    segments: Optional[List[dict]] = None  # Segment info for long-form audio
    is_longform: bool = False


def extract_signals_for_segment(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    ensemble: SignalEnsemble,
    input_features: torch.Tensor,
    segment_tokens: torch.Tensor,
    start_frame: int,
    end_frame: int,
    args,
) -> "UncertaintySignals":
    """
    Extract uncertainty signals for a single segment of long-form audio.

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor
        ensemble: SignalEnsemble instance
        input_features: Full input features tensor
        segment_tokens: Token IDs for this segment
        start_frame: Start frame index (100 fps)
        end_frame: End frame index (100 fps)
        args: Command line arguments

    Returns:
        UncertaintySignals for this segment
    """
    # Extract segment features (pad to 3000 frames if needed)
    segment_features = input_features[:, :, start_frame:end_frame]
    if segment_features.shape[-1] < MAX_ENCODER_FRAMES:
        segment_features = torch.nn.functional.pad(
            segment_features,
            (0, MAX_ENCODER_FRAMES - segment_features.shape[-1]),
        )

    # Extract signals for this segment
    signals = ensemble.extract_all(
        segment_features,
        segment_tokens,
        skip_nbest=True,
        skip_feature_distance=args.skip_feature_distance,
        skip_attention_entropy=True,
    )

    return signals


def aggregate_signals(
    segment_signals: List["UncertaintySignals"],
    segment_weights: List[float],
) -> "UncertaintySignals":
    """
    Aggregate signals from multiple segments into a single result.

    Uses weighted average based on segment duration for utterance-level metrics.
    Concatenates token-level signals.

    Args:
        segment_signals: List of UncertaintySignals from each segment
        segment_weights: Weights for each segment (typically proportional to duration)

    Returns:
        Aggregated UncertaintySignals
    """
    from uncertainty_distillation.signals.attention_entropy import AttentionEntropyResult
    from uncertainty_distillation.signals.nbest_disagreement import NBestDisagreementResult
    from uncertainty_distillation.signals.feature_distance import FeatureDistanceResult
    from uncertainty_distillation.signals.predictive_entropy import PredictiveEntropyResult

    if len(segment_signals) == 1:
        return segment_signals[0]

    # Normalize weights
    total_weight = sum(segment_weights)
    weights = [w / total_weight for w in segment_weights]

    # Concatenate token-level signals
    all_token_entropy = torch.cat([s.predictive_entropy.token_entropy for s in segment_signals])
    all_normalized_entropy = torch.cat([s.predictive_entropy.normalized_entropy for s in segment_signals])
    all_logit_margin = torch.cat([s.predictive_entropy.logit_margin for s in segment_signals])

    # Weighted average for utterance-level signals
    # Convert to floats for aggregation
    utterance_pred_entropy = torch.tensor(sum(
        w * float(s.predictive_entropy.utterance_entropy) for w, s in zip(weights, segment_signals)
    ))
    utterance_margin = torch.tensor(sum(
        w * float(s.predictive_entropy.utterance_margin) for w, s in zip(weights, segment_signals)
    ))
    utterance_feature_dist = torch.tensor(sum(
        w * float(s.feature_distance.mahalanobis_distance) for w, s in zip(weights, segment_signals)
    ))

    # Create aggregated results
    pred_entropy_result = PredictiveEntropyResult(
        token_entropy=all_token_entropy,
        normalized_entropy=all_normalized_entropy,
        utterance_entropy=utterance_pred_entropy,
        logit_margin=all_logit_margin,
        utterance_margin=utterance_margin,
    )

    # For attention entropy and n-best, use placeholders (skip for now)
    attention_result = AttentionEntropyResult(
        token_entropy=torch.zeros(len(all_token_entropy)),
        layer_entropies=torch.zeros(1, len(all_token_entropy)),
        head_entropies=None,
        utterance_entropy=torch.tensor(0.0),
    )

    nbest_result = NBestDisagreementResult(
        position_disagreement=torch.tensor([0.0]),
        utterance_disagreement=torch.tensor(0.0),
        hypotheses=[],
    )

    feature_result = FeatureDistanceResult(
        mahalanobis_distance=utterance_feature_dist,
        euclidean_distance=torch.tensor(sum(w * float(s.feature_distance.euclidean_distance) for w, s in zip(weights, segment_signals))),
        cosine_distance=torch.tensor(sum(w * float(s.feature_distance.cosine_distance) for w, s in zip(weights, segment_signals))),
    )

    return UncertaintySignals(
        attention_entropy=attention_result,
        nbest_disagreement=nbest_result,
        feature_distance=feature_result,
        predictive_entropy=pred_entropy_result,
    )


def process_sample(
    audio: np.ndarray,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    ensemble: SignalEnsemble,
    gradnorm_extractor: Optional[GradNormExtractor],
    args,
    shallow_fusion_processor: Optional[ShallowFusionProcessor] = None,
) -> SampleResult:
    """
    Process a full audio sample and extract uncertainty signals.

    For short audio (<30s): processes directly.
    For long audio (>30s): uses segment-wise processing with aggregation.

    Args:
        audio: Full audio array
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor
        ensemble: SignalEnsemble instance
        gradnorm_extractor: GradNorm extractor (or None)
        args: Command line arguments
        shallow_fusion_processor: Optional shallow fusion processor for entity biasing

    Returns:
        SampleResult with transcription and signals
    """
    device = next(model.parameters()).device

    # Prepare input features (don't truncate - let generate() handle long-form)
    input_features = prepare_input_features(processor, audio, device, truncation=False)

    # Check if this is long-form audio (>30 seconds = >3000 frames)
    n_frames = input_features.shape[-1]
    is_longform = n_frames > MAX_ENCODER_FRAMES

    # Transcribe - use appropriate method
    if shallow_fusion_processor is not None:
        trans_result = transcribe_with_shallow_fusion(
            model, processor, input_features, shallow_fusion_processor,
            n_beams=args.n_beams, language=args.language
        )
    else:
        trans_result = transcribe_standard(
            model, processor, input_features,
            n_beams=args.n_beams, language=args.language
        )

    transcription = trans_result.transcription
    nbest_hypotheses = trans_result.hypotheses
    segments = trans_result.segments

    # Build tokens from transcription
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language,
        task="transcribe",
        no_timestamps=True,
    )
    text_tokens = processor.tokenizer.encode(transcription, add_special_tokens=False)

    # Build full token sequence: decoder_start + forced_decoder_ids + text_tokens
    decoder_start = [model.config.decoder_start_token_id]
    forced_tokens = [t[1] for t in forced_decoder_ids] if forced_decoder_ids else []
    tokens = torch.tensor(decoder_start + forced_tokens + text_tokens).to(device)
    sot_len = len(decoder_start) + len(forced_tokens)

    # Extract signals
    if is_longform and segments:
        # Long-form: extract signals per segment and aggregate
        print(f"  Long-form audio: {n_frames} frames ({n_frames/100:.1f}s), {len(segments)} segments")

        segment_signals = []
        segment_weights = []

        for seg_idx, segment in enumerate(segments):
            # Get segment boundaries (in frames, 100 fps from mel spectrogram)
            # Whisper uses time in seconds, convert to frames
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 30)
            start_frame = int(start_time * 100)  # 100 fps for mel frames
            end_frame = min(int(end_time * 100), n_frames)

            # Get segment tokens
            seg_token_ids = segment.get("tokens", [])
            # Handle tensor or list
            if isinstance(seg_token_ids, torch.Tensor):
                seg_token_ids = seg_token_ids.tolist()
            if not seg_token_ids:
                continue

            # Build segment token sequence with SOT tokens
            seg_tokens = torch.tensor(decoder_start + forced_tokens + list(seg_token_ids)).to(device)

            # Extract signals for this segment
            seg_signals = extract_signals_for_segment(
                model, processor, ensemble, input_features,
                seg_tokens, start_frame, end_frame, args
            )

            segment_signals.append(seg_signals)
            segment_weights.append(end_time - start_time)  # Weight by duration

            print(f"    Segment {seg_idx+1}: {start_time:.1f}s - {end_time:.1f}s, {len(seg_token_ids)} tokens")

        # Aggregate signals across segments
        if segment_signals:
            signals = aggregate_signals(segment_signals, segment_weights)
        else:
            # Fallback if no segments - truncate features to 30s
            truncated_features = input_features[:, :, :MAX_ENCODER_FRAMES]
            signals = ensemble.extract_all(
                truncated_features,
                tokens,
                skip_nbest=True,
                skip_feature_distance=args.skip_feature_distance,
                skip_attention_entropy=True,
            )
    else:
        # Short-form: process directly
        # Pad features to 30s if shorter (required by encoder)
        if n_frames < MAX_ENCODER_FRAMES:
            input_features = torch.nn.functional.pad(
                input_features,
                (0, MAX_ENCODER_FRAMES - n_frames),
            )

        signals = ensemble.extract_all(
            input_features,
            tokens,
            skip_nbest=True,
            skip_feature_distance=args.skip_feature_distance,
            skip_attention_entropy=True,
        )

    # Extract GradNorm if enabled (only for first 30s for long-form)
    gradnorm = None
    if gradnorm_extractor is not None:
        # Use first 30s of features for GradNorm
        gn_features = input_features[:, :, :MAX_ENCODER_FRAMES]
        if gn_features.shape[-1] < MAX_ENCODER_FRAMES:
            gn_features = torch.nn.functional.pad(
                gn_features,
                (0, MAX_ENCODER_FRAMES - gn_features.shape[-1]),
            )
        gradnorm_result = gradnorm_extractor.extract(gn_features, tokens)
        gradnorm = gradnorm_result.gradnorm

    return SampleResult(
        transcription=transcription,
        tokens=tokens,
        sot_len=sot_len,
        signals=signals,
        nbest_hypotheses=nbest_hypotheses,
        gradnorm=gradnorm,
        segments=segments,
        is_longform=is_longform,
    )


def format_uncertainty_bar(value: float, max_width: int = 20) -> str:
    """Create a visual bar for uncertainty value."""
    filled = int(value * max_width)
    return "█" * filled + "░" * (max_width - filled)


def print_token_level_analysis(tokenizer, tokens, signals, sot_len: int):
    """Print token-level uncertainty analysis."""
    # Get token strings (skip SOT tokens)
    text_tokens = tokens[sot_len:].tolist()

    # Get uncertainties (also skip SOT tokens)
    pred_entropy = signals.predictive_entropy.normalized_entropy[sot_len:].cpu()
    logit_margin = signals.predictive_entropy.logit_margin[sot_len:].cpu()

    print("\n  TOKEN-LEVEL UNCERTAINTY ANALYSIS")
    print("  " + "-" * 65)
    print(f"  {'Token':<12} {'Entropy':>7} {'Margin':>7}  {'Predictive Entropy':<20}")
    print("  " + "-" * 65)

    for i, token_id in enumerate(text_tokens):
        if i >= len(pred_entropy):
            break

        # Decode single token
        token_str = tokenizer.decode([token_id]).strip()
        if not token_str:
            token_str = f"[{token_id}]"

        # Truncate long tokens
        if len(token_str) > 10:
            token_str = token_str[:7] + "..."

        pred_val = pred_entropy[i].item()
        margin_val = logit_margin[i].item()

        # Create visual bar (scaled to 20 chars)
        pred_bar = format_uncertainty_bar(min(pred_val, 1.0), max_width=20)

        print(f"  {token_str:<12} {pred_val:>7.3f} {margin_val:>7.2f}  {pred_bar:<20}")

    print("  " + "-" * 65)


def print_high_uncertainty_tokens(tokenizer, tokens, signals, sot_len: int, threshold: float = 0.15):
    """Print tokens with high uncertainty."""
    text_tokens = tokens[sot_len:].tolist()
    pred_entropy = signals.predictive_entropy.normalized_entropy[sot_len:].cpu()

    high_uncertainty = []
    for i, token_id in enumerate(text_tokens):
        if i >= len(pred_entropy):
            break
        if pred_entropy[i].item() > threshold:
            token_str = tokenizer.decode([token_id]).strip()
            high_uncertainty.append((token_str, pred_entropy[i].item()))

    if high_uncertainty:
        print(f"\n  HIGH UNCERTAINTY TOKENS (pred_entropy > {threshold}):")
        for token_str, entropy in high_uncertainty:
            print(f"      '{token_str}' -> entropy: {entropy:.4f}")
    else:
        print(f"\n  No tokens above uncertainty threshold ({threshold})")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test uncertainty signal extraction on various speech datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  librispeech_dummy   Small LibriSpeech dummy set (default, fast)
  librispeech_clean   LibriSpeech test-clean (read speech)
  librispeech_other   LibriSpeech test-other (more challenging)
  peoples_speech      People's Speech (diverse American English)
  tedlium             TED-LIUM 3 (TED talk transcriptions)
  voxpopuli           VoxPopuli English (European Parliament)
  gigaspeech          GigaSpeech XS (audiobooks, podcasts, YouTube)
  earnings22          Earnings-22 (financial earnings calls)
  speech_robust_bench Speech Robust Bench (robustness with accents/noise)
  contextasr_dialogue ContextASR-Bench dialogue (conversational speech)
  contextasr_speech   ContextASR-Bench speech (individual samples)
  custom              Custom audio files from a directory

Examples:
  python test_signal_ensemble.py
  python test_signal_ensemble.py --dataset librispeech_clean --num-samples 10
  python test_signal_ensemble.py --dataset tedlium --model openai/whisper-small
  python test_signal_ensemble.py --dataset contextasr_speech --use-shallow-fusion
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_dummy",
        choices=list(DATASET_CONFIGS.keys()) + ["custom"],
        help="Dataset to use for testing (default: librispeech_dummy)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset/configuration to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to process (default: 3)",
    )
    parser.add_argument(
        "--num-ref-samples",
        type=int,
        default=20,
        help="Number of samples for reference statistics (default: 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="Whisper model name or path (default: openai/whisper-large-v3)",
    )
    parser.add_argument(
        "--custom-path",
        type=str,
        default=None,
        help="Path to custom audio files (required if --dataset=custom)",
    )
    parser.add_argument(
        "--n-beams",
        type=int,
        default=10,
        help="Number of beams for N-best hypotheses (default: 10)",
    )
    parser.add_argument(
        "--skip-feature-distance",
        action="store_true",
        help="Skip feature distance computation",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.10,
        help="Threshold for flagging high uncertainty tokens (default: 0.10)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for transcription (default: en)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample from the dataset instead of sequential loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility when using --random",
    )
    parser.add_argument(
        "--no-gradnorm",
        action="store_true",
        help="Disable GradNorm computation (enabled by default)",
    )
    parser.add_argument(
        "--use-shallow-fusion",
        action="store_true",
        help="Enable shallow fusion with entity biasing (requires ContextASR dataset with entity_list)",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=2.0,
        help="Lambda value for entity boosting in shallow fusion (default: 2.0)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter ContextASR samples by domain (e.g., 'medical', 'finance', 'news')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing samples (default: 1)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Whisper model and processor
    print(f"\nLoading Whisper model: {args.model}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    processor = WhisperProcessor.from_pretrained(args.model)
    print(f"  Model loaded: vocab_size={model.config.vocab_size}, d_model={model.config.d_model}")

    # Create text normalizer for WER computation
    # Use processor's tokenizer method which handles English spelling mapping
    if args.language == "en":
        normalizer = processor.tokenizer._normalize
    else:
        normalizer = BasicTextNormalizer()

    # Load test samples from selected dataset
    samples = load_audio_samples(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        custom_path=args.custom_path,
        random_sample=args.random,
        seed=args.seed,
        subset=args.subset,
        split=args.split,
        domain_filter=args.domain,
    )

    # Create signal ensemble
    print("\nCreating SignalEnsemble...")
    ensemble = SignalEnsemble(
        model,
        processor,
        n_attention_layers=4,
        n_beams=args.n_beams,
        language=args.language,
    )

    # Create GradNorm extractor (enabled by default)
    gradnorm_extractor = None
    if not args.no_gradnorm:
        print("Creating GradNorm extractor...")
        gradnorm_extractor = GradNormExtractor(model)

    # Build reference stats from samples
    if not args.skip_feature_distance:
        print("\nCollecting features for reference statistics...")

        # Use same dataset for reference stats, but more samples
        ref_samples = load_audio_samples(
            dataset_name=args.dataset,
            num_samples=args.num_ref_samples,
            custom_path=args.custom_path,
            random_sample=args.random,
            seed=args.seed + 1000 if args.seed is not None else None,
            subset=args.subset,
            split=args.split,
            domain_filter=args.domain,
        )

        # Process reference samples in batches for efficiency
        ref_features = []
        for batch in batch_iterator(ref_samples, args.batch_size):
            batch_audio = [sample["audio"] for sample in batch]
            batch_features = prepare_batch_input_features(processor, batch_audio, device)
            ref_features.extend([batch_features[i] for i in range(batch_features.shape[0])])

        features_stacked = torch.stack(ref_features, dim=0)
        print(f"  Computing reference stats from {len(ref_features)} samples (using Ledoit-Wolf shrinkage)...")
        ensemble.compute_reference_stats_from_data(features_stacked, method="ledoit_wolf")

    # Extract signals with token-level analysis
    print("\n" + "=" * 80)
    fusion_status = f" [SHALLOW FUSION lambda={args.lambda_val}]" if args.use_shallow_fusion else ""
    print(f"TOKEN-LEVEL UNCERTAINTY SIGNAL EXTRACTION ({args.dataset}){fusion_status}")
    print(f"Batch size: {args.batch_size}, Total samples: {len(samples)}")
    print("=" * 80)

    sample_idx = 0
    for batch_num, batch in enumerate(batch_iterator(samples, args.batch_size)):
        if args.batch_size > 1:
            print(f"\n{'#'*80}")
            print(f"BATCH {batch_num + 1}/{(len(samples) + args.batch_size - 1) // args.batch_size} ({len(batch)} samples)")
            print(f"{'#'*80}")

        for i, sample in enumerate(batch):
            global_idx = sample_idx + i
            print(f"\n{'='*80}")
            print(f"SAMPLE {global_idx+1}/{len(samples)}: {sample['id']}")
            print("=" * 80)
            print(f"\nGround truth: {sample['text']}")

            # Create per-sample shallow fusion processor if enabled
            shallow_fusion_processor = None
            if args.use_shallow_fusion:
                entities = sample.get("entity_list", [])
                if entities:
                    bias_list = create_entity_bias_list(
                        entities=entities,
                        tokenizer=processor.tokenizer,
                        lambda_val=args.lambda_val,
                    )
                    if len(bias_list) > 0:
                        config = ShallowFusionConfig(global_scale=1.0)
                        shallow_fusion_processor = ShallowFusionProcessor.from_bias_list(
                            bias_list, config
                        )
                        print(f"  Shallow fusion: {len(bias_list)} entities loaded")
                else:
                    print("  Shallow fusion: No entities available for this sample")

            # Get audio duration
            audio = sample["audio"]
            duration = get_audio_duration(audio)
            longform_note = " (LONG-FORM)" if duration > 30 else ""
            print(f"Audio duration: {duration:.2f}s{longform_note}")

            # Process full audio sample
            result = process_sample(
                audio=audio,
                model=model,
                processor=processor,
                ensemble=ensemble,
                gradnorm_extractor=gradnorm_extractor,
                args=args,
                shallow_fusion_processor=shallow_fusion_processor,
            )

            # Display long-form processing info
            if result.is_longform and result.segments:
                print(f"  Processed {len(result.segments)} segments for signal extraction")

            print(f"\nTranscription: {result.transcription}")

            # Compute and print WER
            ground_truth = sample.get('text', '')
            if ground_truth:
                wer = compute_wer(ground_truth, result.transcription, normalizer)
                print(f"WER: {wer:.1%}")

            # Print utterance-level summary
            print("\n  UTTERANCE-LEVEL SUMMARY")
            print("  " + "-" * 40)

            signals = result.signals
            print(f"  Mahalanobis Distance: {float(signals.utterance_feature_distance):.4f}")
            print(f"  Predictive Entropy:   {float(signals.utterance_predictive_entropy):.4f}")
            print(f"  Logit Margin:         {float(signals.predictive_entropy.utterance_margin):.4f}  (higher = more confident)")

            if result.gradnorm is not None:
                print(f"  GradNorm:             {float(result.gradnorm):.4f}  (lower = OOD/unfamiliar)")

            # Print N-best hypotheses
            if result.nbest_hypotheses:
                print(f"\n  N-BEST HYPOTHESES (top {args.n_beams}):")
                for j, hyp in enumerate(result.nbest_hypotheses[:args.n_beams]):
                    print(f"    {j+1}. {hyp}")

            # Print token-level analysis
            print_token_level_analysis(
                processor.tokenizer,
                result.tokens,
                result.signals,
                result.sot_len,
            )

            # Print high uncertainty tokens
            print_high_uncertainty_tokens(
                processor.tokenizer,
                result.tokens,
                result.signals,
                result.sot_len,
                threshold=args.uncertainty_threshold,
            )

        sample_idx += len(batch)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
