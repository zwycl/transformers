"""
Unified transcription with token-level uncertainty signals.

Provides a transcribe() function using Transformers WhisperForConditionalGeneration
that outputs:
- Token-level predictive entropy (normalized)
- Token-level logit margin (top1 - top2)
- Per-segment uncertainty statistics

Supports both baseline transcription and batch processing.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import WhisperForConditionalGeneration, WhisperProcessor


# Audio constants (matching Whisper)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000


@dataclass
class TokenUncertainty:
    """Uncertainty signals for a single token."""
    token_id: int
    token_str: str
    entropy: float  # Normalized predictive entropy [0, 1]
    margin: float   # Logit margin (top1 - top2), higher = more confident


@dataclass
class SegmentResult:
    """Result for a single transcription segment."""
    start: float
    end: float
    text: str
    tokens: List[int]
    token_uncertainties: List[TokenUncertainty]

    # Segment-level aggregates
    avg_entropy: float
    avg_margin: float
    max_entropy: float
    min_margin: float

    # Additional metadata
    avg_logprob: float
    no_speech_prob: float
    temperature: float


@dataclass
class TranscriptionResult:
    """Complete transcription result with uncertainty signals."""
    text: str
    segments: List[SegmentResult]
    language: str

    # Utterance-level aggregates
    avg_entropy: float
    avg_margin: float

    # Raw data for further analysis
    all_tokens: List[int] = field(default_factory=list)
    all_token_uncertainties: List[TokenUncertainty] = field(default_factory=list)


def compute_token_uncertainties(
    model: WhisperForConditionalGeneration,
    encoder_outputs: Tensor,
    tokens: Tensor,
    processor: WhisperProcessor,
    sot_len: int,
) -> List[TokenUncertainty]:
    """
    Compute token-level uncertainty signals for a sequence.

    Args:
        model: WhisperForConditionalGeneration model
        encoder_outputs: Encoded audio from model.model.encoder [1, n_ctx, d_model]
        tokens: Token sequence [seq_len]
        processor: WhisperProcessor for decoding tokens
        sot_len: Length of SOT sequence to skip

    Returns:
        List of TokenUncertainty for each non-SOT token
    """
    vocab_size = model.config.vocab_size
    max_entropy = math.log(vocab_size)

    # Forward pass to get logits
    tokens_input = tokens.unsqueeze(0) if tokens.ndim == 1 else tokens
    with torch.no_grad():
        decoder_outputs = model.model.decoder(
            input_ids=tokens_input,
            encoder_hidden_states=encoder_outputs,
            use_cache=False,
            return_dict=True,
        )
        # Get logits through projection layer
        logits = model.proj_out(decoder_outputs.last_hidden_state)  # [1, seq, vocab]

    logits = logits.squeeze(0)  # [seq, vocab]

    uncertainties = []
    text_tokens = tokens[sot_len:].tolist()

    for i, token_id in enumerate(text_tokens):
        pos = sot_len + i
        if pos >= logits.shape[0]:
            break

        token_logits = logits[pos]

        # Compute entropy
        probs = F.softmax(token_logits, dim=-1)
        log_probs = F.log_softmax(token_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        normalized_entropy = entropy / max_entropy

        # Compute margin (top1 - top2)
        top2, _ = token_logits.topk(2)
        margin = (top2[0] - top2[1]).item()

        # Decode token
        token_str = processor.tokenizer.decode([token_id])

        uncertainties.append(TokenUncertainty(
            token_id=token_id,
            token_str=token_str,
            entropy=normalized_entropy,
            margin=margin,
        ))

    return uncertainties


def transcribe_with_uncertainty(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    language: str = "en",
    temperature: float = 0.0,
    num_beams: Optional[int] = 5,
    condition_on_previous_text: bool = False,
    logits_processor=None,
    verbose: bool = False,
    **generate_kwargs,
) -> TranscriptionResult:
    """
    Transcribe audio with token-level uncertainty signals.

    This is a unified transcription function that:
    1. Uses Transformers' generate() for transcription
    2. Outputs token-level predictive entropy and margin
    3. Supports optional logits processor for biasing

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor for audio processing and tokenization
        audio: Audio file path, numpy array, or torch tensor (raw waveform at 16kHz)
        language: Language code (default: "en")
        temperature: Sampling temperature (0 = greedy/beam search)
        num_beams: Beam size for beam search (None or 1 = greedy)
        condition_on_previous_text: Use previous output as prompt
        logits_processor: Optional LogitsProcessor for custom biasing
        verbose: Print progress
        **generate_kwargs: Additional arguments for model.generate()

    Returns:
        TranscriptionResult with text, segments, and uncertainty signals
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Load and preprocess audio
    if isinstance(audio, str):
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        audio = waveform.squeeze().numpy()
    elif isinstance(audio, torch.Tensor):
        audio = audio.numpy()

    # Process audio to get input features
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device).to(dtype)

    # Get decoder prompt IDs for language/task
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe",
        no_timestamps=True,
    )

    # Prepare logits processors
    logits_processor_list = []
    if logits_processor is not None:
        logits_processor_list.append(logits_processor)

    # Generate with output_scores to get logits
    generate_kwargs.setdefault("max_new_tokens", 448)

    outputs = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        num_beams=num_beams if num_beams and num_beams > 1 else 1,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        logits_processor=logits_processor_list if logits_processor_list else None,
        return_dict_in_generate=True,
        output_scores=True,
        **generate_kwargs,
    )

    generated_ids = outputs.sequences
    scores = outputs.scores  # Tuple of logits at each step

    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Get encoder outputs for uncertainty computation
    with torch.no_grad():
        encoder_outputs = model.model.encoder(input_features, return_dict=True)

    # Extract tokens (excluding special tokens at start)
    all_tokens = generated_ids[0].tolist()

    # Find where actual content starts (after forced decoder ids)
    sot_len = len(forced_decoder_ids) + 1 if forced_decoder_ids else 1  # +1 for decoder_start_token
    text_tokens = [t for t in all_tokens[sot_len:] if t != model.config.eos_token_id]

    # Compute uncertainties from scores
    vocab_size = model.config.vocab_size
    max_entropy = math.log(vocab_size)

    token_uncertainties = []
    for i, (score, token_id) in enumerate(zip(scores, text_tokens)):
        if score.ndim > 1:
            score = score[0]  # Take first beam/batch

        # Compute entropy
        probs = F.softmax(score, dim=-1)
        log_probs = F.log_softmax(score, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        normalized_entropy = entropy / max_entropy

        # Compute margin
        top2, _ = score.topk(2)
        margin = (top2[0] - top2[1]).item()

        token_str = processor.tokenizer.decode([token_id])

        token_uncertainties.append(TokenUncertainty(
            token_id=token_id,
            token_str=token_str,
            entropy=normalized_entropy,
            margin=margin,
        ))

    # Compute segment-level stats
    if token_uncertainties:
        entropies = [u.entropy for u in token_uncertainties]
        margins = [u.margin for u in token_uncertainties]
        avg_entropy = float(np.mean(entropies))
        avg_margin = float(np.mean(margins))
        max_entropy_val = float(np.max(entropies))
        min_margin = float(np.min(margins))
    else:
        avg_entropy = avg_margin = max_entropy_val = min_margin = 0.0

    # Create single segment (for simplicity - can be extended for long-form)
    segment = SegmentResult(
        start=0.0,
        end=len(audio) / SAMPLE_RATE,
        text=transcription.strip(),
        tokens=text_tokens,
        token_uncertainties=token_uncertainties,
        avg_entropy=avg_entropy,
        avg_margin=avg_margin,
        max_entropy=max_entropy_val,
        min_margin=min_margin,
        avg_logprob=0.0,  # Would need sequence scores to compute
        no_speech_prob=0.0,
        temperature=temperature,
    )

    if verbose:
        print(f"[0.00s -> {segment.end:.2f}s] {transcription.strip()}")
        print(f"  avg_entropy: {avg_entropy:.4f}, avg_margin: {avg_margin:.2f}")

    return TranscriptionResult(
        text=transcription.strip(),
        segments=[segment],
        language=language,
        avg_entropy=avg_entropy,
        avg_margin=avg_margin,
        all_tokens=text_tokens,
        all_token_uncertainties=token_uncertainties,
    )


def batch_transcribe_with_uncertainty(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_list: List[Union[str, np.ndarray, torch.Tensor]],
    *,
    language: str = "en",
    temperature: float = 0.0,
    num_beams: Optional[int] = 5,
    logits_processors: Optional[List] = None,
    batch_size: int = 8,
    verbose: bool = False,
    **generate_kwargs,
) -> List[TranscriptionResult]:
    """
    Batch transcribe multiple audio samples with token-level uncertainty signals.

    This function processes multiple audio samples efficiently by:
    1. Batching feature extraction
    2. Using batched generation with Transformers

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor for audio processing
        audio_list: List of audio inputs (file paths, numpy arrays, or tensors)
        language: Language code (default: "en")
        temperature: Sampling temperature (0 = greedy/beam search)
        num_beams: Beam size for beam search (None or 1 = greedy)
        logits_processors: Optional list of LogitsProcessor instances,
            one per sample. If None, no custom processing is applied.
        batch_size: Number of samples to process in parallel
        verbose: Print progress
        **generate_kwargs: Additional arguments for model.generate()

    Returns:
        List of TranscriptionResult, one per input audio
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    n_samples = len(audio_list)
    results = []

    # Process in batches
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_audio = audio_list[batch_start:batch_end]
        current_batch_size = len(batch_audio)

        if verbose:
            print(f"Processing batch {batch_start // batch_size + 1}: "
                  f"samples {batch_start + 1}-{batch_end}")

        # Load and preprocess audio
        processed_audio = []
        for audio in batch_audio:
            if isinstance(audio, str):
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio)
                if sample_rate != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
                audio = waveform.squeeze().numpy()
            elif isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            processed_audio.append(audio)

        # Batch process features
        inputs = processor(
            processed_audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(device).to(dtype)

        # Get decoder prompt IDs
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe",
            no_timestamps=True,
        )

        # Generate
        generate_kwargs.setdefault("max_new_tokens", 448)

        outputs = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=num_beams if num_beams and num_beams > 1 else 1,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            return_dict_in_generate=True,
            output_scores=True,
            **generate_kwargs,
        )

        generated_ids = outputs.sequences
        scores = outputs.scores

        # Decode all transcriptions
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Process each sample in batch
        vocab_size = model.config.vocab_size
        max_entropy = math.log(vocab_size)
        sot_len = len(forced_decoder_ids) + 1 if forced_decoder_ids else 1

        for b in range(current_batch_size):
            all_tokens = generated_ids[b].tolist()
            text_tokens = [t for t in all_tokens[sot_len:] if t != model.config.eos_token_id]

            # Compute uncertainties from scores
            token_uncertainties = []
            for i, token_id in enumerate(text_tokens):
                if i >= len(scores):
                    break

                score = scores[i]
                if score.ndim > 1:
                    score = score[b]

                probs = F.softmax(score, dim=-1)
                log_probs = F.log_softmax(score, dim=-1)
                entropy = -torch.sum(probs * log_probs).item()
                normalized_entropy = entropy / max_entropy

                top2, _ = score.topk(2)
                margin = (top2[0] - top2[1]).item()

                token_str = processor.tokenizer.decode([token_id])

                token_uncertainties.append(TokenUncertainty(
                    token_id=token_id,
                    token_str=token_str,
                    entropy=normalized_entropy,
                    margin=margin,
                ))

            # Compute stats
            if token_uncertainties:
                entropies = [u.entropy for u in token_uncertainties]
                margins = [u.margin for u in token_uncertainties]
                avg_entropy = float(np.mean(entropies))
                avg_margin = float(np.mean(margins))
                max_entropy_val = float(np.max(entropies))
                min_margin = float(np.min(margins))
            else:
                avg_entropy = avg_margin = max_entropy_val = min_margin = 0.0

            audio_duration = len(processed_audio[b]) / SAMPLE_RATE

            segment = SegmentResult(
                start=0.0,
                end=audio_duration,
                text=transcriptions[b].strip(),
                tokens=text_tokens,
                token_uncertainties=token_uncertainties,
                avg_entropy=avg_entropy,
                avg_margin=avg_margin,
                max_entropy=max_entropy_val,
                min_margin=min_margin,
                avg_logprob=0.0,
                no_speech_prob=0.0,
                temperature=temperature,
            )

            results.append(TranscriptionResult(
                text=transcriptions[b].strip(),
                segments=[segment],
                language=language,
                avg_entropy=avg_entropy,
                avg_margin=avg_margin,
                all_tokens=text_tokens,
                all_token_uncertainties=token_uncertainties,
            ))

    return results


def print_uncertainty_analysis(result: TranscriptionResult, threshold: float = 0.1):
    """Print detailed uncertainty analysis for a transcription result."""
    print("\n" + "=" * 80)
    print("TRANSCRIPTION WITH UNCERTAINTY ANALYSIS")
    print("=" * 80)

    print(f"\nFull text: {result.text}")
    print(f"\nUtterance-level stats:")
    print(f"  Average entropy: {result.avg_entropy:.4f}")
    print(f"  Average margin:  {result.avg_margin:.2f}")

    print(f"\nSegments: {len(result.segments)}")

    for i, seg in enumerate(result.segments):
        print(f"\n--- Segment {i+1} [{seg.start:.2f}s - {seg.end:.2f}s] ---")
        print(f"Text: {seg.text}")
        print(f"Avg entropy: {seg.avg_entropy:.4f}, Max entropy: {seg.max_entropy:.4f}")
        print(f"Avg margin: {seg.avg_margin:.2f}, Min margin: {seg.min_margin:.2f}")

        # Show high uncertainty tokens
        high_uncertainty = [u for u in seg.token_uncertainties if u.entropy > threshold]
        if high_uncertainty:
            print(f"\nHigh uncertainty tokens (entropy > {threshold}):")
            for u in high_uncertainty:
                print(f"  '{u.token_str}' -> entropy: {u.entropy:.4f}, margin: {u.margin:.2f}")

    print("\n" + "=" * 80)
