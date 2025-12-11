"""
Uncertainty Distillation for Safety-Critical ASR

This module implements training-free uncertainty signals that can be distilled
into lightweight learned heads for safety-critical speech recognition.

Uses the Transformers WhisperForConditionalGeneration API for model loading
and inference, with support for batch processing.

Stage 1: Extract Training-Free Uncertainty Signals
- Attention Entropy: Cross-attention focus diffusion (epistemic-like)
- N-best Disagreement: Hypothesis diversity (aleatoric-like)
- Feature Distance: Mahalanobis OOD detection (epistemic)
- Predictive Entropy: Token-level uncertainty (mixed)

Usage:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from uncertainty_distillation import SignalEnsemble, ReferenceStats

    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")

    # Create ensemble extractor
    ensemble = SignalEnsemble(model, processor, language="en")

    # Optionally set reference stats for feature distance
    # stats = ReferenceStats.load("reference_stats.pt")
    # ensemble.set_reference_stats(stats)

    # Extract all signals
    signals = ensemble.extract_all(input_features, tokens)

    # Access individual signals
    attn_entropy = signals.attention_entropy.token_entropy
    nbest_disagree = signals.nbest_disagreement.utterance_disagreement
    feature_dist = signals.feature_distance.mahalanobis_distance
    pred_entropy = signals.predictive_entropy.normalized_entropy
"""

from .signals import (
    # Signal extractors
    AttentionEntropyExtractor,
    NBestDisagreementExtractor,
    FeatureDistanceExtractor,
    PredictiveEntropyExtractor,
    # Result types
    AttentionEntropyResult,
    NBestDisagreementResult,
    FeatureDistanceResult,
    PredictiveEntropyResult,
    # Reference stats
    ReferenceStats,
    # Ensemble
    SignalEnsemble,
    UncertaintySignals,
)

__all__ = [
    # Signal extractors
    "AttentionEntropyExtractor",
    "NBestDisagreementExtractor",
    "FeatureDistanceExtractor",
    "PredictiveEntropyExtractor",
    # Result types
    "AttentionEntropyResult",
    "NBestDisagreementResult",
    "FeatureDistanceResult",
    "PredictiveEntropyResult",
    # Reference stats
    "ReferenceStats",
    # Ensemble
    "SignalEnsemble",
    "UncertaintySignals",
]
