"""
Training-Free Uncertainty Signal Extractors (Stage 1)

This module provides signal extractors for computing uncertainty estimates
from frozen Whisper models without requiring any additional training.

Uses the Transformers WhisperForConditionalGeneration API for all model
interactions, with support for batch processing.

Five complementary signals:
1. Attention Entropy - epistemic-like (model confusion about alignment)
2. N-best Disagreement - aleatoric-like (inherent audio ambiguity)
3. Feature Distance - epistemic (OOD detection via Mahalanobis)
4. Predictive Entropy - mixed (general token uncertainty)
5. GradNorm - epistemic (OOD detection via gradient magnitude)
"""

from .attention_entropy import AttentionEntropyExtractor, AttentionEntropyResult
from .nbest_disagreement import NBestDisagreementExtractor, NBestDisagreementResult
from .feature_distance import (
    FeatureDistanceExtractor,
    FeatureDistanceResult,
    ReferenceStats,
)
from .predictive_entropy import PredictiveEntropyExtractor, PredictiveEntropyResult
from .gradnorm import GradNormExtractor, GradNormResult
from .signal_ensemble import SignalEnsemble, UncertaintySignals

__all__ = [
    # Signal 1: Attention Entropy
    "AttentionEntropyExtractor",
    "AttentionEntropyResult",
    # Signal 2: N-best Disagreement
    "NBestDisagreementExtractor",
    "NBestDisagreementResult",
    # Signal 3: Feature Distance
    "FeatureDistanceExtractor",
    "FeatureDistanceResult",
    "ReferenceStats",
    # Signal 4: Predictive Entropy
    "PredictiveEntropyExtractor",
    "PredictiveEntropyResult",
    # Signal 5: GradNorm
    "GradNormExtractor",
    "GradNormResult",
    # Ensemble
    "SignalEnsemble",
    "UncertaintySignals",
]
