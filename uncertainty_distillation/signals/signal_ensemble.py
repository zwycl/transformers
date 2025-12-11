"""
Signal Ensemble Module

Combines all four training-free uncertainty signals into a unified interface.
This module provides the complete Stage 1 pipeline for extracting uncertainty
signals from frozen Whisper models.

Uses Transformers WhisperForConditionalGeneration API.

Signals:
1. Attention Entropy - epistemic-like (model confusion)
2. N-best Disagreement - aleatoric-like (inherent ambiguity)
3. Feature Distance - epistemic (OOD detection)
4. Predictive Entropy - mixed (general uncertainty)
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .attention_entropy import AttentionEntropyExtractor, AttentionEntropyResult
from .nbest_disagreement import NBestDisagreementExtractor, NBestDisagreementResult
from .feature_distance import (
    FeatureDistanceExtractor,
    FeatureDistanceResult,
    ReferenceStats,
)
from .predictive_entropy import PredictiveEntropyExtractor, PredictiveEntropyResult


@dataclass
class UncertaintySignals:
    """
    Container for all uncertainty signals from a single input.

    All signals are token-level where applicable, with utterance-level
    summaries also provided.
    """

    # Signal 1: Attention entropy (epistemic-like)
    attention_entropy: AttentionEntropyResult

    # Signal 2: N-best disagreement (aleatoric-like)
    nbest_disagreement: NBestDisagreementResult

    # Signal 3: Feature distance (epistemic)
    feature_distance: FeatureDistanceResult

    # Signal 4: Predictive entropy (mixed)
    predictive_entropy: PredictiveEntropyResult

    # Convenience accessors for utterance-level values
    @property
    def utterance_attention_entropy(self) -> Tensor:
        return self.attention_entropy.utterance_entropy

    @property
    def utterance_nbest_disagreement(self) -> Tensor:
        return self.nbest_disagreement.utterance_disagreement

    @property
    def utterance_feature_distance(self) -> Tensor:
        return self.feature_distance.mahalanobis_distance

    @property
    def utterance_predictive_entropy(self) -> Tensor:
        return self.predictive_entropy.utterance_entropy

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "attention_entropy": {
                "token": self.attention_entropy.token_entropy,
                "utterance": self.attention_entropy.utterance_entropy,
            },
            "nbest_disagreement": {
                "position": self.nbest_disagreement.position_disagreement,
                "utterance": self.nbest_disagreement.utterance_disagreement,
            },
            "feature_distance": {
                "mahalanobis": self.feature_distance.mahalanobis_distance,
                "euclidean": self.feature_distance.euclidean_distance,
                "cosine": self.feature_distance.cosine_distance,
            },
            "predictive_entropy": {
                "token": self.predictive_entropy.token_entropy,
                "normalized": self.predictive_entropy.normalized_entropy,
                "utterance": self.predictive_entropy.utterance_entropy,
            },
        }


class SignalEnsemble:
    """
    Unified extractor for all training-free uncertainty signals.

    This class orchestrates the extraction of all four signals defined
    in Stage 1 of the research plan.

    Args:
        model: The WhisperForConditionalGeneration model instance
        processor: WhisperProcessor for tokenization and decoding
        reference_stats: Pre-computed reference statistics for feature distance
        n_attention_layers: Number of decoder layers for attention entropy (default: 4)
        n_beams: Number of beams for N-best disagreement (default: 5)
        language: Language for transcription (default: "en")
        attention_aggregate: How to aggregate attention entropy across heads ('mean', 'max', 'min')
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        reference_stats: Optional[ReferenceStats] = None,
        n_attention_layers: int = 4,
        n_beams: int = 5,
        language: str = "en",
        attention_aggregate: str = "mean",
    ):
        self.model = model
        self.processor = processor
        self.language = language

        # Initialize all extractors
        self.attention_extractor = AttentionEntropyExtractor(
            model, n_layers=n_attention_layers, aggregate_heads=attention_aggregate
        )

        self.nbest_extractor = NBestDisagreementExtractor(
            model, processor, n_beams=n_beams, language=language
        )

        self.feature_extractor = FeatureDistanceExtractor(
            model, reference_stats=reference_stats
        )

        self.entropy_extractor = PredictiveEntropyExtractor(model)

    def set_reference_stats(self, stats: ReferenceStats):
        """Set or update reference statistics for feature distance."""
        self.feature_extractor.set_reference_stats(stats)

    @torch.no_grad()
    def extract_all(
        self,
        input_features: Tensor,
        tokens: Tensor,
        skip_nbest: bool = False,
        skip_feature_distance: bool = False,
        skip_attention_entropy: bool = False,
    ) -> UncertaintySignals:
        """
        Extract all uncertainty signals for given audio features and tokens.

        Args:
            input_features: Input features from processor, shape [batch, n_mels, n_frames]
                or [n_mels, n_frames]
            tokens: Decoder input tokens, shape [batch, seq_len] or [seq_len]
            skip_nbest: Skip N-best computation (slow due to beam search)
            skip_feature_distance: Skip feature distance (requires reference stats)
            skip_attention_entropy: Skip attention entropy (requires eager attention implementation)

        Returns:
            UncertaintySignals containing all four signal types
        """
        # Signal 1: Attention entropy (requires eager attention, not SDPA)
        if skip_attention_entropy:
            single = input_features.ndim == 2
            seq_len = tokens.shape[-1] if tokens.ndim > 1 else tokens.shape[0]
            attention_result = AttentionEntropyResult(
                token_entropy=torch.zeros(seq_len) if single else torch.zeros(input_features.shape[0], seq_len),
                layer_entropies=torch.zeros(1, seq_len) if single else torch.zeros(1, input_features.shape[0], seq_len),
                head_entropies=None,
                utterance_entropy=torch.tensor(0.0) if single else torch.zeros(input_features.shape[0]),
            )
        else:
            attention_result = self.attention_extractor.extract(input_features, tokens)

        # Signal 2: N-best disagreement (optional, slow)
        if skip_nbest:
            # Return placeholder
            single = input_features.ndim == 2
            nbest_result = NBestDisagreementResult(
                position_disagreement=torch.tensor([0.0]) if single else torch.zeros(input_features.shape[0], 1),
                utterance_disagreement=torch.tensor(0.0) if single else torch.zeros(input_features.shape[0]),
                hypotheses=[],
            )
        else:
            nbest_result = self.nbest_extractor.extract(input_features)

        # Signal 3: Feature distance (optional, requires reference)
        if skip_feature_distance or self.feature_extractor.reference_stats is None:
            single = input_features.ndim == 2
            feature_result = FeatureDistanceResult(
                mahalanobis_distance=torch.tensor(0.0) if single else torch.zeros(input_features.shape[0]),
                euclidean_distance=torch.tensor(0.0) if single else torch.zeros(input_features.shape[0]),
                cosine_distance=torch.tensor(0.0) if single else torch.zeros(input_features.shape[0]),
            )
        else:
            feature_result = self.feature_extractor.extract(input_features)

        # Signal 4: Predictive entropy
        entropy_result = self.entropy_extractor.extract(input_features, tokens)

        return UncertaintySignals(
            attention_entropy=attention_result,
            nbest_disagreement=nbest_result,
            feature_distance=feature_result,
            predictive_entropy=entropy_result,
        )

    @torch.no_grad()
    def extract_fast(
        self,
        input_features: Tensor,
        tokens: Tensor,
    ) -> UncertaintySignals:
        """
        Fast extraction skipping slow components (N-best, feature distance).

        Useful for real-time applications where only attention and predictive
        entropy are needed.

        Args:
            input_features: Input features from processor
            tokens: Decoder input tokens

        Returns:
            UncertaintySignals (nbest and feature distance will be placeholders)
        """
        return self.extract_all(
            input_features, tokens, skip_nbest=True, skip_feature_distance=True
        )

    def compute_reference_stats_from_data(
        self,
        reference_features: Tensor,
        method: str = "ledoit_wolf",
        batch_size: int = 32,
        compute_decoder_stats: bool = True,
    ) -> ReferenceStats:
        """
        Compute reference statistics from a dataset of input features.

        Args:
            reference_features: Reference data, shape [n_samples, n_mels, n_frames]
            method: Covariance estimation method
            batch_size: Batch size for feature extraction
            compute_decoder_stats: Whether to also compute decoder-level stats
                for token-level feature distances (requires transcribing each sample)

        Returns:
            Computed ReferenceStats (also set on feature extractor)
        """
        all_encoder_features = []
        all_decoder_features = []
        n_samples = reference_features.shape[0]

        # Get decoder prompt IDs
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language,
            task="transcribe",
            no_timestamps=True,
        )

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = reference_features[i : i + batch_size]
                features = self.feature_extractor.extract_features(batch)
                all_encoder_features.append(features.cpu())

            # Compute decoder stats if requested
            if compute_decoder_stats:
                for i in range(n_samples):
                    sample_features = reference_features[i:i+1]

                    # Transcribe to get tokens
                    # Use max_new_tokens that accounts for decoder prompt length
                    outputs = self.model.generate(
                        sample_features,
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=440,  # Leave room for decoder prompt tokens
                        return_dict_in_generate=True,
                    )

                    tokens = outputs.sequences

                    # Extract decoder features
                    decoder_feats = self._extract_decoder_features(sample_features, tokens)
                    all_decoder_features.append(decoder_feats.cpu())

        all_encoder_features = torch.cat(all_encoder_features, dim=0)
        stats = self.feature_extractor.compute_reference_stats(all_encoder_features, method)

        # Add decoder stats if computed
        if compute_decoder_stats and all_decoder_features:
            all_decoder_features = torch.cat(all_decoder_features, dim=0)
            decoder_mean = all_decoder_features.mean(dim=0)

            # Use same method for decoder covariance
            from sklearn.covariance import LedoitWolf, EmpiricalCovariance

            decoder_np = all_decoder_features.numpy()
            if method == "ledoit_wolf" or decoder_np.shape[0] <= decoder_np.shape[1]:
                estimator = LedoitWolf()
            else:
                estimator = EmpiricalCovariance()

            estimator.fit(decoder_np)
            decoder_precision = torch.tensor(estimator.precision_, dtype=torch.float32)

            stats.decoder_mean = decoder_mean
            stats.decoder_precision = decoder_precision

        return stats

    def _extract_decoder_features(self, input_features: Tensor, tokens: Tensor) -> Tensor:
        """Extract decoder hidden states for reference stats."""
        if input_features.ndim == 2:
            input_features = input_features.unsqueeze(0)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        # Encode audio
        encoder_outputs = self.model.model.encoder(
            input_features,
            return_dict=True,
        )

        # Get decoder hidden states
        hidden_states = []

        def hook(module, input, output):
            hidden_states.append(output[0].detach())

        last_block = self.model.model.decoder.layers[-1]
        handle = last_block.register_forward_hook(hook)

        try:
            _ = self.model.model.decoder(
                input_ids=tokens,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                return_dict=True,
            )
            if hidden_states:
                # Shape: [batch, seq_len, hidden_dim]
                # Flatten to [seq_len, hidden_dim] for this sample
                return hidden_states[0].squeeze(0)
            else:
                raise RuntimeError("Failed to capture decoder hidden states")
        finally:
            handle.remove()
