"""
Feature-Space Distance Signal Extractor (Signal 3)

Computes Mahalanobis distance from a reference distribution as a proxy for
epistemic uncertainty. Inputs far from the training distribution indicate
out-of-distribution samples where the model has limited knowledge.

Uses Transformers WhisperForConditionalGeneration API.

From the research plan:
    "High Mahalanobis distance → input unlike training data
     → epistemic (knowledge gap)"

Requirements:
    pip install scikit-learn

Interpretation:
- High distance → OOD input → epistemic uncertainty (model doesn't know)
- Low distance → in-distribution → model should be confident
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet

from transformers import WhisperForConditionalGeneration


@dataclass
class FeatureDistanceResult:
    """Results from feature distance computation."""

    # Mahalanobis distance per sample, shape: [batch] or scalar
    mahalanobis_distance: Tensor

    # Euclidean distance for comparison, shape: [batch] or scalar
    euclidean_distance: Tensor

    # Cosine distance from mean, shape: [batch] or scalar
    cosine_distance: Tensor

    # Raw features used, shape: [batch, hidden_dim]
    features: Optional[Tensor] = None


@dataclass
class ReferenceStats:
    """Reference distribution statistics for Mahalanobis distance."""

    mean: Tensor  # [hidden_dim]
    precision: Tensor  # [hidden_dim, hidden_dim] - inverse covariance
    covariance: Optional[Tensor] = None  # [hidden_dim, hidden_dim]

    def save(self, path: Union[str, Path]):
        """Save statistics to file."""
        torch.save(
            {
                "mean": self.mean,
                "precision": self.precision,
                "covariance": self.covariance,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "ReferenceStats":
        """Load statistics from file."""
        data = torch.load(path, map_location=device, weights_only=True)
        return cls(
            mean=data["mean"],
            precision=data["precision"],
            covariance=data.get("covariance"),
        )


def compute_feature_distance(
    model: WhisperForConditionalGeneration,
    input_features: Tensor,
    reference_stats: ReferenceStats,
) -> Tensor:
    """
    Compute Mahalanobis distance from reference distribution.

    Simple functional interface matching the research plan pseudocode.

    Args:
        model: WhisperForConditionalGeneration model
        input_features: Input features from processor
        reference_stats: Pre-computed reference distribution statistics

    Returns:
        Mahalanobis distance, shape [batch] or scalar
    """
    extractor = FeatureDistanceExtractor(model, reference_stats)
    result = extractor.extract(input_features)
    return result.mahalanobis_distance


class FeatureDistanceExtractor:
    """
    Extract feature-space distance from Whisper encoder representations.

    Computes Mahalanobis distance using pre-computed reference statistics
    from a representative dataset (e.g., LibriSpeech).

    Args:
        model: The WhisperForConditionalGeneration model instance
        reference_stats: Pre-computed mean and precision matrix, or None to compute later
        pooling: How to pool encoder outputs ('mean', 'max', 'cls')
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        reference_stats: Optional[ReferenceStats] = None,
        pooling: str = "mean",
    ):
        self.model = model
        self.reference_stats = reference_stats
        self.pooling = pooling
        self.hidden_dim = model.config.d_model

    def set_reference_stats(self, stats: ReferenceStats):
        """Set or update reference statistics."""
        self.reference_stats = stats

    def _pool_features(self, encoder_output: Tensor) -> Tensor:
        """
        Pool encoder output to utterance-level features.

        Args:
            encoder_output: Shape [batch, seq_len, hidden_dim]

        Returns:
            Pooled features, shape [batch, hidden_dim]
        """
        if self.pooling == "mean":
            return encoder_output.mean(dim=1)
        elif self.pooling == "max":
            return encoder_output.max(dim=1)[0]
        elif self.pooling == "cls":
            # Use first token as CLS
            return encoder_output[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    @staticmethod
    def compute_mahalanobis(
        features: Tensor,
        mean: Tensor,
        precision: Tensor,
    ) -> Tensor:
        """
        Compute Mahalanobis distance from reference distribution.

        Implements: sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))

        Args:
            features: Sample features, shape [batch, hidden_dim]
            mean: Reference mean, shape [hidden_dim]
            precision: Inverse covariance (precision matrix), shape [hidden_dim, hidden_dim]

        Returns:
            Mahalanobis distance, shape [batch]
        """
        diff = features - mean  # [batch, hidden_dim]

        # Compute (x - μ)ᵀ Σ⁻¹ (x - μ)
        # = diff @ precision @ diff.T for each sample
        left = diff @ precision  # [batch, hidden_dim]
        mahal_sq = torch.sum(left * diff, dim=-1)  # [batch]

        # Clamp for numerical stability
        mahal_sq = torch.clamp(mahal_sq, min=0.0)

        return torch.sqrt(mahal_sq)

    @staticmethod
    def compute_euclidean(features: Tensor, mean: Tensor) -> Tensor:
        """Compute Euclidean distance from mean."""
        diff = features - mean
        return torch.norm(diff, dim=-1)

    @staticmethod
    def compute_cosine_distance(features: Tensor, mean: Tensor) -> Tensor:
        """Compute cosine distance (1 - cosine similarity) from mean."""
        features_norm = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-8)
        mean_norm = mean / (torch.norm(mean) + 1e-8)
        similarity = torch.sum(features_norm * mean_norm, dim=-1)
        return 1.0 - similarity

    @torch.no_grad()
    def extract(
        self,
        input_features: Tensor,
        return_features: bool = False,
    ) -> FeatureDistanceResult:
        """
        Extract feature distance for given audio features.

        Implements Signal 3 from the research plan.

        Args:
            input_features: Input features from processor, shape [batch, n_mels, n_frames]
                or [n_mels, n_frames]
            return_features: Whether to return extracted features

        Returns:
            FeatureDistanceResult containing distance values
        """
        if self.reference_stats is None:
            raise ValueError(
                "Reference statistics not set. Call set_reference_stats() or "
                "compute_reference_stats() first."
            )

        single_batch = input_features.ndim == 2
        if single_batch:
            input_features = input_features.unsqueeze(0)

        # Extract encoder representations
        encoder_outputs = self.model.model.encoder(
            input_features,
            return_dict=True,
        )
        encoder_out = encoder_outputs.last_hidden_state  # [batch, seq, hidden]

        # Pool to utterance-level
        features = self._pool_features(encoder_out)  # [batch, hidden]

        # Ensure stats are on same device
        mean = self.reference_stats.mean.to(features.device)
        precision = self.reference_stats.precision.to(features.device)

        # Compute utterance-level distances
        mahal_dist = self.compute_mahalanobis(features, mean, precision)
        eucl_dist = self.compute_euclidean(features, mean)
        cos_dist = self.compute_cosine_distance(features, mean)

        if single_batch:
            mahal_dist = mahal_dist.squeeze(0)
            eucl_dist = eucl_dist.squeeze(0)
            cos_dist = cos_dist.squeeze(0)
            features = features.squeeze(0) if return_features else None

        return FeatureDistanceResult(
            mahalanobis_distance=mahal_dist,
            euclidean_distance=eucl_dist,
            cosine_distance=cos_dist,
            features=features if return_features else None,
        )

    @torch.no_grad()
    def extract_features(self, input_features: Tensor) -> Tensor:
        """
        Extract pooled encoder features without computing distances.

        Useful for building reference statistics.

        Args:
            input_features: Input features from processor

        Returns:
            Pooled features, shape [batch, hidden_dim]
        """
        single_batch = input_features.ndim == 2
        if single_batch:
            input_features = input_features.unsqueeze(0)

        encoder_outputs = self.model.model.encoder(
            input_features,
            return_dict=True,
        )
        encoder_out = encoder_outputs.last_hidden_state
        features = self._pool_features(encoder_out)

        if single_batch:
            features = features.squeeze(0)

        return features

    def compute_reference_stats(
        self,
        features: Tensor,
        method: str = "ledoit_wolf",
    ) -> ReferenceStats:
        """
        Compute reference statistics from a collection of features.

        Uses scikit-learn covariance estimators for robust estimation.

        Args:
            features: Features from reference dataset, shape [n_samples, hidden_dim]
            method: Covariance estimation method:
                - 'empirical': Standard MLE covariance (needs n_samples > hidden_dim)
                - 'ledoit_wolf': Shrinkage estimator (recommended for high-dim)
                - 'mincovdet': Robust to outliers (needs n_samples > hidden_dim)

        Returns:
            ReferenceStats with mean and precision matrix
        """
        import warnings

        features_np = features.cpu().numpy()
        n_samples, hidden_dim = features_np.shape

        # Warn if sample size is too small
        if n_samples <= hidden_dim:
            if method == "empirical":
                warnings.warn(
                    f"n_samples ({n_samples}) <= hidden_dim ({hidden_dim}). "
                    f"Empirical covariance will be singular. "
                    f"Switching to 'ledoit_wolf' for shrinkage estimation.",
                    UserWarning
                )
                method = "ledoit_wolf"
            elif method == "mincovdet":
                warnings.warn(
                    f"n_samples ({n_samples}) <= hidden_dim ({hidden_dim}). "
                    f"MinCovDet requires more samples. "
                    f"Switching to 'ledoit_wolf' for shrinkage estimation.",
                    UserWarning
                )
                method = "ledoit_wolf"

        if n_samples < 10:
            warnings.warn(
                f"Very few samples ({n_samples}) for computing reference statistics. "
                f"Mahalanobis distances may be unreliable. Consider using more reference data.",
                UserWarning
            )

        # Compute mean
        mean = features_np.mean(axis=0)

        # Estimate covariance using sklearn
        if method == "empirical":
            estimator = EmpiricalCovariance()
        elif method == "ledoit_wolf":
            estimator = LedoitWolf()
        elif method == "mincovdet":
            estimator = MinCovDet()
        else:
            raise ValueError(f"Unknown covariance method: {method}")

        estimator.fit(features_np)
        covariance = estimator.covariance_
        precision = estimator.precision_

        # Convert to torch
        stats = ReferenceStats(
            mean=torch.tensor(mean, dtype=torch.float32),
            precision=torch.tensor(precision, dtype=torch.float32),
            covariance=torch.tensor(covariance, dtype=torch.float32),
        )

        self.reference_stats = stats
        return stats

    @classmethod
    def from_reference_data(
        cls,
        model: WhisperForConditionalGeneration,
        reference_features: Tensor,
        method: str = "ledoit_wolf",
        batch_size: int = 32,
    ) -> "FeatureDistanceExtractor":
        """
        Create extractor with reference stats computed from data.

        Args:
            model: WhisperForConditionalGeneration model
            reference_features: Reference input features, shape [n_samples, n_mels, n_frames]
            method: Covariance estimation method
            batch_size: Batch size for feature extraction

        Returns:
            Configured FeatureDistanceExtractor
        """
        extractor = cls(model)

        # Extract features in batches
        all_features = []
        n_samples = reference_features.shape[0]

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = reference_features[i : i + batch_size]
                features = extractor.extract_features(batch)
                all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0)

        # Compute reference stats
        extractor.compute_reference_stats(all_features, method=method)

        return extractor
