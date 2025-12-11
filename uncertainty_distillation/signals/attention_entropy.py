"""
Attention Entropy Signal Extractor

Computes entropy of cross-attention distributions as a proxy for epistemic uncertainty.
High attention entropy indicates the model is uncertain about which audio frames
correspond to which text tokens.

Uses Transformers WhisperForConditionalGeneration API.

Interpretation:
- High entropy → diffuse attention → uncertain alignment → epistemic-like uncertainty
- Low entropy → focused attention → confident alignment
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from transformers import WhisperForConditionalGeneration


@dataclass
class AttentionEntropyResult:
    """Results from attention entropy computation."""

    # Per-token entropy values, shape: [batch, seq_len]
    token_entropy: Tensor

    # Per-layer entropy values, shape: [n_layers, batch, seq_len]
    layer_entropies: Tensor

    # Per-head entropy values (before averaging), shape: [n_layers, batch, n_heads, seq_len]
    head_entropies: Optional[Tensor] = None

    # Utterance-level aggregated entropy, shape: [batch]
    utterance_entropy: Optional[Tensor] = None


class AttentionEntropyExtractor:
    """
    Extract cross-attention entropy from Whisper's decoder.

    This extractor uses output_attentions=True to get attention weights
    from the decoder's cross-attention layers.

    Args:
        model: The WhisperForConditionalGeneration model instance
        n_layers: Number of decoder layers to use (from the end). Default uses last 4.
        aggregate_heads: How to aggregate across attention heads ('mean', 'max', 'min')
        return_head_entropies: Whether to return per-head entropy values
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        n_layers: int = 4,
        aggregate_heads: str = "mean",
        return_head_entropies: bool = False,
    ):
        self.model = model
        self.n_layers = min(n_layers, model.config.decoder_layers)
        self.aggregate_heads = aggregate_heads
        self.return_head_entropies = return_head_entropies

    @staticmethod
    def compute_entropy(
        attention_weights: Tensor,
        eps: float = 1e-10,
    ) -> Tensor:
        """
        Compute entropy of attention distributions.

        Args:
            attention_weights: Attention probabilities, shape [..., seq_len, audio_frames]
            eps: Small constant for numerical stability

        Returns:
            Entropy values, shape [..., seq_len]
        """
        # Ensure probabilities sum to 1
        attn_probs = attention_weights / (
            attention_weights.sum(dim=-1, keepdim=True) + eps
        )

        # Compute entropy: H = -sum(p * log(p))
        log_probs = torch.log(attn_probs + eps)
        entropy = -torch.sum(attn_probs * log_probs, dim=-1)

        return entropy

    @staticmethod
    def normalize_entropy(entropy: Tensor, n_audio_frames: int) -> Tensor:
        """
        Normalize entropy by maximum possible entropy (log of audio frames).

        This gives values in [0, 1] where 1 = uniform attention (maximum uncertainty).

        Args:
            entropy: Raw entropy values
            n_audio_frames: Number of audio frames (determines max entropy)

        Returns:
            Normalized entropy in [0, 1]
        """
        max_entropy = math.log(n_audio_frames)
        return entropy / max_entropy if max_entropy > 0 else entropy

    def _aggregate_head_entropies(
        self,
        head_entropies: Tensor
    ) -> Tensor:
        """
        Aggregate entropy values across attention heads.

        Args:
            head_entropies: Shape [batch, n_heads, seq_len]

        Returns:
            Aggregated values, shape [batch, seq_len]
        """
        if self.aggregate_heads == "mean":
            return head_entropies.mean(dim=1)
        elif self.aggregate_heads == "max":
            return head_entropies.max(dim=1)[0]
        elif self.aggregate_heads == "min":
            return head_entropies.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregate_heads}")

    @torch.no_grad()
    def extract(
        self,
        input_features: Tensor,
        tokens: Tensor,
        normalize: bool = True,
    ) -> AttentionEntropyResult:
        """
        Extract attention entropy for given audio features and text tokens.

        Args:
            input_features: Input features from processor, shape [batch, n_mels, n_frames]
                or [n_mels, n_frames]
            tokens: Decoder input tokens, shape [batch, seq_len] or [seq_len]
            normalize: Whether to normalize entropy to [0, 1]

        Returns:
            AttentionEntropyResult containing entropy values at various granularities
        """
        # Handle single-sample inputs
        single_batch = input_features.ndim == 2
        if single_batch:
            input_features = input_features.unsqueeze(0)

        # Handle 1D token tensor (unbatched)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
            single_batch = True  # Also flag for output squeeze

        # Forward pass with attention output
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=tokens,
            output_attentions=True,
            return_dict=True,
        )

        # Get cross-attention weights from last n_layers
        # cross_attentions is tuple of (batch, n_heads, seq_len, encoder_seq_len)
        cross_attentions = outputs.cross_attentions

        if cross_attentions is None:
            raise RuntimeError(
                "No cross-attention weights returned. "
                "Ensure output_attentions=True is supported."
            )

        # Use last n_layers
        n_total_layers = len(cross_attentions)
        start_layer = max(0, n_total_layers - self.n_layers)
        selected_attentions = cross_attentions[start_layer:]

        layer_entropies = []
        all_head_entropies = []
        n_audio_frames = selected_attentions[0].shape[-1]

        for attn_weights in selected_attentions:
            # attn_weights shape: [batch, heads, seq, audio_frames]

            # Compute entropy per head
            # Result shape: [batch, heads, seq]
            head_entropy = self.compute_entropy(attn_weights)

            if normalize:
                head_entropy = self.normalize_entropy(head_entropy, n_audio_frames)

            all_head_entropies.append(head_entropy)

            # Aggregate across heads
            # Result shape: [batch, seq]
            aggregated = self._aggregate_head_entropies(head_entropy)
            layer_entropies.append(aggregated)

        # Stack layer entropies: [n_layers, batch, seq]
        layer_entropies = torch.stack(layer_entropies, dim=0)

        # Average across layers for final token-level entropy: [batch, seq]
        token_entropy = layer_entropies.mean(dim=0)

        # Compute utterance-level entropy (mean across tokens): [batch]
        utterance_entropy = token_entropy.mean(dim=-1)

        # Prepare head entropies if requested
        head_entropies_result = None
        if self.return_head_entropies:
            # Stack: [n_layers, batch, heads, seq]
            head_entropies_result = torch.stack(all_head_entropies, dim=0)

        result = AttentionEntropyResult(
            token_entropy=token_entropy.squeeze(0) if single_batch else token_entropy,
            layer_entropies=layer_entropies.squeeze(1) if single_batch else layer_entropies,
            head_entropies=head_entropies_result,
            utterance_entropy=utterance_entropy.squeeze(0) if single_batch else utterance_entropy,
        )

        return result

    @torch.no_grad()
    def extract_from_encoder_outputs(
        self,
        encoder_outputs: Tensor,
        tokens: Tensor,
        normalize: bool = True,
    ) -> AttentionEntropyResult:
        """
        Extract attention entropy from pre-computed encoder outputs.

        This is more efficient when encoder outputs are already available
        (e.g., from a transcription pass).

        Args:
            encoder_outputs: Encoded audio, shape [batch, n_ctx, d_model]
            tokens: Decoder input tokens, shape [batch, seq_len]
            normalize: Whether to normalize entropy to [0, 1]

        Returns:
            AttentionEntropyResult containing entropy values
        """
        single_batch = encoder_outputs.ndim == 2
        if single_batch:
            encoder_outputs = encoder_outputs.unsqueeze(0)
            tokens = tokens.unsqueeze(0)

        # Run decoder with encoder outputs
        decoder_outputs = self.model.model.decoder(
            input_ids=tokens,
            encoder_hidden_states=encoder_outputs,
            output_attentions=True,
            return_dict=True,
        )

        cross_attentions = decoder_outputs.cross_attentions

        if cross_attentions is None:
            raise RuntimeError("No cross-attention weights returned.")

        # Use last n_layers
        n_total_layers = len(cross_attentions)
        start_layer = max(0, n_total_layers - self.n_layers)
        selected_attentions = cross_attentions[start_layer:]

        layer_entropies = []
        all_head_entropies = []
        n_audio_frames = selected_attentions[0].shape[-1]

        for attn_weights in selected_attentions:
            head_entropy = self.compute_entropy(attn_weights)

            if normalize:
                head_entropy = self.normalize_entropy(head_entropy, n_audio_frames)

            all_head_entropies.append(head_entropy)
            aggregated = self._aggregate_head_entropies(head_entropy)
            layer_entropies.append(aggregated)

        layer_entropies = torch.stack(layer_entropies, dim=0)
        token_entropy = layer_entropies.mean(dim=0)
        utterance_entropy = token_entropy.mean(dim=-1)

        head_entropies_result = None
        if self.return_head_entropies:
            head_entropies_result = torch.stack(all_head_entropies, dim=0)

        result = AttentionEntropyResult(
            token_entropy=token_entropy.squeeze(0) if single_batch else token_entropy,
            layer_entropies=layer_entropies.squeeze(1) if single_batch else layer_entropies,
            head_entropies=head_entropies_result,
            utterance_entropy=utterance_entropy.squeeze(0) if single_batch else utterance_entropy,
        )

        return result
