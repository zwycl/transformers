"""
Predictive Entropy Signal Extractor (Signal 4)

Computes entropy over the vocabulary distribution at each decoding step.
High entropy indicates uncertainty about the next token prediction.

Uses Transformers WhisperForConditionalGeneration API.

From the research plan:
    "High predictive entropy → uncertain word choice
     → mixed (could be either type)"

Interpretation:
- High entropy → uncertain about next token → mixed epistemic/aleatoric
- Low entropy → confident prediction
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import WhisperForConditionalGeneration


@dataclass
class PredictiveEntropyResult:
    """Results from predictive entropy computation."""

    # Per-token entropy, shape: [batch, seq_len] or [seq_len]
    token_entropy: Tensor

    # Normalized entropy (by log vocab size), shape: [batch, seq_len] or [seq_len]
    normalized_entropy: Tensor

    # Utterance-level entropy (mean), shape: [batch] or scalar
    utterance_entropy: Tensor

    # Top-k probability mass (confidence measure), shape: [batch, seq_len] or [seq_len]
    top_k_mass: Optional[Tensor] = None

    # Difference between top-2 logits (margin), shape: [batch, seq_len] or [seq_len]
    # Lower margin = more uncertain (close competition between top choices)
    logit_margin: Optional[Tensor] = None

    # Utterance-level margin (mean), shape: [batch] or scalar
    utterance_margin: Optional[Tensor] = None


def compute_predictive_entropy(
    model: WhisperForConditionalGeneration,
    input_features: Tensor,
    tokens: Tensor,
) -> Tensor:
    """
    Compute predictive entropy over vocabulary.

    Simple functional interface matching the research plan pseudocode.

    Args:
        model: WhisperForConditionalGeneration model
        input_features: Input features from processor
        tokens: Decoder input tokens

    Returns:
        Normalized entropy per token, shape [batch, seq_len] or [seq_len]
    """
    extractor = PredictiveEntropyExtractor(model)
    result = extractor.extract(input_features, tokens)
    return result.normalized_entropy


class PredictiveEntropyExtractor:
    """
    Extract predictive entropy from Whisper's decoder output distribution.

    Computes entropy of the softmax distribution over vocabulary at each
    decoding position.

    Args:
        model: The WhisperForConditionalGeneration model instance
        top_k: Number of top tokens to track for confidence measure
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        top_k: int = 10,
    ):
        self.model = model
        self.top_k = top_k
        self.vocab_size = model.config.vocab_size

    @staticmethod
    def compute_entropy(logits: Tensor, eps: float = 1e-10) -> Tensor:
        """
        Compute entropy of probability distribution from logits.

        Implements: H = -sum(p * log(p))

        Args:
            logits: Logits over vocabulary, shape [..., vocab_size]
            eps: Small constant for numerical stability

        Returns:
            Entropy values, shape [...]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # H = -sum(p * log(p))
        entropy = -torch.sum(probs * log_probs, dim=-1)

        return entropy

    def normalize_entropy(self, entropy: Tensor) -> Tensor:
        """
        Normalize entropy by maximum possible entropy (log vocab size).

        From research plan:
            normalized_entropy = entropy / math.log(model.config.vocab_size)

        Args:
            entropy: Raw entropy values

        Returns:
            Normalized entropy in [0, 1]
        """
        max_entropy = math.log(self.vocab_size)
        return entropy / max_entropy

    @staticmethod
    def compute_top_k_mass(logits: Tensor, k: int) -> Tensor:
        """
        Compute probability mass in top-k tokens.

        High mass = confident, low mass = uncertain (spread out distribution).

        Args:
            logits: Logits over vocabulary
            k: Number of top tokens

        Returns:
            Sum of top-k probabilities
        """
        probs = F.softmax(logits, dim=-1)
        top_k_probs, _ = probs.topk(k, dim=-1)
        return top_k_probs.sum(dim=-1)

    @staticmethod
    def compute_logit_margin(logits: Tensor) -> Tensor:
        """
        Compute difference between top-2 logits (margin).

        A small margin indicates the model is uncertain between two choices.
        A large margin indicates confident prediction.

        Args:
            logits: Logits over vocabulary, shape [..., vocab_size]

        Returns:
            Margin (top1 - top2 logit), shape [...]
        """
        # Get top 2 logits
        top2_logits, _ = logits.topk(2, dim=-1)  # [..., 2]
        # Margin = difference between 1st and 2nd
        margin = top2_logits[..., 0] - top2_logits[..., 1]
        return margin

    @torch.no_grad()
    def extract(
        self,
        input_features: Tensor,
        tokens: Tensor,
        compute_top_k: bool = True,
    ) -> PredictiveEntropyResult:
        """
        Extract predictive entropy for given audio features and text tokens.

        Implements Signal 4 from the research plan.

        Args:
            input_features: Input features from processor, shape [batch, n_mels, n_frames]
                or [n_mels, n_frames]
            tokens: Decoder input tokens, shape [batch, seq_len] or [seq_len]
            compute_top_k: Whether to compute top-k probability mass

        Returns:
            PredictiveEntropyResult containing entropy values
        """
        single_batch = input_features.ndim == 2
        if single_batch:
            input_features = input_features.unsqueeze(0)

        # Handle 1D token tensor (unbatched)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
            single_batch = True  # Also flag for output squeeze

        # Forward pass to get logits
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=tokens,
            return_dict=True,
        )
        logits = outputs.logits  # [batch, seq, vocab]

        # Compute entropy at each position
        token_entropy = self.compute_entropy(logits)  # [batch, seq]

        # Normalize by log(vocab_size)
        normalized_entropy = self.normalize_entropy(token_entropy)

        # Utterance-level (mean across tokens)
        utterance_entropy = token_entropy.mean(dim=-1)

        # Top-k mass
        top_k_mass = None
        if compute_top_k:
            top_k_mass = self.compute_top_k_mass(logits, self.top_k)

        # Logit margin (top1 - top2)
        logit_margin = self.compute_logit_margin(logits)
        utterance_margin = logit_margin.mean(dim=-1)

        if single_batch:
            token_entropy = token_entropy.squeeze(0)
            normalized_entropy = normalized_entropy.squeeze(0)
            utterance_entropy = utterance_entropy.squeeze(0)
            logit_margin = logit_margin.squeeze(0)
            utterance_margin = utterance_margin.squeeze(0)
            if top_k_mass is not None:
                top_k_mass = top_k_mass.squeeze(0)

        return PredictiveEntropyResult(
            token_entropy=token_entropy,
            normalized_entropy=normalized_entropy,
            utterance_entropy=utterance_entropy,
            top_k_mass=top_k_mass,
            logit_margin=logit_margin,
            utterance_margin=utterance_margin,
        )

    @torch.no_grad()
    def extract_from_encoder_outputs(
        self,
        encoder_outputs: Tensor,
        tokens: Tensor,
        compute_top_k: bool = True,
    ) -> PredictiveEntropyResult:
        """
        Extract predictive entropy from pre-computed encoder outputs.

        More efficient when encoder outputs are already available.

        Args:
            encoder_outputs: Encoded audio, shape [batch, n_ctx, d_model]
            tokens: Decoder input tokens, shape [batch, seq_len]
            compute_top_k: Whether to compute top-k probability mass

        Returns:
            PredictiveEntropyResult containing entropy values
        """
        single_batch = encoder_outputs.ndim == 2
        if single_batch:
            encoder_outputs = encoder_outputs.unsqueeze(0)
            tokens = tokens.unsqueeze(0)

        # Run decoder with encoder outputs
        decoder_outputs = self.model.model.decoder(
            input_ids=tokens,
            encoder_hidden_states=encoder_outputs,
            return_dict=True,
        )
        # Get logits through projection layer
        logits = self.model.proj_out(decoder_outputs.last_hidden_state)

        token_entropy = self.compute_entropy(logits)
        normalized_entropy = self.normalize_entropy(token_entropy)
        utterance_entropy = token_entropy.mean(dim=-1)

        top_k_mass = None
        if compute_top_k:
            top_k_mass = self.compute_top_k_mass(logits, self.top_k)

        # Logit margin (top1 - top2)
        logit_margin = self.compute_logit_margin(logits)
        utterance_margin = logit_margin.mean(dim=-1)

        if single_batch:
            token_entropy = token_entropy.squeeze(0)
            normalized_entropy = normalized_entropy.squeeze(0)
            utterance_entropy = utterance_entropy.squeeze(0)
            logit_margin = logit_margin.squeeze(0)
            utterance_margin = utterance_margin.squeeze(0)
            if top_k_mass is not None:
                top_k_mass = top_k_mass.squeeze(0)

        return PredictiveEntropyResult(
            token_entropy=token_entropy,
            normalized_entropy=normalized_entropy,
            utterance_entropy=utterance_entropy,
            top_k_mass=top_k_mass,
            logit_margin=logit_margin,
            utterance_margin=utterance_margin,
        )
