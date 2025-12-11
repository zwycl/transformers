"""
GradNorm Signal Extractor (Signal 5)

Computes GradNorm as a measure of input familiarity/OOD detection.
Based on the insight that gradient norms of KL(uniform || softmax(logits))
w.r.t. output layer weights indicate how familiar the model is with the input.

Uses Transformers WhisperForConditionalGeneration API.

Key insight (counterintuitive):
- Lower GradNorm → OOD/unfamiliar input → epistemic uncertainty
- Higher GradNorm → in-distribution → model is confident

This works because familiar inputs produce sharper logit distributions,
which have larger gradients when pushed toward uniform.

From the research:
    "GradNorm provides a training-free OOD detection signal by measuring
     how strongly the model's predictions resist being pushed toward uniform."
"""

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from transformers import WhisperForConditionalGeneration


@dataclass
class GradNormResult:
    """Results from GradNorm computation."""

    # GradNorm per sample, shape: [batch] or scalar
    # Lower = more OOD/uncertain, Higher = more in-distribution
    gradnorm: Tensor


def compute_gradnorm(
    model: WhisperForConditionalGeneration,
    input_features: Tensor,
    tokens: Tensor,
) -> Tensor:
    """
    Compute GradNorm for given audio features and tokens.

    Simple functional interface.

    Args:
        model: WhisperForConditionalGeneration model
        input_features: Input features from processor
        tokens: Decoder input tokens

    Returns:
        GradNorm value, shape [batch] or scalar
    """
    extractor = GradNormExtractor(model)
    result = extractor.extract(input_features, tokens)
    return result.gradnorm


class GradNormExtractor:
    """
    Extract GradNorm signal from Whisper decoder.

    GradNorm measures the gradient norm of KL(uniform || softmax(logits))
    with respect to the output projection weights. This provides a
    training-free signal for OOD detection.

    Args:
        model: The WhisperForConditionalGeneration model instance
        temperature: Temperature for softmax (default: 1.0)
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        temperature: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.vocab_size = model.config.vocab_size

    def _compute_kl_uniform_to_pred(self, logits: Tensor) -> Tensor:
        """
        Compute KL(uniform || softmax(logits/T)).

        This measures how far the predictions are from uniform.
        Sharper predictions = larger KL.

        Args:
            logits: Model logits, shape [..., vocab_size]

        Returns:
            KL divergence, scalar (summed over all positions)
        """
        # Apply temperature
        scaled_logits = logits / self.temperature

        # Softmax predictions: q(y|x) as log probabilities
        log_probs = F.log_softmax(scaled_logits, dim=-1)

        # Uniform distribution: p(y) = 1/V
        uniform = torch.full_like(log_probs, 1.0 / self.vocab_size)

        # KL(uniform || pred) = sum_y uniform(y) * (log uniform(y) - log pred(y))
        # F.kl_div expects (log_probs, target_probs) and computes sum(target * (log(target) - log_probs))
        # So we pass (log_probs, uniform) to get KL(uniform || pred)
        kl = F.kl_div(log_probs, uniform, reduction='sum')

        return kl

    def extract(
        self,
        input_features: Tensor,
        tokens: Tensor,
    ) -> GradNormResult:
        """
        Extract GradNorm for given audio features and tokens.

        Args:
            input_features: Input features from processor, shape [batch, n_mels, n_frames]
                or [n_mels, n_frames]
            tokens: Decoder input tokens, shape [batch, seq_len] or [seq_len]

        Returns:
            GradNormResult containing utterance-level GradNorm
        """
        single_batch = input_features.ndim == 2
        if single_batch:
            input_features = input_features.unsqueeze(0)

        # Handle 1D token tensor (unbatched)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
            single_batch = True  # Also flag for output squeeze

        batch_size = input_features.shape[0]
        device = input_features.device

        # Get the output projection weights
        # In Transformers Whisper: logits = decoder_hidden @ proj_out.weight.T
        output_weight = self.model.proj_out.weight

        # Ensure weight requires grad for this computation
        original_requires_grad = output_weight.requires_grad
        output_weight.requires_grad_(True)

        try:
            # Compute per-sample GradNorm
            gradnorms = []

            for i in range(batch_size):
                sample_features = input_features[i:i+1]
                sample_tokens = tokens[i:i+1]

                # Zero any existing gradients
                if output_weight.grad is not None:
                    output_weight.grad.zero_()

                # Forward pass using the full model
                outputs = self.model(
                    input_features=sample_features,
                    decoder_input_ids=sample_tokens,
                    return_dict=True,
                )
                logits = outputs.logits  # [1, seq_len, vocab_size]

                # Compute KL(uniform || pred) and backprop
                kl_loss = self._compute_kl_uniform_to_pred(logits)
                kl_loss.backward()

                # Get gradient norm
                if output_weight.grad is not None:
                    grad_norm = output_weight.grad.norm().detach()
                else:
                    grad_norm = torch.tensor(0.0, device=device)

                gradnorms.append(grad_norm)

            # Stack results
            gradnorm = torch.stack(gradnorms)

        finally:
            # Restore original requires_grad state
            output_weight.requires_grad_(original_requires_grad)
            # Clear any remaining gradients
            if output_weight.grad is not None:
                output_weight.grad = None

        if single_batch:
            gradnorm = gradnorm.squeeze(0)

        return GradNormResult(gradnorm=gradnorm)
