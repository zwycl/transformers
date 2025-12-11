"""
Shallow Fusion Module for Whisper.

Implements trie-based shallow fusion for biasing Whisper's decoder towards
or away from specific token sequences during decoding.

Shallow fusion combines the acoustic model (Whisper) with an external bias
by adding weighted log-probabilities:
    log P(y|x) = log P_AM(y|x) + λ * bias(y)

This implementation uses a trie structure for efficient O(k) prefix matching
where k is the length of the decoded sequence so far.

Usage:
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer
    from shallow_fusion import ShallowFusionProcessor, BiasList

    # Load model and tokenizer
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

    # Create bias list
    bias_list = BiasList(tokenizer)
    bias_list.add_text("hello world", lambda_val=2.0)  # Boost
    bias_list.add_text("bad phrase", lambda_val=-5.0)  # Suppress

    # Create shallow fusion processor
    processor = ShallowFusionProcessor.from_bias_list(bias_list)

    # Get logits processor for generation
    logits_processor = processor.get_logits_processor()

    # Use with model.generate()
    outputs = model.generate(input_features, logits_processor=[logits_processor])
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch import Tensor

from .trie import BatchTokenTrie
from .bias_list import BiasList


@dataclass
class ShallowFusionConfig:
    """Configuration for shallow fusion behavior."""

    # Global scale factor applied to all biases
    global_scale: float = 1.0

    # Whether to apply bias only at sequence completion (terminal nodes)
    # If False, applies proportional bias at each step along the path
    terminal_only: bool = True

    # Minimum bias value to apply (clips smaller values)
    min_bias: float = -100.0

    # Maximum bias value to apply (clips larger values)
    max_bias: float = 100.0

    # Whether to normalize biases by sequence length
    length_normalize: bool = False

    # Apply softmax temperature scaling to biases
    temperature: float = 1.0


@dataclass
class ShallowFusionResult:
    """Result from shallow fusion application."""

    # Token IDs that received bias at this step
    biased_tokens: List[int]

    # Bias values applied to each token
    bias_values: List[float]

    # Whether any terminal sequences were completed
    completed_sequences: int

    # Current prefix length
    prefix_length: int


class ShallowFusionProcessor:
    """
    Trie-based shallow fusion processor for Whisper decoding.

    Maintains a trie of biased token sequences and provides efficient
    lookup during decoding to determine which tokens should be biased.

    The shallow fusion equation is:
        logits'[t] = logits[t] + λ_t * scale

    Where λ_t is the bias for token t given the current prefix.

    Args:
        config: Configuration options for shallow fusion behavior
    """

    def __init__(self, config: Optional[ShallowFusionConfig] = None):
        self.config = config or ShallowFusionConfig()
        self.trie = BatchTokenTrie()
        self._sequence_count = 0
        # Track first tokens of all sequences - these are always biased
        # to encourage starting any biased sequence
        self._first_token_biases: Dict[int, float] = {}

    @classmethod
    def from_bias_list(
        cls,
        bias_list: BiasList,
        config: Optional[ShallowFusionConfig] = None,
    ) -> "ShallowFusionProcessor":
        """
        Create a ShallowFusionProcessor from a BiasList.

        Args:
            bias_list: BiasList containing token sequences and their biases
            config: Optional configuration

        Returns:
            Configured ShallowFusionProcessor
        """
        processor = cls(config)
        processor.load_bias_list(bias_list)
        return processor

    @classmethod
    def from_token_lambda_pairs(
        cls,
        pairs: List[Tuple[List[int], float]],
        config: Optional[ShallowFusionConfig] = None,
    ) -> "ShallowFusionProcessor":
        """
        Create a ShallowFusionProcessor from (tokens, lambda) pairs.

        Args:
            pairs: List of (token_sequence, lambda_value) tuples
            config: Optional configuration

        Returns:
            Configured ShallowFusionProcessor
        """
        processor = cls(config)
        for tokens, lambda_val in pairs:
            processor.add_sequence(tokens, lambda_val)
        return processor

    def load_bias_list(self, bias_list: BiasList) -> int:
        """
        Load all entries from a BiasList into the trie.

        Args:
            bias_list: BiasList to load

        Returns:
            Number of sequences loaded
        """
        count = 0
        for entry in bias_list:
            self.add_sequence(entry.tokens, entry.lambda_val)
            count += 1
        return count

    def add_sequence(self, tokens: List[int], lambda_val: float) -> None:
        """
        Add a token sequence with its bias value to the trie.

        Args:
            tokens: Token IDs representing the sequence
            lambda_val: Bias value (positive = boost, negative = suppress)
        """
        if not tokens:
            return

        # Apply global scale
        scaled_lambda = lambda_val * self.config.global_scale

        # Apply length normalization if configured
        if self.config.length_normalize:
            scaled_lambda = scaled_lambda / len(tokens)

        # Clip to configured bounds
        scaled_lambda = max(self.config.min_bias,
                           min(self.config.max_bias, scaled_lambda))

        # Store the FIRST token - always biased to encourage starting the sequence
        first_token = tokens[0]
        if first_token in self._first_token_biases:
            existing = self._first_token_biases[first_token]
            # Keep the stronger bias
            if scaled_lambda < 0:
                self._first_token_biases[first_token] = min(existing, scaled_lambda)
            else:
                self._first_token_biases[first_token] = max(existing, scaled_lambda)
        else:
            self._first_token_biases[first_token] = scaled_lambda

        # Insert full sequence into trie for prefix-based continuation biasing
        self.trie.insert(tokens, scaled_lambda)
        self._sequence_count += 1

    def get_next_token_biases(
        self,
        prefix: List[int],
        sample_begin: int = 0,
    ) -> Dict[int, float]:
        """
        Get biases for possible next tokens given the current prefix.

        Biasing strategy:
        - First tokens of sequences are biased ONLY at the start of generation
          or after timestamp tokens (to allow natural flow)
        - Continuation tokens are biased ONLY if the prefix matches

        Example for sequence [1, 2, 3]:
        - Token 1: biased at start or after timestamps
        - Token 2: biased only if prefix ends with [..., 1]
        - Token 3: biased only if prefix ends with [..., 1, 2]

        Args:
            prefix: Token sequence decoded so far
            sample_begin: Index where actual sampling begins (after SOT sequence)

        Returns:
            Dict mapping token IDs to bias values
        """
        biases = {}

        # Get effective prefix (after SOT sequence)
        effective_prefix = prefix[sample_begin:] if sample_begin > 0 else prefix

        # NOTE: We no longer unconditionally apply first-token biases.
        # This was causing beam search to create spurious hypotheses that skip content.
        # Instead, biases are only applied when the trie matches the current context.

        # Determine if this is TRUE start of generation (raw prefix is empty)
        # vs empty effective prefix after timestamps
        is_true_start = len(prefix) == 0 or (sample_begin > 0 and len(prefix) == sample_begin)

        # Check all possible suffix matches for continuation biasing
        # For prefix [a, b, c, d], check if any suffix [d], [c,d], [b,c,d], [a,b,c,d]
        # matches the start of a biased sequence
        for i in range(len(effective_prefix) + 1):
            suffix = effective_prefix[i:]
            if not suffix:
                # Empty suffix case - only apply first-token biases at TRUE start
                if is_true_start:
                    suffix_biases = self.trie.get_next_token_biases(suffix)
                else:
                    # After timestamps - skip first-token biasing to avoid
                    # beam search creating hypotheses that skip content
                    continue
            else:
                suffix_biases = self.trie.get_next_token_biases(suffix)

            for token, bias in suffix_biases.items():
                if token in biases:
                    # Combine biases - take stronger
                    if bias < 0:
                        biases[token] = min(biases[token], bias)
                    else:
                        biases[token] = max(biases[token], bias)
                else:
                    biases[token] = bias

        # Apply temperature scaling
        if self.config.temperature != 1.0:
            biases = {
                token: bias / self.config.temperature
                for token, bias in biases.items()
            }

        return biases

    def get_completion_bias(
        self,
        tokens: List[int],
        sample_begin: int = 0,
    ) -> float:
        """
        Get the bias if the sequence completes a biased sequence.

        Args:
            tokens: Current token sequence
            sample_begin: Index where sampling begins

        Returns:
            Bias value if sequence completes a bias entry, 0.0 otherwise
        """
        effective_tokens = tokens[sample_begin:] if sample_begin > 0 else tokens
        return self.trie.get_completion_bias(effective_tokens)

    def apply_biases(
        self,
        logits: Tensor,
        prefix: List[int],
        sample_begin: int = 0,
    ) -> ShallowFusionResult:
        """
        Apply shallow fusion biases to logits tensor.

        Modifies logits in-place.

        Args:
            logits: Logits tensor of shape [vocab_size] or [batch, vocab_size]
            prefix: Current token prefix
            sample_begin: Index where sampling begins

        Returns:
            ShallowFusionResult with details of applied biases
        """
        biases = self.get_next_token_biases(prefix, sample_begin)

        biased_tokens = []
        bias_values = []
        completed = 0

        for token_id, bias_val in biases.items():
            if logits.ndim == 1:
                logits[token_id] += bias_val
            else:
                logits[:, token_id] += bias_val

            biased_tokens.append(token_id)
            bias_values.append(bias_val)

            # Check if this would complete a sequence
            test_seq = prefix + [token_id]
            if self.trie.search(test_seq[sample_begin:]) is not None:
                completed += 1

        effective_prefix = prefix[sample_begin:] if sample_begin > 0 else prefix

        return ShallowFusionResult(
            biased_tokens=biased_tokens,
            bias_values=bias_values,
            completed_sequences=completed,
            prefix_length=len(effective_prefix),
        )

    def batch_apply_biases(
        self,
        logits: Tensor,
        prefixes: List[List[int]],
        sample_begin: int = 0,
    ) -> List[ShallowFusionResult]:
        """
        Apply shallow fusion biases to batched logits.

        Args:
            logits: Logits tensor of shape [batch, vocab_size]
            prefixes: List of token prefixes, one per batch item
            sample_begin: Index where sampling begins

        Returns:
            List of ShallowFusionResult, one per batch item
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits for batch, got {logits.ndim}D")

        if len(prefixes) != logits.shape[0]:
            raise ValueError(
                f"Number of prefixes ({len(prefixes)}) doesn't match "
                f"batch size ({logits.shape[0]})"
            )

        results = []
        for i, prefix in enumerate(prefixes):
            biases = self.get_next_token_biases(prefix, sample_begin)

            biased_tokens = []
            bias_values = []

            for token_id, bias_val in biases.items():
                logits[i, token_id] += bias_val
                biased_tokens.append(token_id)
                bias_values.append(bias_val)

            effective_prefix = prefix[sample_begin:] if sample_begin > 0 else prefix

            results.append(ShallowFusionResult(
                biased_tokens=biased_tokens,
                bias_values=bias_values,
                completed_sequences=0,  # Could compute if needed
                prefix_length=len(effective_prefix),
            ))

        return results

    def get_logits_processor(self, sample_begin: int = 0) -> "ShallowFusionLogitsProcessor":
        """
        Get a LogitsProcessor for use with Transformers' generate() method.

        Args:
            sample_begin: Index where token sampling begins (after initial tokens)

        Returns:
            ShallowFusionLogitsProcessor compatible with Transformers' generation pipeline
        """
        return ShallowFusionLogitsProcessor(self, sample_begin)

    def get_cost_subtraction_processor(
        self, sample_begin: int = 0
    ) -> "CostSubtractionLogitsProcessor":
        """
        Get a LogitsProcessor with cost subtraction for beam search.

        This processor tracks accumulated bias per beam and revokes it when
        a biased sequence is broken. This is recommended when using num_beams > 1
        to prevent beams from gaining unfair advantage from partial matches.

        Args:
            sample_begin: Index where actual token sampling begins

        Returns:
            CostSubtractionLogitsProcessor for beam search compatibility
        """
        return CostSubtractionLogitsProcessor(self, sample_begin)

    def clear(self) -> None:
        """Remove all sequences from the processor."""
        self.trie.clear()
        self._sequence_count = 0
        self._first_token_biases.clear()

    @property
    def num_sequences(self) -> int:
        """Number of bias sequences loaded."""
        return self._sequence_count

    def __len__(self) -> int:
        return self._sequence_count


class ShallowFusionLogitsProcessor:
    """
    LogitsProcessor implementation for shallow fusion.

    Compatible with Transformers' generation pipeline. Applies trie-based
    biases to logits at each decoding step.

    This class implements the LogitsProcessor interface from transformers,
    allowing it to be passed to model.generate(logits_processor=[...]).

    Usage:
        processor = ShallowFusionProcessor.from_bias_list(bias_list)
        logits_processor = processor.get_logits_processor()

        # Use with generate()
        outputs = model.generate(input_features, logits_processor=[logits_processor])
    """

    def __init__(
        self,
        processor: ShallowFusionProcessor,
        sample_begin: int = 0,
    ):
        """
        Initialize the shallow fusion logits processor.

        Args:
            processor: ShallowFusionProcessor with loaded bias sequences
            sample_begin: Index where token sampling begins
        """
        self.processor = processor
        self.sample_begin = sample_begin
        self._last_result: Optional[ShallowFusionResult] = None

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """
        Apply shallow fusion biases to scores (logits).

        This method is called by Transformers' generate() at each step.
        Returns modified scores tensor.

        Args:
            input_ids: Input token IDs, shape [batch, seq_len]
            scores: Logits tensor, shape [batch, vocab_size]

        Returns:
            Modified scores tensor with biases applied
        """
        batch_size = input_ids.shape[0]

        if batch_size == 1:
            # Single sequence - use simple apply
            prefix = input_ids[0].tolist()
            self._last_result = self.processor.apply_biases(
                scores[0],
                prefix,
                self.sample_begin,
            )
        else:
            # Batch processing
            prefixes = [input_ids[i].tolist() for i in range(batch_size)]
            results = self.processor.batch_apply_biases(
                scores,
                prefixes,
                self.sample_begin,
            )
            # Store first result for debugging
            self._last_result = results[0] if results else None

        return scores

    @property
    def last_result(self) -> Optional[ShallowFusionResult]:
        """Get the result from the last __call__() invocation."""
        return self._last_result


@dataclass
class BeamState:
    """State tracking for a single beam in cost subtraction."""
    trie_node_path: List[int]  # Path of token IDs we're tracking in trie
    accumulated_bias: float  # Total bias accumulated on current trie path


class CostSubtractionLogitsProcessor:
    """
    LogitsProcessor with cost subtraction for beam search compatibility.

    This processor tracks accumulated bias per beam and revokes (subtracts)
    the bias when a biased sequence is broken. This prevents beams from
    "cheating" by accumulating bias for partial matches and then diverging.

    The key insight: if a beam received +2.0 bias for tokens [A, B] expecting
    sequence [A, B, C], but then generates token D instead of C, we subtract
    the accumulated +2.0 so the beam doesn't keep its unfair advantage.

    This implements the "cost subtraction" approach from the literature on
    biased beam search.

    Usage:
        processor = ShallowFusionProcessor.from_bias_list(bias_list)
        logits_processor = CostSubtractionLogitsProcessor(processor)

        outputs = model.generate(
            input_features,
            num_beams=10,
            logits_processor=[logits_processor]
        )
    """

    def __init__(
        self,
        processor: "ShallowFusionProcessor",
        sample_begin: int = 0,
    ):
        """
        Initialize the cost subtraction logits processor.

        Args:
            processor: ShallowFusionProcessor with loaded bias sequences
            sample_begin: Index where token sampling begins (after SOT tokens)
        """
        self.processor = processor
        self.sample_begin = sample_begin
        # Track state per beam: beam_id -> BeamState
        self._beam_states: Dict[int, BeamState] = {}
        # Track previous input_ids to detect beam expansions
        self._prev_input_ids: Optional[List[List[int]]] = None

    def _get_beam_key(self, tokens: List[int]) -> int:
        """Create a hashable key for beam state lookup."""
        return hash(tuple(tokens))

    def _check_trie_path(self, tokens: List[int]) -> Tuple[bool, List[int]]:
        """
        Check if tokens follow a valid trie path.

        Returns:
            Tuple of (is_on_valid_path, path_tokens)
        """
        effective_tokens = tokens[self.sample_begin:] if self.sample_begin > 0 else tokens

        # Try to find the longest suffix that's on a trie path
        for start in range(len(effective_tokens)):
            suffix = effective_tokens[start:]
            if self.processor.trie.starts_with(suffix):
                return True, suffix

        return False, []

    def __call__(self, input_ids: "Tensor", scores: "Tensor") -> "Tensor":
        """
        Apply shallow fusion with cost subtraction.

        This method:
        1. Detects which beams are continuing vs new
        2. For continuing beams, checks if they're still on a trie path
        3. If a beam broke its trie path, subtracts accumulated bias
        4. Applies new biases for next token predictions
        5. Tracks accumulated bias for each beam

        Args:
            input_ids: Input token IDs, shape [batch, seq_len]
            scores: Logits tensor, shape [batch, vocab_size]

        Returns:
            Modified scores tensor with biases applied and cost subtraction
        """
        import torch

        batch_size = input_ids.shape[0]
        device = scores.device

        # Convert to lists for processing
        current_sequences = [input_ids[i].tolist() for i in range(batch_size)]

        # Initialize new beam states dict for this step
        new_beam_states: Dict[int, BeamState] = {}

        for i in range(batch_size):
            tokens = current_sequences[i]
            effective_tokens = tokens[self.sample_begin:] if self.sample_begin > 0 else tokens

            # Find parent beam state (tokens minus last token)
            if len(effective_tokens) > 0:
                parent_tokens = tokens[:-1]
                parent_key = self._get_beam_key(parent_tokens)
                parent_state = self._beam_states.get(parent_key)
            else:
                parent_state = None

            # Check if we're on a valid trie path
            on_path, path_tokens = self._check_trie_path(tokens)

            if parent_state is not None and not on_path:
                # We were on a trie path but broke it - SUBTRACT accumulated bias
                # This revokes the unfair advantage gained from partial matches
                if parent_state.accumulated_bias != 0:
                    scores[i] -= parent_state.accumulated_bias

                # Reset state for this beam
                new_beam_states[self._get_beam_key(tokens)] = BeamState(
                    trie_node_path=[],
                    accumulated_bias=0.0,
                )
            elif on_path:
                # We're on a valid trie path - track accumulated bias
                # Get bias for the last token if we're continuing a path
                accumulated = parent_state.accumulated_bias if parent_state else 0.0

                new_beam_states[self._get_beam_key(tokens)] = BeamState(
                    trie_node_path=path_tokens,
                    accumulated_bias=accumulated,
                )
            else:
                # Not on any path, no parent - fresh start
                new_beam_states[self._get_beam_key(tokens)] = BeamState(
                    trie_node_path=[],
                    accumulated_bias=0.0,
                )

            # Apply biases for next token predictions
            biases = self.processor.get_next_token_biases(tokens, self.sample_begin)

            for token_id, bias_val in biases.items():
                scores[i, token_id] += bias_val

            # Track the bias we're adding for cost subtraction later
            # We need to track the max bias being added since we don't know
            # which token will be selected yet
            if biases:
                max_bias = max(biases.values())
                beam_key = self._get_beam_key(tokens)
                if beam_key in new_beam_states:
                    new_beam_states[beam_key].accumulated_bias += max_bias

        # Update beam states for next iteration
        self._beam_states = new_beam_states
        self._prev_input_ids = current_sequences

        return scores

    def reset(self) -> None:
        """Reset beam state tracking. Call between different audio samples."""
        self._beam_states.clear()
        self._prev_input_ids = None


class ContextualShallowFusion:
    """
    Context-aware shallow fusion that can adjust biases based on decoded content.

    Extends basic shallow fusion with the ability to:
    - Enable/disable bias sequences based on context
    - Adjust bias strength dynamically
    - Track which biases have been triggered

    This is useful for more sophisticated biasing strategies where
    the bias should depend on what has already been decoded.
    """

    def __init__(
        self,
        processor: ShallowFusionProcessor,
        sample_begin: int = 0,
    ):
        self.processor = processor
        self.sample_begin = sample_begin
        self._triggered_sequences: List[Tuple[List[int], float]] = []
        self._active = True

    def set_active(self, active: bool) -> None:
        """Enable or disable shallow fusion."""
        self._active = active

    def apply(self, logits: Tensor, tokens: Tensor) -> Optional[ShallowFusionResult]:
        """
        Apply context-aware shallow fusion.

        Args:
            logits: Logits tensor
            tokens: Current tokens

        Returns:
            ShallowFusionResult if fusion was applied, None if inactive
        """
        if not self._active:
            return None

        prefix = tokens[0].tolist() if tokens.ndim == 2 else tokens.tolist()
        result = self.processor.apply_biases(logits, prefix, self.sample_begin)

        # Track completed sequences
        if result.completed_sequences > 0:
            effective_prefix = prefix[self.sample_begin:]
            completions = self.processor.trie.get_all_continuations([])
            for seq, lambda_val in completions:
                if effective_prefix[-len(seq):] == seq if len(seq) <= len(effective_prefix) else False:
                    self._triggered_sequences.append((seq, lambda_val))

        return result

    @property
    def triggered_sequences(self) -> List[Tuple[List[int], float]]:
        """Get list of (sequence, lambda) pairs that were triggered."""
        return self._triggered_sequences.copy()

    def reset(self) -> None:
        """Reset triggered sequence tracking."""
        self._triggered_sequences.clear()


def create_shallow_fusion_decoder(
    bias_list: BiasList,
    config: Optional[ShallowFusionConfig] = None,
) -> Tuple[ShallowFusionProcessor, ShallowFusionLogitsProcessor]:
    """
    Convenience function to create shallow fusion components.

    Args:
        bias_list: BiasList with token sequences and lambdas
        config: Optional configuration

    Returns:
        Tuple of (processor, logits_processor) ready for use with model.generate()
    """
    processor = ShallowFusionProcessor.from_bias_list(bias_list, config)
    logits_processor = processor.get_logits_processor(sample_begin=0)
    return processor, logits_processor
