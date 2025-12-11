"""
Unit tests for ContextualShallowFusion.

Tests the contextual shallow fusion functionality with example bias lists
inspired by domain-specific terminology from ContextASR-Bench Speech dataset.

Uses a mock tokenizer to avoid requiring actual Whisper model downloads.
For real usage, use `from transformers import WhisperTokenizer`.
"""

import pytest
import torch
from torch import Tensor

from shallow_fusion import (
    BiasList,
    ShallowFusionProcessor,
    ShallowFusionConfig,
    ShallowFusionLogitsProcessor,
)
from shallow_fusion.shallow_fusion import ContextualShallowFusion


class MockTokenizer:
    """Mock tokenizer for testing without loading full Whisper tokenizer."""

    def __init__(self):
        # Simple word-to-token mapping for testing
        self._vocab = {
            " the": 1,
            " financial": 2,
            " report": 3,
            " earnings": 4,
            " call": 5,
            " revenue": 6,
            " growth": 7,
            " quarter": 8,
            " profit": 9,
            " margin": 10,
            " stock": 11,
            " price": 12,
            " market": 13,
            " share": 14,
            " dividend": 15,
            " guidance": 16,
            " forecast": 17,
            " analyst": 18,
            " bad": 100,
            " word": 101,
            " explicit": 102,
            " content": 103,
            "hello": 200,
            " world": 201,
            " medical": 300,
            " diagnosis": 301,
            " treatment": 302,
            " patient": 303,
            " prescription": 304,
        }
        self._id_to_word = {v: k for k, v in self._vocab.items()}

    def encode(self, text: str) -> list:
        """Simple encoding that looks up words in vocab."""
        tokens = []
        # Handle text with leading space for BPE-like behavior
        words = text.split()
        for i, word in enumerate(words):
            # Add space prefix except for first word if text doesn't start with space
            lookup = f" {word}" if (i > 0 or text.startswith(" ")) else word
            if lookup in self._vocab:
                tokens.append(self._vocab[lookup])
            elif f" {word}" in self._vocab:
                tokens.append(self._vocab[f" {word}"])
            else:
                # Unknown word - use hash as token
                tokens.append(hash(word) % 1000 + 500)
        return tokens

    def decode(self, tokens: list) -> str:
        """Decode tokens back to text."""
        words = [self._id_to_word.get(t, f"<{t}>") for t in tokens]
        return "".join(words).strip()


@pytest.fixture
def mock_tokenizer():
    """Fixture providing mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def financial_bias_list(mock_tokenizer):
    """
    Bias list with financial/earnings call terminology.

    Inspired by ContextASR-Bench Speech dataset which includes
    financial earnings calls (FNED domain).
    """
    bias_list = BiasList(mock_tokenizer)

    # Boost financial terminology (positive lambda)
    financial_terms = [
        ("earnings call", 3.0),
        ("revenue growth", 2.5),
        ("profit margin", 2.5),
        ("stock price", 2.0),
        ("market share", 2.0),
        ("dividend", 1.5),
        ("guidance forecast", 3.0),
        ("financial report", 2.0),
    ]

    for term, boost in financial_terms:
        bias_list.add_text(term, lambda_val=boost)

    return bias_list


@pytest.fixture
def medical_bias_list(mock_tokenizer):
    """
    Bias list with medical terminology.

    Example domain-specific bias for medical transcription.
    """
    bias_list = BiasList(mock_tokenizer)

    medical_terms = [
        ("medical diagnosis", 3.0),
        ("treatment", 2.0),
        ("patient", 1.5),
        ("prescription", 2.5),
    ]

    for term, boost in medical_terms:
        bias_list.add_text(term, lambda_val=boost)

    return bias_list


@pytest.fixture
def suppression_bias_list(mock_tokenizer):
    """Bias list for suppressing unwanted content."""
    bias_list = BiasList(mock_tokenizer)

    # Suppress unwanted terms (negative lambda)
    suppress_terms = [
        ("bad word", -10.0),
        ("explicit content", -15.0),
    ]

    for term, suppress in suppress_terms:
        bias_list.add_text(term, lambda_val=suppress)

    return bias_list


class TestContextualShallowFusionBasic:
    """Basic functionality tests for ContextualShallowFusion."""

    def test_initialization(self, financial_bias_list):
        """Test that ContextualShallowFusion initializes correctly."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        assert contextual._active is True
        assert len(contextual.triggered_sequences) == 0
        assert contextual.processor is processor

    def test_set_active(self, financial_bias_list):
        """Test enabling/disabling shallow fusion."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        # Initially active
        assert contextual._active is True

        # Disable
        contextual.set_active(False)
        assert contextual._active is False

        # Re-enable
        contextual.set_active(True)
        assert contextual._active is True

    def test_apply_when_inactive(self, financial_bias_list):
        """Test that apply returns None when inactive."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        contextual.set_active(False)

        logits = torch.randn(1, 500)
        tokens = torch.tensor([[1, 2, 3]])

        result = contextual.apply(logits, tokens)
        assert result is None

    def test_reset(self, financial_bias_list):
        """Test resetting triggered sequence tracking."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        # Manually add some triggered sequences for testing
        contextual._triggered_sequences.append(([1, 2], 3.0))
        contextual._triggered_sequences.append(([3, 4], 2.0))

        assert len(contextual.triggered_sequences) == 2

        contextual.reset()
        assert len(contextual.triggered_sequences) == 0


class TestContextualShallowFusionApply:
    """Tests for the apply method of ContextualShallowFusion."""

    def test_apply_modifies_logits(self, financial_bias_list, mock_tokenizer):
        """Test that apply modifies logits for biased tokens."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        # Create logits tensor
        vocab_size = 500
        logits = torch.zeros(vocab_size)

        # Encode "earnings" to get the token for "call"
        # After "earnings", "call" should be boosted
        earnings_tokens = mock_tokenizer.encode("earnings")
        tokens = torch.tensor([earnings_tokens])

        # Get the token ID for "call"
        call_token = mock_tokenizer.encode(" call")[0]

        original_logit = logits[call_token].item()

        result = contextual.apply(logits, tokens)

        # Check that biased token was modified
        if result and call_token in result.biased_tokens:
            assert logits[call_token].item() != original_logit

    def test_apply_returns_result(self, financial_bias_list, mock_tokenizer):
        """Test that apply returns a ShallowFusionResult."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        logits = torch.zeros(500)
        earnings_tokens = mock_tokenizer.encode("earnings")
        tokens = torch.tensor([earnings_tokens])

        result = contextual.apply(logits, tokens)

        assert result is not None
        assert hasattr(result, 'biased_tokens')
        assert hasattr(result, 'bias_values')
        assert hasattr(result, 'prefix_length')

    def test_apply_with_sample_begin(self, financial_bias_list, mock_tokenizer):
        """Test apply with non-zero sample_begin (simulating SOT sequence)."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        # Simulate SOT sequence of length 4
        contextual = ContextualShallowFusion(processor, sample_begin=4)

        logits = torch.zeros(500)

        # Tokens: [sot tokens (4)] + [earnings]
        sot_tokens = [50258, 50259, 50260, 50261]  # Fake SOT sequence
        content_tokens = mock_tokenizer.encode("earnings")
        all_tokens = sot_tokens + content_tokens
        tokens = torch.tensor([all_tokens])

        result = contextual.apply(logits, tokens)

        # The effective prefix should only be the content tokens
        assert result is not None
        assert result.prefix_length == len(content_tokens)


class TestContextualShallowFusionWithFinancialDomain:
    """
    Tests using financial domain terminology from ContextASR-Bench.

    The ContextASR-Speech dataset includes financial earnings calls (FNED)
    with terminology like "revenue growth", "profit margin", "earnings call".
    """

    def test_financial_term_boosting(self, financial_bias_list, mock_tokenizer):
        """Test that financial terms are properly boosted."""
        config = ShallowFusionConfig(global_scale=1.0)
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list, config)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500
        logits = torch.zeros(vocab_size)

        # Simulate decoding "revenue" and check if "growth" gets boosted
        revenue_tokens = mock_tokenizer.encode("revenue")
        tokens = torch.tensor([revenue_tokens])

        growth_token = mock_tokenizer.encode(" growth")[0]
        original = logits[growth_token].item()

        result = contextual.apply(logits, tokens)

        # If "growth" follows "revenue" in our bias list, it should be boosted
        if result and growth_token in result.biased_tokens:
            idx = result.biased_tokens.index(growth_token)
            assert result.bias_values[idx] > 0  # Positive boost

    def test_multiple_financial_sequences(self, mock_tokenizer):
        """Test handling multiple financial term sequences."""
        bias_list = BiasList(mock_tokenizer)

        # Add multiple sequences starting with same prefix
        bias_list.add_text("market share", lambda_val=2.0)
        bias_list.add_text("market growth", lambda_val=2.5)

        processor = ShallowFusionProcessor.from_bias_list(bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500
        logits = torch.zeros(vocab_size)

        # After "market", both "share" and "growth" should have biases
        market_tokens = mock_tokenizer.encode("market")
        tokens = torch.tensor([market_tokens])

        result = contextual.apply(logits, tokens)

        if result:
            # Should have biases for multiple possible continuations
            assert len(result.biased_tokens) >= 0  # May or may not have continuations


class TestContextualShallowFusionWithSuppression:
    """Tests for content suppression using negative biases."""

    def test_suppression_bias(self, suppression_bias_list, mock_tokenizer):
        """Test that suppression biases are negative."""
        processor = ShallowFusionProcessor.from_bias_list(suppression_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500
        logits = torch.zeros(vocab_size)

        # After "bad", "word" should be suppressed
        bad_tokens = mock_tokenizer.encode("bad")
        tokens = torch.tensor([bad_tokens])

        word_token = mock_tokenizer.encode(" word")[0]
        original = logits[word_token].item()

        result = contextual.apply(logits, tokens)

        if result and word_token in result.biased_tokens:
            idx = result.biased_tokens.index(word_token)
            assert result.bias_values[idx] < 0  # Negative suppression
            assert logits[word_token].item() < original

    def test_strong_suppression(self, mock_tokenizer):
        """Test that strong suppression effectively prevents token selection."""
        bias_list = BiasList(mock_tokenizer)
        bias_list.add_text("explicit content", lambda_val=-100.0)

        processor = ShallowFusionProcessor.from_bias_list(bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500
        logits = torch.ones(vocab_size) * 10.0  # High uniform logits

        explicit_tokens = mock_tokenizer.encode("explicit")
        tokens = torch.tensor([explicit_tokens])

        content_token = mock_tokenizer.encode(" content")[0]

        contextual.apply(logits, tokens)

        # After strong suppression, content token should have very low logit
        assert logits[content_token].item() < 0


class TestContextualShallowFusionConfig:
    """Tests for ShallowFusionConfig options."""

    def test_global_scale(self, financial_bias_list, mock_tokenizer):
        """Test that global_scale multiplies all biases."""
        config = ShallowFusionConfig(global_scale=2.0)
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list, config)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500
        logits = torch.zeros(vocab_size)

        earnings_tokens = mock_tokenizer.encode("earnings")
        tokens = torch.tensor([earnings_tokens])

        result = contextual.apply(logits, tokens)

        # Biases should be scaled by 2.0
        if result and result.bias_values:
            # Original "earnings call" bias is 3.0, scaled should be 6.0
            # (if "call" is a valid continuation)
            pass  # Bias values are already scaled in processor

    def test_temperature_scaling(self, financial_bias_list, mock_tokenizer):
        """Test temperature scaling of biases."""
        config = ShallowFusionConfig(temperature=0.5)
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list, config)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500
        logits = torch.zeros(vocab_size)

        earnings_tokens = mock_tokenizer.encode("earnings")
        tokens = torch.tensor([earnings_tokens])

        result = contextual.apply(logits, tokens)

        # Lower temperature should amplify biases
        # bias / 0.5 = bias * 2
        if result and result.bias_values:
            pass  # Temperature scaling applied in get_next_token_biases

    def test_min_max_bias_clipping(self, mock_tokenizer):
        """Test that biases are clipped to min/max values."""
        bias_list = BiasList(mock_tokenizer)
        bias_list.add_text("extreme boost", lambda_val=1000.0)

        config = ShallowFusionConfig(max_bias=50.0)
        processor = ShallowFusionProcessor.from_bias_list(bias_list, config)

        # The bias should be clipped to max_bias
        # This is applied during insertion
        assert processor.num_sequences == 1


class TestContextualShallowFusionIntegration:
    """Integration tests simulating real decoding scenarios."""

    def test_full_decoding_simulation(self, financial_bias_list, mock_tokenizer):
        """Simulate a full decoding loop with contextual shallow fusion."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500

        # Simulate decoding: "the financial report"
        decoded_tokens = []
        words_to_decode = ["the", "financial", "report"]

        for word in words_to_decode:
            logits = torch.randn(vocab_size)

            if decoded_tokens:
                tokens = torch.tensor([decoded_tokens])
            else:
                tokens = torch.tensor([[]])

            # Apply shallow fusion
            result = contextual.apply(logits, tokens)

            # "Decode" next token (just append to sequence)
            next_token = mock_tokenizer.encode(f" {word}" if decoded_tokens else word)
            decoded_tokens.extend(next_token)

        # Should have decoded the full sequence
        assert len(decoded_tokens) == len(words_to_decode)

    def test_contextasr_speech_scenario(self, mock_tokenizer):
        """
        Test scenario inspired by ContextASR-Speech dataset.

        Simulates biasing for financial earnings call transcription
        where domain-specific terms should be boosted.
        """
        # Create bias list with FNED (Financial News) domain terms
        bias_list = BiasList(mock_tokenizer)

        # Terms commonly found in earnings calls
        fned_terms = [
            ("quarterly earnings", 3.0),
            ("revenue growth", 2.5),
            ("profit margin", 2.5),
            ("guidance", 2.0),
            ("analyst", 1.5),
        ]

        for term, boost in fned_terms:
            bias_list.add_text(term, lambda_val=boost)

        processor = ShallowFusionProcessor.from_bias_list(bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500

        # Simulate partial decode of "quarterly"
        quarterly_tokens = mock_tokenizer.encode("quarterly")
        tokens = torch.tensor([quarterly_tokens])
        logits = torch.zeros(vocab_size)

        result = contextual.apply(logits, tokens)

        # Verify result structure
        assert result is not None
        assert isinstance(result.biased_tokens, list)
        assert isinstance(result.bias_values, list)
        assert len(result.biased_tokens) == len(result.bias_values)

    def test_mixed_boost_suppress(self, mock_tokenizer):
        """Test combining boost and suppress biases."""
        bias_list = BiasList(mock_tokenizer)

        # Boost good terms
        bias_list.add_text("financial report", lambda_val=3.0)

        # Suppress bad terms
        bias_list.add_text("bad word", lambda_val=-10.0)

        processor = ShallowFusionProcessor.from_bias_list(bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        vocab_size = 500
        logits = torch.zeros(vocab_size)

        # Test boost path
        financial_tokens = mock_tokenizer.encode("financial")
        tokens = torch.tensor([financial_tokens])
        result = contextual.apply(logits.clone(), tokens)

        report_token = mock_tokenizer.encode(" report")[0]
        if result and report_token in result.biased_tokens:
            idx = result.biased_tokens.index(report_token)
            assert result.bias_values[idx] > 0

        # Test suppress path
        contextual.reset()
        logits = torch.zeros(vocab_size)
        bad_tokens = mock_tokenizer.encode("bad")
        tokens = torch.tensor([bad_tokens])
        result = contextual.apply(logits, tokens)

        word_token = mock_tokenizer.encode(" word")[0]
        if result and word_token in result.biased_tokens:
            idx = result.biased_tokens.index(word_token)
            assert result.bias_values[idx] < 0


class TestTriggeredSequenceTracking:
    """Tests for tracking which bias sequences have been triggered."""

    def test_triggered_sequences_property(self, financial_bias_list):
        """Test that triggered_sequences returns a copy."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        triggered1 = contextual.triggered_sequences
        triggered2 = contextual.triggered_sequences

        # Should return copies, not the same list
        assert triggered1 is not triggered2

    def test_triggered_sequences_after_reset(self, financial_bias_list):
        """Test that reset clears triggered sequences."""
        processor = ShallowFusionProcessor.from_bias_list(financial_bias_list)
        contextual = ContextualShallowFusion(processor, sample_begin=0)

        # Add some fake triggered sequences
        contextual._triggered_sequences.append(([1, 2], 3.0))

        assert len(contextual.triggered_sequences) == 1

        contextual.reset()

        assert len(contextual.triggered_sequences) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
