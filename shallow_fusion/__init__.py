"""
Shallow Fusion Module for Whisper ASR.

Provides trie-based shallow fusion for biasing Whisper's decoder towards
or away from specific token sequences during decoding. This is useful for:

- Boosting domain-specific terminology (medical, legal, technical terms)
- Suppressing unwanted outputs (profanity, slurs, hallucinations)
- Implementing hotword boosting for proper nouns and names
- Enforcing or discouraging specific phrasings

Architecture:
- TokenTrie: Efficient O(k) prefix matching data structure
- BiasList: Management of bias entries (text/tokens with lambdas)
- ShallowFusionProcessor: Main processor combining trie and configuration
- ShallowFusionLogitsProcessor: LogitsProcessor for Transformers' generation pipeline

Usage:
    from transformers import WhisperTokenizer, WhisperForConditionalGeneration
    from shallow_fusion import (
        BiasList,
        ShallowFusionProcessor,
        ShallowFusionConfig,
    )

    # Load model and tokenizer
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

    # Create bias list
    bias_list = BiasList(tokenizer)

    # Add biases (positive = boost, negative = suppress)
    bias_list.add_text("technical term", lambda_val=3.0)
    bias_list.add_text("unwanted phrase", lambda_val=-10.0)

    # Or load from file
    bias_list.load_from_json("biases.json")

    # Create processor with optional config
    config = ShallowFusionConfig(
        global_scale=1.0,
        terminal_only=True,
    )
    processor = ShallowFusionProcessor.from_bias_list(bias_list, config)

    # Get logits processor for generation
    logits_processor = processor.get_logits_processor()

    # Use with model.generate()
    outputs = model.generate(
        input_features,
        logits_processor=[logits_processor],
    )

    # Or use the convenience function
    from shallow_fusion import create_shallow_fusion_decoder
    processor, logits_processor = create_shallow_fusion_decoder(bias_list)
"""

# Trie data structures
from .trie import (
    TrieNode,
    TokenTrie,
    BatchTokenTrie,
)

# Bias list management
from .bias_list import (
    BiasEntry,
    BiasList,
)

# Shallow fusion processor and filter
from .shallow_fusion import (
    ShallowFusionConfig,
    ShallowFusionResult,
    ShallowFusionProcessor,
    ShallowFusionLogitsProcessor,
    CostSubtractionLogitsProcessor,
    ContextualShallowFusion,
    create_shallow_fusion_decoder,
)

__all__ = [
    # Trie
    "TrieNode",
    "TokenTrie",
    "BatchTokenTrie",
    # Bias list
    "BiasEntry",
    "BiasList",
    # Shallow fusion
    "ShallowFusionConfig",
    "ShallowFusionResult",
    "ShallowFusionProcessor",
    "ShallowFusionLogitsProcessor",
    "CostSubtractionLogitsProcessor",
    "ContextualShallowFusion",
    "create_shallow_fusion_decoder",
]
