# Shallow Fusion for Whisper

Trie-based shallow fusion module for biasing Whisper's decoder towards or away from specific token sequences during decoding.

## Overview

Shallow fusion combines Whisper's acoustic model with an external bias by modifying logits during decoding:

```
logits'[token] = logits[token] + λ * bias(token)
```

Where `λ` (lambda) is positive for boosting and negative for suppressing sequences.

## Quick Start

```python
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from shallow_fusion import BiasList, ShallowFusionProcessor

# Load Whisper model and tokenizer
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Create bias list
bias_list = BiasList(tokenizer)
bias_list.add_text("technical term", lambda_val=3.0)   # Boost
bias_list.add_text("unwanted phrase", lambda_val=-10.0) # Suppress

# Create shallow fusion processor
sf_processor = ShallowFusionProcessor.from_bias_list(bias_list)

# Get logits processor for generation
logits_processor = sf_processor.get_logits_processor()

# Use with model.generate()
input_features = processor(audio, return_tensors="pt").input_features
outputs = model.generate(input_features, logits_processor=[logits_processor])
transcription = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## Use Cases

### 1. Domain-Specific Terminology Boosting

Boost technical terms for better recognition in specialized domains:

```python
from transformers import WhisperTokenizer
from shallow_fusion import BiasList, ShallowFusionProcessor, ShallowFusionConfig

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

# Financial domain (earnings calls)
bias_list = BiasList(tokenizer)
financial_terms = [
    ("quarterly earnings", 3.0),
    ("revenue growth", 2.5),
    ("profit margin", 2.5),
    ("guidance forecast", 3.0),
    ("year over year", 2.0),
]
for term, boost in financial_terms:
    bias_list.add_text(term, lambda_val=boost)

# Medical domain
medical_terms = [
    ("myocardial infarction", 4.0),
    ("blood pressure", 2.0),
    ("prescription medication", 3.0),
]
for term, boost in medical_terms:
    bias_list.add_text(term, lambda_val=boost)
```

### 2. Hotword/Wake Word Boosting

Boost proper nouns, product names, or custom vocabulary:

```python
bias_list = BiasList(tokenizer)

# Company and product names
bias_list.add_word("Anthropic", lambda_val=5.0, include_variants=True)
bias_list.add_word("Claude", lambda_val=5.0, include_variants=True)

# Person names
names = ["John Smith", "Jane Doe", "Dr. Johnson"]
for name in names:
    bias_list.add_text(name, lambda_val=4.0)
```

### 3. Content Suppression

Suppress unwanted outputs like profanity or filler words:

```python
bias_list = BiasList(tokenizer)

# Suppress filler words
filler_words = ["um", "uh", "like", "you know"]
for word in filler_words:
    bias_list.add_word(word, lambda_val=-5.0)

# Strong suppression for profanity
profanity_list = ["word1", "word2"]  # Your list
for word in profanity_list:
    bias_list.add_word(word, lambda_val=-15.0)
```

### 4. Using with Transformers' Generation Pipeline

Integrate with Transformers' `model.generate()`:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Prepare audio input
input_features = whisper_processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features

# Create shallow fusion logits processor
sf_processor = ShallowFusionProcessor.from_bias_list(bias_list)
logits_processor = sf_processor.get_logits_processor()

# Generate with shallow fusion biasing
outputs = model.generate(
    input_features,
    logits_processor=[logits_processor],
    language="en",
    task="transcribe",
)

# Decode output
transcription = whisper_processor.batch_decode(outputs, skip_special_tokens=True)
```

## Configuration Options

```python
from shallow_fusion import ShallowFusionConfig

config = ShallowFusionConfig(
    # Scale all biases by this factor
    global_scale=1.0,

    # Only apply bias when sequence completes (recommended)
    terminal_only=True,

    # Clip biases to this range
    min_bias=-100.0,
    max_bias=100.0,

    # Normalize bias by sequence length
    length_normalize=False,

    # Temperature scaling (lower = sharper biases)
    temperature=1.0,
)

processor = ShallowFusionProcessor.from_bias_list(bias_list, config)
```

## Loading Biases from Files

### JSON Format

```json
{
  "entries": [
    {"text": "technical term", "lambda": 3.0, "category": "domain"},
    {"text": "suppress this", "lambda": -10.0, "category": "filter"},
    {"tokens": [1234, 5678], "lambda": 2.0}
  ]
}
```

### Simple Format

```json
{
  "quarterly earnings": 3.0,
  "revenue growth": 2.5,
  "bad word": -10.0
}
```

### Loading

```python
bias_list = BiasList(tokenizer)
bias_list.load_from_json("biases.json")

# Or save current biases
bias_list.save_to_json("exported_biases.json")
```

## Contextual Shallow Fusion

For dynamic bias control during decoding:

```python
from shallow_fusion.shallow_fusion import ContextualShallowFusion

sf_processor = ShallowFusionProcessor.from_bias_list(bias_list)
contextual = ContextualShallowFusion(sf_processor, sample_begin=0)

# Disable temporarily
contextual.set_active(False)

# Re-enable
contextual.set_active(True)

# Check which sequences were triggered
triggered = contextual.triggered_sequences
print(f"Triggered {len(triggered)} bias sequences")

# Reset tracking
contextual.reset()
```

## Batch Processing

For processing multiple audio files with the same biases:

```python
from shallow_fusion import BatchTokenTrie

# Create batch-optimized trie
trie = BatchTokenTrie()
for tokens, lambda_val in bias_list.to_token_lambda_pairs():
    trie.insert(tokens, lambda_val)

# Batch lookup
prefixes = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
all_biases = trie.batch_get_next_token_biases(prefixes)
```

## API Reference

### BiasList

| Method | Description |
|--------|-------------|
| `add_text(text, lambda_val)` | Add bias by text (tokenized automatically) |
| `add_tokens(tokens, lambda_val)` | Add bias by token IDs |
| `add_word(word, lambda_val, include_variants)` | Add word with case variants |
| `load_from_json(path)` | Load biases from JSON file |
| `save_to_json(path)` | Save biases to JSON file |
| `get_by_category(category)` | Get entries by category |
| `scale_category(category, scale)` | Scale all biases in category |

### ShallowFusionProcessor

| Method | Description |
|--------|-------------|
| `from_bias_list(bias_list, config)` | Create from BiasList |
| `add_sequence(tokens, lambda_val)` | Add single sequence |
| `get_next_token_biases(prefix)` | Get biases for next tokens |
| `get_logits_processor(sample_begin)` | Get LogitsProcessor for generation |
| `apply_biases(logits, prefix)` | Apply biases to logits tensor |

### ShallowFusionLogitsProcessor

| Method | Description |
|--------|-------------|
| `__call__(input_ids, scores)` | Apply biases (called by generate()) |
| `last_result` | Get result from last invocation |

## Lambda Value Guidelines

| Use Case | Suggested λ Range |
|----------|-------------------|
| Gentle boost | +1.0 to +2.0 |
| Strong boost | +3.0 to +5.0 |
| Hotword/name | +4.0 to +6.0 |
| Gentle suppress | -2.0 to -5.0 |
| Strong suppress | -10.0 to -15.0 |
| Hard block | -50.0 to -100.0 |

## Beam Search Compatibility: Cost Subtraction

When using shallow fusion with beam search (`num_beams > 1`), a naive approach can cause content skipping issues. The standard `ShallowFusionLogitsProcessor` applies biases independently at each step, which can cause beam search to:

1. Create alternative hypotheses that "jump" to biased content
2. Skip over unbiased audio segments to reach boosted phrases
3. Merge timestamps incorrectly when biases pull toward specific content

### The Problem

Consider transcribing: "Hey Jessica, you won't believe what happened with the myocardial infarction case..."

With beam search and a boosted term "myocardial infarction" (λ=4.0):
- Beam 1 might correctly transcribe "Hey Jessica..."
- Beam 2 gets boosted toward "myocardial" early and skips content
- When beams are compared, the biased beam may win despite skipping audio

This manifests as:
- First segment spanning 0.0s-15.0s instead of 0.0s-2.7s
- Missing greeting text like "Hey Jessica, you won't believe..."
- Higher WER despite biasing intended to improve accuracy

### The Solution: Cost Subtraction

Cost subtraction tracks accumulated bias per beam and **revokes** (subtracts) the bias when a beam breaks from the trie path. This prevents beams from gaining unfair advantage by jumping to biased content.

```python
from shallow_fusion import ShallowFusionProcessor, BiasList

# Create bias list
bias_list = BiasList(tokenizer)
bias_list.add_text("myocardial infarction", lambda_val=4.0)
bias_list.add_text("blood pressure", lambda_val=3.0)

# Create processor
sf_processor = ShallowFusionProcessor.from_bias_list(bias_list)

# Use cost subtraction processor for beam search
cost_sub_processor = sf_processor.get_cost_subtraction_processor(sample_begin=4)

# Generate with beam search
outputs = model.generate(
    input_features,
    num_beams=5,
    logits_processor=[cost_sub_processor],
    return_timestamps=True,
)
```

### How Cost Subtraction Works

1. **Track**: For each beam, track the current trie path and accumulated bias
2. **Apply**: Apply biases normally when beam extends a valid trie path
3. **Revoke**: When a beam token breaks from the trie path (doesn't continue the biased sequence), subtract the accumulated bias from scores
4. **Reset**: Reset tracking when the beam starts a new potential match

```
Step 1: Beam generates "my" → matches trie → accumulate +2.0
Step 2: Beam generates "ocardial" → matches trie → accumulate +2.0 (total: +4.0)
Step 3: Beam generates "the" → BREAKS trie path → subtract -4.0 from scores
```

This ensures beams that legitimately complete biased phrases keep their boost, while beams that "jump around" get penalized.

### When to Use Each Processor

| Scenario | Processor | Method |
|----------|-----------|--------|
| Greedy decoding (`num_beams=1`) | `ShallowFusionLogitsProcessor` | `get_logits_processor()` |
| Beam search (`num_beams > 1`) | `CostSubtractionLogitsProcessor` | `get_cost_subtraction_processor()` |
| Dynamic control needed | `ContextualShallowFusion` | Direct instantiation |

### API Reference

```python
# CostSubtractionLogitsProcessor
processor = sf_processor.get_cost_subtraction_processor(sample_begin=4)

# Parameters:
#   sample_begin: Token index where content generation begins (after language/task tokens)
#                 Default: 0. For Whisper, typically 4 (after <|startoftranscript|><|en|><|transcribe|><|notimestamps|>)
```

### Limitations

- Slightly higher computational overhead due to per-beam state tracking
- Beam reordering (when beams are pruned/reordered) may cause tracking discontinuities
- Best results when `sample_begin` is correctly set for your model

## Performance Notes

- Trie lookup is O(k) where k is prefix length
- Memory scales with number of unique token sequences
- For large bias lists (>10k entries), consider category-based loading
- Batch operations available for multi-sequence processing
- Cost subtraction adds O(b) overhead where b is number of beams
