"""
Bias List Management for Shallow Fusion.

Provides classes for defining, loading, and managing bias entries
that control which token sequences are boosted or suppressed during
Whisper decoding. Uses Transformers WhisperTokenizer for tokenization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import json


@dataclass
class BiasEntry:
    """
    A single bias entry representing a token sequence and its bias weight.

    Attributes:
        tokens: List of token IDs representing the sequence
        lambda_val: Bias weight (positive = boost, negative = suppress)
        text: Optional human-readable text representation
        category: Optional category for grouping biases (e.g., "slur", "profanity")
    """

    tokens: List[int]
    lambda_val: float
    text: Optional[str] = None
    category: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.tokens, list):
            self.tokens = list(self.tokens)

    @property
    def is_boost(self) -> bool:
        """Whether this bias boosts the sequence probability."""
        return self.lambda_val > 0

    @property
    def is_suppress(self) -> bool:
        """Whether this bias suppresses the sequence probability."""
        return self.lambda_val < 0


class BiasList:
    """
    Collection of bias entries for shallow fusion.

    Manages a list of token sequences with their associated bias values.
    Supports loading from files, adding entries programmatically, and
    converting text to tokens using a tokenizer.

    Example:
        from transformers import WhisperTokenizer

        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
        bias_list = BiasList(tokenizer)

        # Add biases by text (will be tokenized)
        bias_list.add_text("hello world", lambda_val=0.5)
        bias_list.add_text("bad word", lambda_val=-1.0)

        # Add biases by token IDs directly
        bias_list.add_tokens([1234, 5678], lambda_val=0.3)

        # Load from file
        bias_list.load_from_json("biases.json")
    """

    def __init__(self, tokenizer=None):
        """
        Initialize the bias list.

        Args:
            tokenizer: Optional tokenizer (e.g., WhisperTokenizer) for text-to-token conversion.
                       If not provided, only token-based operations are available.
        """
        self.tokenizer = tokenizer
        self._entries: List[BiasEntry] = []
        self._text_to_entry: Dict[str, BiasEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> List[BiasEntry]:
        """Get all bias entries."""
        return self._entries.copy()

    def add_tokens(
        self,
        tokens: List[int],
        lambda_val: float,
        text: Optional[str] = None,
        category: Optional[str] = None,
    ) -> BiasEntry:
        """
        Add a bias entry using token IDs directly.

        Args:
            tokens: List of token IDs
            lambda_val: Bias weight
            text: Optional text representation
            category: Optional category

        Returns:
            The created BiasEntry
        """
        entry = BiasEntry(
            tokens=tokens,
            lambda_val=lambda_val,
            text=text,
            category=category,
        )
        self._entries.append(entry)
        if text:
            self._text_to_entry[text] = entry
        return entry

    def add_text(
        self,
        text: str,
        lambda_val: float,
        category: Optional[str] = None,
    ) -> BiasEntry:
        """
        Add a bias entry using text (will be tokenized).

        Args:
            text: Text to tokenize and bias
            lambda_val: Bias weight
            category: Optional category

        Returns:
            The created BiasEntry

        Raises:
            ValueError: If no tokenizer is available
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text-based bias. "
                           "Initialize BiasList with a tokenizer.")

        # Use add_special_tokens=False to avoid wrapping with SOT/EOT tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return self.add_tokens(
            tokens=tokens,
            lambda_val=lambda_val,
            text=text,
            category=category,
        )

    def add_word(
        self,
        word: str,
        lambda_val: float,
        category: Optional[str] = None,
        include_variants: bool = True,
    ) -> List[BiasEntry]:
        """
        Add bias entries for a word including common variants.

        Optionally includes variants with different casing and spacing.

        Args:
            word: The word to bias
            lambda_val: Bias weight
            category: Optional category
            include_variants: Whether to include case/spacing variants

        Returns:
            List of created BiasEntry objects
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for word-based bias.")

        entries = []
        variants = [word]

        if include_variants:
            # Add common variants
            variants.extend([
                word.lower(),
                word.upper(),
                word.capitalize(),
                f" {word}",        # With leading space (common in BPE)
                f" {word.lower()}",
                f" {word.capitalize()}",
            ])
            # Remove duplicates while preserving order
            variants = list(dict.fromkeys(variants))

        for variant in variants:
            try:
                entry = self.add_text(variant, lambda_val, category)
                entries.append(entry)
            except Exception:
                # Skip variants that fail to tokenize
                pass

        return entries

    def remove_text(self, text: str) -> bool:
        """
        Remove a bias entry by its text.

        Args:
            text: Text of the entry to remove

        Returns:
            True if entry was found and removed, False otherwise
        """
        if text in self._text_to_entry:
            entry = self._text_to_entry.pop(text)
            self._entries.remove(entry)
            return True
        return False

    def get_by_category(self, category: str) -> List[BiasEntry]:
        """Get all entries in a specific category."""
        return [e for e in self._entries if e.category == category]

    def scale_category(self, category: str, scale: float) -> int:
        """
        Scale all lambda values in a category.

        Args:
            category: Category to scale
            scale: Multiplier for lambda values

        Returns:
            Number of entries scaled
        """
        count = 0
        for entry in self._entries:
            if entry.category == category:
                entry.lambda_val *= scale
                count += 1
        return count

    def load_from_json(self, path: Union[str, Path]) -> int:
        """
        Load bias entries from a JSON file.

        Expected format:
        {
            "entries": [
                {"text": "word", "lambda": 0.5, "category": "optional"},
                {"tokens": [1, 2, 3], "lambda": -0.3}
            ]
        }

        Or simple format:
        {
            "word1": 0.5,
            "word2": -0.3
        }

        Args:
            path: Path to JSON file

        Returns:
            Number of entries loaded
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        count = 0

        # Check for structured format
        if "entries" in data:
            for entry_data in data["entries"]:
                if "tokens" in entry_data:
                    self.add_tokens(
                        tokens=entry_data["tokens"],
                        lambda_val=entry_data.get("lambda", entry_data.get("lambda_val", 0.0)),
                        text=entry_data.get("text"),
                        category=entry_data.get("category"),
                    )
                elif "text" in entry_data:
                    self.add_text(
                        text=entry_data["text"],
                        lambda_val=entry_data.get("lambda", entry_data.get("lambda_val", 0.0)),
                        category=entry_data.get("category"),
                    )
                count += 1
        else:
            # Simple format: {"text": lambda_val}
            for text, lambda_val in data.items():
                if isinstance(lambda_val, (int, float)):
                    self.add_text(text, float(lambda_val))
                    count += 1

        return count

    def save_to_json(self, path: Union[str, Path]) -> None:
        """
        Save bias entries to a JSON file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        entries_data = []

        for entry in self._entries:
            entry_data = {
                "tokens": entry.tokens,
                "lambda": entry.lambda_val,
            }
            if entry.text:
                entry_data["text"] = entry.text
            if entry.category:
                entry_data["category"] = entry.category
            entries_data.append(entry_data)

        with open(path, "w") as f:
            json.dump({"entries": entries_data}, f, indent=2)

    def to_token_lambda_pairs(self) -> List[Tuple[List[int], float]]:
        """
        Convert to list of (tokens, lambda) pairs for trie insertion.

        Returns:
            List of (token_sequence, lambda_value) tuples
        """
        return [(entry.tokens, entry.lambda_val) for entry in self._entries]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._text_to_entry.clear()

    @classmethod
    def from_word_list(
        cls,
        words: List[str],
        lambda_val: float,
        tokenizer,
        category: Optional[str] = None,
        include_variants: bool = True,
    ) -> "BiasList":
        """
        Create a BiasList from a list of words with the same lambda.

        Convenience method for creating suppression or boost lists.

        Args:
            words: List of words to bias
            lambda_val: Bias weight to apply to all words
            tokenizer: Tokenizer instance (e.g., WhisperTokenizer)
            category: Optional category for all entries
            include_variants: Whether to include case/spacing variants

        Returns:
            New BiasList with all words added
        """
        bias_list = cls(tokenizer)
        for word in words:
            bias_list.add_word(word, lambda_val, category, include_variants)
        return bias_list

    @classmethod
    def create_suppression_list(
        cls,
        words: List[str],
        tokenizer,
        suppression_strength: float = -10.0,
        category: str = "suppressed",
    ) -> "BiasList":
        """
        Create a BiasList for suppressing specific words.

        Convenience method for creating word suppression lists.

        Args:
            words: Words to suppress
            tokenizer: Tokenizer instance (e.g., WhisperTokenizer)
            suppression_strength: Negative lambda value (more negative = stronger suppression)
            category: Category name

        Returns:
            New BiasList configured for suppression
        """
        return cls.from_word_list(
            words=words,
            lambda_val=suppression_strength,
            tokenizer=tokenizer,
            category=category,
            include_variants=True,
        )
