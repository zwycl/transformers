"""
Trie Data Structure for Token Sequence Prefix Matching.

Provides efficient O(k) lookup for prefix matching of token sequences,
where k is the length of the query sequence. Used by shallow fusion to
quickly determine which biases apply at each decoding step.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrieNode:
    """A node in the trie representing a position in token sequences."""

    # Map from token ID to child node
    children: Dict[int, "TrieNode"] = field(default_factory=dict)

    # If this node represents the end of a bias sequence, store the lambda value
    # None means this is not an endpoint
    terminal_lambda: Optional[float] = None

    # Store all lambdas for sequences that pass through this node
    # This allows applying partial biases during decoding
    prefix_lambdas: List[float] = field(default_factory=list)

    # Whether this node is a terminal (end of a bias sequence)
    @property
    def is_terminal(self) -> bool:
        return self.terminal_lambda is not None


class TokenTrie:
    """
    Trie for efficient prefix matching of token sequences.

    Supports:
    - Inserting token sequences with associated lambda (bias) values
    - Looking up the bias for a given prefix
    - Finding the next tokens that continue valid sequences
    - Batch operations for efficiency

    Example:
        trie = TokenTrie()
        trie.insert([1, 2, 3], lambda_val=0.5)  # Boost sequence [1,2,3]
        trie.insert([1, 2, 4], lambda_val=-0.3) # Suppress sequence [1,2,4]

        # During decoding after seeing [1, 2]:
        next_biases = trie.get_next_token_biases([1, 2])
        # Returns {3: 0.5, 4: -0.3}
    """

    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def __len__(self) -> int:
        """Return the number of sequences in the trie."""
        return self._size

    def insert(self, tokens: List[int], lambda_val: float) -> None:
        """
        Insert a token sequence with its associated bias value.

        Args:
            tokens: List of token IDs representing the sequence to bias
            lambda_val: The bias value (positive = boost, negative = suppress)
        """
        if not tokens:
            return

        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            node.prefix_lambdas.append(lambda_val)

        node.terminal_lambda = lambda_val
        self._size += 1

    def search(self, tokens: List[int]) -> Optional[float]:
        """
        Search for an exact token sequence match.

        Args:
            tokens: Token sequence to search for

        Returns:
            The lambda value if the exact sequence exists, None otherwise
        """
        node = self._traverse(tokens)
        if node is None:
            return None
        return node.terminal_lambda

    def starts_with(self, prefix: List[int]) -> bool:
        """
        Check if any sequence in the trie starts with the given prefix.

        Args:
            prefix: Token sequence prefix to check

        Returns:
            True if at least one sequence starts with this prefix
        """
        return self._traverse(prefix) is not None

    def _traverse(self, tokens: List[int]) -> Optional[TrieNode]:
        """
        Traverse the trie following the given token sequence.

        Args:
            tokens: Token sequence to follow

        Returns:
            The node reached after following the sequence, or None if path doesn't exist
        """
        node = self.root
        for token in tokens:
            if token not in node.children:
                return None
            node = node.children[token]
        return node

    def get_next_token_biases(self, prefix: List[int]) -> Dict[int, float]:
        """
        Get biases for all possible next tokens given a prefix.

        This is the key method for shallow fusion during decoding.
        Given the tokens decoded so far, returns a mapping from possible
        next tokens to their bias values.

        Args:
            prefix: Token sequence decoded so far

        Returns:
            Dict mapping token IDs to their bias values for valid continuations
        """
        node = self._traverse(prefix)
        if node is None:
            return {}

        biases = {}
        for token_id, child in node.children.items():
            # Use terminal lambda if this is end of sequence,
            # otherwise use the strongest bias from sequences passing through
            if child.is_terminal:
                biases[token_id] = child.terminal_lambda
            elif child.prefix_lambdas:
                # For non-terminal nodes, use the strongest bias
                # (most negative for suppression, most positive for boosting)
                lambdas = child.prefix_lambdas
                # Take the most extreme value
                max_pos = max((l for l in lambdas if l > 0), default=0)
                min_neg = min((l for l in lambdas if l < 0), default=0)
                if abs(max_pos) >= abs(min_neg):
                    biases[token_id] = max_pos
                else:
                    biases[token_id] = min_neg

        return biases

    def get_completion_bias(self, tokens: List[int]) -> float:
        """
        Get the bias if the token sequence completes a biased sequence.

        Args:
            tokens: Complete token sequence

        Returns:
            The bias value if this completes a sequence, 0.0 otherwise
        """
        node = self._traverse(tokens)
        if node is None or not node.is_terminal:
            return 0.0
        return node.terminal_lambda

    def get_all_continuations(self, prefix: List[int]) -> List[Tuple[List[int], float]]:
        """
        Get all sequences that continue from the given prefix.

        Args:
            prefix: Starting prefix

        Returns:
            List of (remaining_tokens, lambda_val) tuples
        """
        node = self._traverse(prefix)
        if node is None:
            return []

        results = []
        self._collect_continuations(node, [], results)
        return results

    def _collect_continuations(
        self,
        node: TrieNode,
        current_path: List[int],
        results: List[Tuple[List[int], float]],
    ) -> None:
        """Recursively collect all continuations from a node."""
        if node.is_terminal:
            results.append((current_path.copy(), node.terminal_lambda))

        for token_id, child in node.children.items():
            current_path.append(token_id)
            self._collect_continuations(child, current_path, results)
            current_path.pop()

    def clear(self) -> None:
        """Remove all entries from the trie."""
        self.root = TrieNode()
        self._size = 0


class BatchTokenTrie(TokenTrie):
    """
    Extended trie with batch operations for efficiency.

    Optimized for cases where multiple sequences need to be checked
    simultaneously during batch decoding.
    """

    def batch_get_next_token_biases(
        self, prefixes: List[List[int]]
    ) -> List[Dict[int, float]]:
        """
        Get next token biases for multiple prefixes at once.

        Args:
            prefixes: List of token sequence prefixes

        Returns:
            List of bias dictionaries, one per prefix
        """
        return [self.get_next_token_biases(prefix) for prefix in prefixes]

    def batch_get_completion_bias(self, sequences: List[List[int]]) -> List[float]:
        """
        Get completion biases for multiple sequences at once.

        Args:
            sequences: List of token sequences

        Returns:
            List of bias values
        """
        return [self.get_completion_bias(seq) for seq in sequences]
