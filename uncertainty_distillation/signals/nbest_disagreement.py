"""
N-Best Hypothesis Disagreement Signal Extractor (Signal 2)

Computes disagreement among top-N beam search hypotheses as a proxy for aleatoric
uncertainty. High disagreement indicates multiple plausible interpretations of
the audio input.

Uses Transformers WhisperForConditionalGeneration API with num_return_sequences
for efficient N-best generation.

From the research plan:
    "High N-best disagreement → multiple plausible transcriptions
     → aleatoric-like (inherent ambiguity in audio)"

Requirements:
    pip install rapidfuzz

Interpretation:
- High disagreement → multiple valid transcriptions → aleatoric-like uncertainty
- Low disagreement → clear consensus → confident prediction
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from transformers import WhisperForConditionalGeneration, WhisperProcessor


@dataclass
class NBestDisagreementResult:
    """Results from N-best disagreement computation."""

    # Per-position disagreement scores (word-level), shape: [max_len] or [batch, max_len]
    position_disagreement: Tensor

    # Utterance-level disagreement, shape: scalar or [batch]
    utterance_disagreement: Tensor

    # Raw hypotheses for each batch item
    hypotheses: List[List[str]]

    # Token-level disagreement aligned with decoder tokens, shape: [seq_len] or [batch, seq_len]
    token_disagreement: Optional[Tensor] = None

    # Token sequences for each hypothesis (for alignment), List of List of token ids
    token_sequences: Optional[List[List[List[int]]]] = None

    # Log probabilities for each hypothesis
    hypothesis_scores: Optional[List[List[float]]] = None

    # Pairwise distance matrix between hypotheses
    pairwise_distances: Optional[Tensor] = None


def compute_nbest_disagreement(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    input_features: Tensor,
    n_beams: int = 5,
    language: Optional[str] = "en",
) -> List[float]:
    """
    Compute N-best hypothesis disagreement.

    Simple functional interface matching the research plan pseudocode.

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor for decoding
        input_features: Input features from processor
        n_beams: Number of beam hypotheses
        language: Language code

    Returns:
        List of disagreement scores per position
    """
    extractor = NBestDisagreementExtractor(model, processor, n_beams=n_beams, language=language)
    result = extractor.extract(input_features)
    return result.position_disagreement.tolist()


def align_hypotheses(hypotheses: List[str]) -> List[List[Optional[str]]]:
    """
    Align multiple hypotheses using rapidfuzz's Levenshtein operations.

    Uses word-level alignment via edit operations.

    Args:
        hypotheses: List of hypothesis strings

    Returns:
        List of aligned word sequences (with None for gaps)
    """
    if len(hypotheses) == 0:
        return []

    # Convert to word lists
    word_seqs = [h.split() for h in hypotheses]

    if len(word_seqs) == 1:
        return [word_seqs[0]]

    # Use first as reference, align all others
    reference = word_seqs[0]
    aligned_all = [list(reference)]

    for seq in word_seqs[1:]:
        # Get Levenshtein edit operations using rapidfuzz
        ref_str = " ".join(reference)
        seq_str = " ".join(seq)
        opcodes = Levenshtein.opcodes(ref_str, seq_str)

        # Build aligned sequences from opcodes
        aligned_ref = []
        aligned_seq = []

        for op, i1, i2, j1, j2 in opcodes:
            ref_part = ref_str[i1:i2]
            seq_part = seq_str[j1:j2]

            if op == "equal":
                # Both match
                ref_words = ref_part.split() if ref_part else []
                seq_words = seq_part.split() if seq_part else []
                aligned_ref.extend(ref_words)
                aligned_seq.extend(seq_words)
            elif op == "replace":
                # Substitution - align word by word
                ref_words = ref_part.split() if ref_part else []
                seq_words = seq_part.split() if seq_part else []
                max_len = max(len(ref_words), len(seq_words))
                for k in range(max_len):
                    aligned_ref.append(ref_words[k] if k < len(ref_words) else None)
                    aligned_seq.append(seq_words[k] if k < len(seq_words) else None)
            elif op == "delete":
                # In reference but not in seq
                ref_words = ref_part.split() if ref_part else []
                for w in ref_words:
                    aligned_ref.append(w)
                    aligned_seq.append(None)
            elif op == "insert":
                # In seq but not in reference
                seq_words = seq_part.split() if seq_part else []
                for w in seq_words:
                    aligned_ref.append(None)
                    aligned_seq.append(w)

        aligned_all.append(aligned_seq)

    # Ensure all same length
    max_len = max(len(a) for a in aligned_all)
    for a in aligned_all:
        while len(a) < max_len:
            a.append(None)

    return aligned_all


class NBestDisagreementExtractor:
    """
    Extract N-best hypothesis disagreement from Whisper beam search.

    Uses Transformers' generate() with num_return_sequences for efficient
    N-best generation, and rapidfuzz for string distance computations.

    Args:
        model: The WhisperForConditionalGeneration model instance
        processor: WhisperProcessor for tokenization/decoding
        n_beams: Number of beam search hypotheses to generate
        language: Language for transcription (None for auto-detect)
        task: Task type ('transcribe' or 'translate')
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        n_beams: int = 5,
        language: Optional[str] = "en",
        task: str = "transcribe",
    ):
        self.model = model
        self.processor = processor
        self.n_beams = n_beams
        self.language = language
        self.task = task

    @staticmethod
    def compute_position_disagreement(
        aligned_hypotheses: List[List[Optional[str]]],
        n_beams: int,
    ) -> Tensor:
        """
        Compute disagreement at each aligned position.

        From research plan:
            unique_words = len(set(position))
            disagreement.append(unique_words / n_beams)

        Args:
            aligned_hypotheses: Aligned hypothesis sequences
            n_beams: Number of hypotheses

        Returns:
            Disagreement scores in [0, 1] for each position
        """
        if not aligned_hypotheses or not aligned_hypotheses[0]:
            return torch.tensor([0.0])

        n_pos = len(aligned_hypotheses[0])
        disagreement = []

        for pos in range(n_pos):
            values_at_pos = [h[pos] for h in aligned_hypotheses if pos < len(h)]
            unique_words = len(set(v for v in values_at_pos if v is not None))

            if None in values_at_pos:
                unique_words += 1

            score = unique_words / n_beams
            disagreement.append(score)

        return torch.tensor(disagreement, dtype=torch.float32)

    @staticmethod
    def compute_pairwise_distances(hypotheses: List[str]) -> Tensor:
        """
        Compute pairwise normalized Levenshtein distances using rapidfuzz.cdist.

        Args:
            hypotheses: List of hypothesis strings

        Returns:
            Distance matrix, shape [n_hyps, n_hyps], values in [0, 1]
        """
        # Use rapidfuzz cdist for efficient pairwise computation
        # Returns similarity scores (0-100), convert to normalized distance
        similarity_matrix = cdist(
            hypotheses,
            hypotheses,
            scorer=fuzz.ratio,
            dtype=float,
        )

        # Convert similarity (0-100) to distance (0-1)
        distance_matrix = 1.0 - (similarity_matrix / 100.0)

        return torch.tensor(distance_matrix, dtype=torch.float32)

    @staticmethod
    def compute_diversity_score(hypotheses: List[str]) -> float:
        """
        Compute overall diversity score from pairwise distances.

        Higher score means more diverse hypotheses (more aleatoric uncertainty).

        Args:
            hypotheses: List of hypothesis strings

        Returns:
            Mean pairwise distance (0 = identical, 1 = completely different)
        """
        if len(hypotheses) < 2:
            return 0.0

        distances = NBestDisagreementExtractor.compute_pairwise_distances(hypotheses)

        # Get upper triangle (excluding diagonal)
        n = len(hypotheses)
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        pairwise_dists = distances[mask]

        return pairwise_dists.mean().item()

    @torch.no_grad()
    def extract(
        self,
        input_features: Tensor,
        return_hypotheses: bool = True,
        return_pairwise: bool = False,
        return_token_sequences: bool = False,
    ) -> NBestDisagreementResult:
        """
        Extract N-best disagreement for given audio features.

        Implements Signal 2 from the research plan using Transformers' generate()
        with num_return_sequences for efficient N-best generation.

        Args:
            input_features: Input features from processor, shape [batch, n_mels, n_frames]
                or [n_mels, n_frames]
            return_hypotheses: Whether to return raw text hypotheses
            return_pairwise: Whether to compute pairwise distance matrix
            return_token_sequences: Whether to return token sequences

        Returns:
            NBestDisagreementResult containing disagreement values
        """
        single_batch = input_features.ndim == 2
        if single_batch:
            input_features = input_features.unsqueeze(0)

        batch_size = input_features.shape[0]

        all_position_disagreements = []
        all_token_disagreements = []
        all_utterance_disagreements = []
        all_hypotheses = []
        all_token_seqs = []
        all_scores = []
        all_pairwise = []

        # Get decoder prompt IDs
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language,
            task=self.task,
            no_timestamps=True,
        )

        # Process each sample (beam search doesn't batch well across samples)
        for b in range(batch_size):
            sample_features = input_features[b:b+1]

            hypotheses, scores, token_seqs = self._get_nbest_hypotheses(
                sample_features, forced_decoder_ids
            )

            if return_hypotheses:
                all_hypotheses.append(hypotheses)
            if return_token_sequences:
                all_token_seqs.append(token_seqs)
            all_scores.append(scores)

            # Align hypotheses (word-level)
            aligned = align_hypotheses(hypotheses)

            # Compute word-level position disagreement
            pos_disagreement = self.compute_position_disagreement(
                aligned, self.n_beams
            )
            all_position_disagreements.append(pos_disagreement)

            # Compute token-level disagreement
            token_disagreement = self.compute_token_disagreement(
                token_seqs, self.n_beams
            )
            all_token_disagreements.append(token_disagreement)

            # Utterance-level: use diversity score from pairwise distances
            diversity = self.compute_diversity_score(hypotheses)
            all_utterance_disagreements.append(diversity)

            if return_pairwise:
                pairwise = self.compute_pairwise_distances(hypotheses)
                all_pairwise.append(pairwise)

        # Pad position disagreements (word-level)
        max_len = max(len(d) for d in all_position_disagreements) if all_position_disagreements else 1
        padded = torch.zeros(batch_size, max_len)
        for b, d in enumerate(all_position_disagreements):
            padded[b, : len(d)] = d

        # Pad token disagreements
        max_token_len = max(len(d) for d in all_token_disagreements) if all_token_disagreements else 1
        padded_tokens = torch.zeros(batch_size, max_token_len)
        for b, d in enumerate(all_token_disagreements):
            padded_tokens[b, : len(d)] = d

        utterance = torch.tensor(all_utterance_disagreements)

        if single_batch:
            padded = padded.squeeze(0)
            padded_tokens = padded_tokens.squeeze(0)
            utterance = utterance.squeeze(0)
            all_hypotheses = all_hypotheses[0] if all_hypotheses else []
            all_token_seqs = all_token_seqs[0] if all_token_seqs else None
            all_scores = all_scores[0] if all_scores else []
            all_pairwise = all_pairwise[0] if all_pairwise else None

        return NBestDisagreementResult(
            position_disagreement=padded,
            utterance_disagreement=utterance,
            hypotheses=all_hypotheses if return_hypotheses else [],
            token_disagreement=padded_tokens,
            token_sequences=all_token_seqs if return_token_sequences else None,
            hypothesis_scores=all_scores,
            pairwise_distances=all_pairwise if return_pairwise else None,
        )

    def _get_nbest_hypotheses(
        self,
        input_features: Tensor,
        forced_decoder_ids: Optional[List] = None,
    ) -> Tuple[List[str], List[float], List[List[int]]]:
        """Get N-best hypotheses from beam search using generate()."""

        # Generate with beam search and return multiple sequences
        # Use max_new_tokens that accounts for decoder prompt length
        outputs = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            num_beams=self.n_beams,
            num_return_sequences=self.n_beams,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=440,
        )

        # outputs.sequences shape: (n_beams, seq_len)
        generated_ids = outputs.sequences

        # Decode all hypotheses
        hypotheses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Extract token sequences (excluding special tokens)
        token_seqs = []
        eos_token_id = self.model.config.eos_token_id
        for seq in generated_ids:
            # Find where content starts (after forced decoder ids)
            start_idx = len(forced_decoder_ids) + 1 if forced_decoder_ids else 1
            cleaned = [t.item() for t in seq[start_idx:] if t.item() != eos_token_id]
            token_seqs.append(cleaned)

        # Get sequence scores if available
        scores = []
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            scores = outputs.sequences_scores.tolist()
        else:
            scores = [0.0] * len(hypotheses)

        return (
            hypotheses[: self.n_beams],
            scores[: self.n_beams],
            token_seqs[: self.n_beams],
        )

    @staticmethod
    def compute_token_disagreement(
        token_sequences: List[List[int]],
        n_beams: int,
    ) -> Tensor:
        """
        Compute token-level disagreement across N-best hypotheses.

        Aligns token sequences and computes disagreement at each position.

        Args:
            token_sequences: List of token id sequences from each hypothesis
            n_beams: Number of hypotheses

        Returns:
            Disagreement scores per token position, shape [max_len]
        """
        if not token_sequences:
            return torch.tensor([0.0])

        # Use the first (best) hypothesis as reference length
        ref_len = len(token_sequences[0])
        if ref_len == 0:
            return torch.tensor([0.0])

        # Align sequences using dynamic programming
        aligned_seqs = align_token_sequences(token_sequences)

        # Compute disagreement at each position
        n_pos = len(aligned_seqs[0]) if aligned_seqs else 0
        disagreement = []

        for pos in range(n_pos):
            tokens_at_pos = [seq[pos] for seq in aligned_seqs if pos < len(seq)]

            # Count unique tokens (including None for gaps)
            unique_tokens = set(tokens_at_pos)
            n_unique = len(unique_tokens)

            # Disagreement: (n_unique - 1) / (n_beams - 1) if n_beams > 1
            # This gives 0 when all agree, 1 when all different
            if n_beams > 1:
                score = (n_unique - 1) / (n_beams - 1)
            else:
                score = 0.0

            disagreement.append(score)

        return torch.tensor(disagreement, dtype=torch.float32)


def align_token_sequences(
    sequences: List[List[int]],
) -> List[List[Optional[int]]]:
    """
    Align multiple token sequences using progressive pairwise alignment.

    Uses the first sequence as reference and aligns all others to it,
    properly inserting gaps into all sequences to maintain alignment.

    Args:
        sequences: List of token id sequences

    Returns:
        List of aligned sequences with None for gaps
    """
    if len(sequences) == 0:
        return []
    if len(sequences) == 1:
        return [list(sequences[0])]

    # Start with first sequence as the "master" alignment
    # We'll progressively add gaps as we align each new sequence
    master_alignment: List[List[Optional[int]]] = [[t for t in sequences[0]]]

    for seq in sequences[1:]:
        # Align current master reference (first sequence) with new sequence
        # Get the current reference (with any gaps already inserted)
        current_ref = [t for t in master_alignment[0] if t is not None]

        aligned_ref, aligned_new = _align_two_sequences(current_ref, list(seq))

        # Now we need to update all existing alignments to match the new aligned_ref
        # Find where gaps were inserted into the reference
        new_master: List[List[Optional[int]]] = []

        for existing_seq in master_alignment:
            new_aligned: List[Optional[int]] = []
            existing_pos = 0

            for ref_token in aligned_ref:
                if ref_token is None:
                    # Gap inserted into reference - insert gap into all existing sequences
                    new_aligned.append(None)
                else:
                    # Real token from reference
                    # Find corresponding position in existing sequence
                    # Skip gaps in existing sequence until we find the real token
                    while existing_pos < len(existing_seq) and existing_seq[existing_pos] is None:
                        new_aligned.append(None)
                        existing_pos += 1

                    if existing_pos < len(existing_seq):
                        new_aligned.append(existing_seq[existing_pos])
                        existing_pos += 1
                    else:
                        new_aligned.append(None)

            # Append any remaining gaps from existing sequence
            while existing_pos < len(existing_seq):
                new_aligned.append(existing_seq[existing_pos])
                existing_pos += 1

            new_master.append(new_aligned)

        # Add the newly aligned sequence
        new_master.append(aligned_new)
        master_alignment = new_master

    # Ensure all same length
    max_len = max(len(a) for a in master_alignment)
    for a in master_alignment:
        while len(a) < max_len:
            a.append(None)

    return master_alignment


def _align_two_sequences(
    seq1: List[int], seq2: List[int]
) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    """
    Align two token sequences using dynamic programming.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Tuple of aligned sequences with None for gaps
    """
    n, m = len(seq1), len(seq2)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    aligned1: List[Optional[int]] = []
    aligned2: List[Optional[int]] = []

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + 1):
            aligned1.append(None)
            aligned2.append(seq2[j - 1])
            j -= 1
        else:
            aligned1.append(seq1[i - 1])
            aligned2.append(None)
            i -= 1

    aligned1.reverse()
    aligned2.reverse()

    return aligned1, aligned2
