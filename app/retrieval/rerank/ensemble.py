"""Ensemble scoring for combining multiple reranking methods."""

import numpy as np
from typing import List, Tuple, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EnsembleScorer:
    """
    Ensemble scorer for combining multiple reranking method scores.
    Handles score normalization, weighting, and final ranking.
    """

    def __init__(self):
        """Initialize ensemble scorer."""
        pass

    def combine_scores(
        self,
        candidates: List[Any],
        score_dict: Dict[str, List[Tuple[Any, float]]],
        weights: Dict[str, float],
        final_top_k: Optional[int] = None
    ) -> List[Tuple[Any, float]]:
        """
        Combine scores from multiple reranking methods.

        Args:
            candidates: Original list of candidates
            score_dict: Dictionary mapping method names to (candidate, score) lists
            weights: Dictionary mapping method names to weights
            final_top_k: Number of final results to return (None = no limit)

        Returns:
            List of (candidate, final_score) tuples sorted by score descending
        """
        try:
            if not candidates or not score_dict:
                return []

            logger.debug(
                f"Combining scores from {len(score_dict)} methods: {list(score_dict.keys())}")

            # Create candidate to index mapping
            candidate_to_idx = {
                id(cand): i for i, cand in enumerate(candidates)}

            # Initialize score matrix
            n_candidates = len(candidates)
            n_methods = len(score_dict)
            score_matrix = np.zeros((n_candidates, n_methods))
            method_names = list(score_dict.keys())

            # Populate score matrix
            for method_idx, (method_name, method_results) in enumerate(score_dict.items()):
                method_scores = np.zeros(n_candidates)

                # Fill scores for candidates that have scores from this method
                for candidate, score in method_results:
                    cand_idx = candidate_to_idx.get(id(candidate))
                    if cand_idx is not None:
                        method_scores[cand_idx] = score

                # Normalize scores to [0, 1] range
                normalized_scores = self._normalize_scores(method_scores)
                score_matrix[:, method_idx] = normalized_scores

                logger.debug(f"Method '{method_name}': {len(method_results)} scores, "
                             f"range [{normalized_scores.min():.3f}, {normalized_scores.max():.3f}]")

            # Apply weights
            weight_vector = np.array([weights.get(name, 0.0)
                                     for name in method_names])
            weighted_scores = score_matrix @ weight_vector

            # Handle case where all weights are zero
            if weight_vector.sum() == 0:
                logger.warning(
                    "All method weights are zero, using equal weights")
                weighted_scores = score_matrix.mean(axis=1)
            else:
                # Normalize by total weight
                weighted_scores = weighted_scores / weight_vector.sum()

            # Create final results
            results = []
            for i, candidate in enumerate(candidates):
                final_score = float(weighted_scores[i])
                results.append((candidate, final_score))

            # Sort by final score descending
            results.sort(key=lambda x: x[1], reverse=True)

            # Apply final_top_k limit: None = no limit, positive = limit
            if final_top_k is not None and final_top_k > 0:
                results = results[:final_top_k]
                logger.debug(f"Ensemble scoring completed, limited to top-{len(results)} scores: "
                             f"[{results[0][1]:.3f}, {results[-1][1]:.3f}]")
            else:
                logger.debug(f"Ensemble scoring completed, no limit applied, returning {len(results)} results: "
                             f"[{results[0][1]:.3f}, {results[-1][1]:.3f}]" if results else "no results")

            return results

        except Exception as e:
            logger.error(f"Ensemble scoring failed: {e}")
            # Fallback: return original candidates with zero scores
            return [(cand, 0.0) for cand in candidates[:final_top_k]]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: Array of scores

        Returns:
            Normalized scores
        """
        try:
            if len(scores) == 0:
                return scores

            min_score = scores.min()
            max_score = scores.max()

            if max_score > min_score:
                return (scores - min_score) / (max_score - min_score)
            else:
                # All scores are the same
                return np.ones_like(scores) * 0.5

        except Exception as e:
            logger.warning(f"Score normalization failed: {e}")
            return np.clip(scores, 0, 1)

    def aggregate_method_results(
        self,
        all_candidates: List[Any],
        method_results: Dict[str, List[Tuple[Any, float]]]
    ) -> Dict[Any, Dict[str, float]]:
        """
        Aggregate results from all methods into candidate-centric format.

        Args:
            all_candidates: Complete list of candidates
            method_results: Results from each method

        Returns:
            Dictionary mapping candidates to their scores from each method
        """
        try:
            candidate_scores = {}

            # Initialize all candidates with zero scores
            for candidate in all_candidates:
                candidate_scores[id(candidate)] = {
                    'candidate': candidate,
                    'scores': {}
                }

            # Populate scores from each method
            for method_name, results in method_results.items():
                for candidate, score in results:
                    cand_id = id(candidate)
                    if cand_id in candidate_scores:
                        candidate_scores[cand_id]['scores'][method_name] = score

            return candidate_scores

        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            return {}

    def compute_method_stats(
        self,
        method_results: Dict[str, List[Tuple[Any, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each method's scores.

        Args:
            method_results: Results from each method

        Returns:
            Dictionary of statistics for each method
        """
        stats = {}

        for method_name, results in method_results.items():
            if not results:
                stats[method_name] = {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
                continue

            scores = [score for _, score in results]
            scores_array = np.array(scores)

            stats[method_name] = {
                'count': len(scores),
                'mean': float(scores_array.mean()),
                'std': float(scores_array.std()),
                'min': float(scores_array.min()),
                'max': float(scores_array.max())
            }

        return stats
