"""SuperGlobal reranking implementation."""

import numpy as np
from typing import List, Tuple, Any, Dict
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SuperGlobalReranker:
    """
    SuperGlobal reranking using global feature aggregation and query expansion.
    Implements lightweight reranking without requiring additional models.
    """

    def __init__(self, model_service=None):
        """
        Initialize SuperGlobal reranker.

        Args:
            model_service: Service for embedding computation
        """
        self.model_service = model_service

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_m: int = 500,
        qexp_k: int = 10,
        img_knn: int = 10,
        gem_p: float = 3.0
    ) -> List[Tuple[Any, float]]:
        """
        Perform SuperGlobal reranking.

        Args:
            query: Original text query
            candidates: List of candidate items 
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_m: Number of top candidates to consider
            qexp_k: Number of top candidates for query expansion
            img_knn: Number of nearest neighbors for image aggregation
            gem_p: Generalized mean pooling parameter

        Returns:
            List of (candidate, score) tuples sorted by score descending
        """
        try:
            if len(candidates) == 0:
                return []

            # Limit to top_m candidates
            n_candidates = min(len(candidates), top_m)
            candidates = candidates[:n_candidates]
            candidate_embeddings = candidate_embeddings[:n_candidates]

            # Convert to numpy arrays
            query_emb = np.array(query_embedding).reshape(1, -1)
            cand_embs = np.array(candidate_embeddings)

            # 1. Compute initial cosine similarities
            base_scores = cosine_similarity(query_emb, cand_embs)[0]

            # 2. Query expansion using top-K candidates
            top_k_indices = np.argsort(base_scores)[-qexp_k:]
            top_k_embeddings = cand_embs[top_k_indices]

            # Aggregate top-K embeddings using max pooling
            expanded_query = np.max(top_k_embeddings, axis=0, keepdims=True)
            expanded_scores = cosine_similarity(expanded_query, cand_embs)[0]

            # 3. Image feature aggregation using GeM pooling for top candidates
            img_knn_indices = np.argsort(base_scores)[-img_knn:]
            img_knn_embeddings = cand_embs[img_knn_indices]

            # GeM pooling: (mean of p-th powers)^(1/p)
            gem_features = np.power(
                np.mean(np.power(np.abs(img_knn_embeddings), gem_p), axis=0),
                1.0 / gem_p
            ).reshape(1, -1)

            gem_scores = cosine_similarity(gem_features, cand_embs)[0]

            # 4. Combine scores
            # s_sg = 0.5 * (cos(q, v'_i) + cos(q', v_i))
            final_scores = 0.5 * (expanded_scores + gem_scores)

            # 5. Normalize to [0, 1]
            min_score, max_score = final_scores.min(), final_scores.max()
            if max_score > min_score:
                final_scores = (final_scores - min_score) / \
                    (max_score - min_score)
            else:
                final_scores = np.ones_like(final_scores) * 0.5

            # 6. Create result tuples and sort
            results = list(zip(candidates, final_scores.tolist()))
            results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"SuperGlobal reranked {n_candidates} candidates, "
                         f"score range: [{final_scores.min():.3f}, {final_scores.max():.3f}]")

            return results

        except Exception as e:
            logger.error(f"SuperGlobal reranking failed: {e}")
            # Fallback: return candidates with original scores
            fallback_scores = cosine_similarity(query_emb, cand_embs)[0]
            return list(zip(candidates, fallback_scores.tolist()))

    def _gem_pooling(self, features: np.ndarray, p: float = 3.0) -> np.ndarray:
        """
        Generalized Mean Pooling implementation.

        Args:
            features: Feature matrix of shape (N, D)
            p: Pooling parameter

        Returns:
            Pooled features of shape (1, D)
        """
        return np.power(
            np.mean(np.power(np.abs(features), p), axis=0),
            1.0 / p
        ).reshape(1, -1)
