"""SuperGlobal reranking implementation (vectorized, NumPy-only)."""

import time
import numpy as np
from typing import List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def l2norm(X: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: (m, D), B: (n, D), both assumed L2-normalized
    return A @ B.T


def gem_pool(X: np.ndarray, p: float = 3.0, eps: float = 1e-12) -> np.ndarray:
    # X: (N, D) -> (D,)
    if np.isinf(p):
        return X.max(axis=0)
    Xp = np.power(np.maximum(X, 0.0), p)
    g = np.power(np.mean(Xp, axis=0) + eps, 1.0 / p)
    return g


def superglobal_rerank(query_emb: np.ndarray, cand_embs: np.ndarray, options) -> np.ndarray:
    # --- params & guards ---
    assert query_emb.ndim == 1 and cand_embs.ndim == 2
    M, D = cand_embs.shape
    if M == 0:
        return np.empty((0,), dtype=np.float32)

    # Options with defaults
    K = int(getattr(options, "img_knn", getattr(options, "sg_img_knn", 10)))
    Kq = int(getattr(options, "qexp_k", getattr(options, "sg_qexp_k", 10)))
    beta = float(getattr(options, "beta", getattr(options, "sg_beta", 1.8)))
    alpha = float(getattr(options, "alpha", getattr(options, "sg_alpha", 0.5)))
    # Prefer p_query; fall back to various legacy names
    p_query = float(
        getattr(
            options,
            "p_query",
            getattr(options, "sg_p_query", getattr(options, "sg_gem_p", 100.0)),
        )
    )

    # --- L2-normalize ---
    q = l2norm(query_emb.reshape(1, -1))  # (1,D)
    V = l2norm(cand_embs)  # (M,D)

    # --- base scores q->V (for choosing top-Kq for QE) ---
    t0 = time.time()
    s_qV = cosine_matrix(q, V).ravel()  # (M,)

    # --- candidate-candidate similarities (DB-side refine) ---
    t_smm0 = time.time()
    Smm = cosine_matrix(V, V)  # (M,M)
    np.fill_diagonal(Smm, 0.0)
    t_smm1 = time.time()

    # Clamp K and Kq
    K = int(max(0, min(K, M)))
    Kq = int(max(1, min(Kq, M)))

    # If no DB refinement requested (K==0 or beta<=0), keep V
    if K == 0 or beta <= 0:
        gdr = V.copy()
        t_refine0 = t_refine1 = t_smm1
    else:
        # knn indices for each row
        knn_idx = np.argpartition(-Smm, K - 1, axis=1)[:, :K]  # (M,K)
        rows = np.arange(M)[:, None]
        W = np.clip(Smm[rows, knn_idx], 0.0, None) ** beta  # (M,K)

        t_refine0 = time.time()
        G_neighbors = V[knn_idx]  # (M,K,D)
        W_sum = (W.sum(axis=1, keepdims=True) + 1.0)  # +1 for self
        gdr = (V + (W[..., None] * G_neighbors).sum(axis=1)) / W_sum  # (M,D)
        gdr = l2norm(gdr)
        t_refine1 = time.time()

    # --- query-side GeM / ~max over refined features of top-Kq ---
    t_qe0 = time.time()
    top_idx = np.argpartition(-s_qV, Kq - 1)[:Kq]
    g_qe = gem_pool(gdr[top_idx], p=p_query).reshape(1, -1)
    g_qe = l2norm(g_qe)
    t_qe1 = time.time()

    # --- scores per paper ---
    t_score0 = time.time()
    S1 = cosine_matrix(q, gdr).ravel()
    S2 = cosine_matrix(g_qe, V).ravel()
    final = alpha * S1 + (1.0 - alpha) * S2
    t_score1 = time.time()

    logger.info(
        f"SuperGlobal params: M={M}, K={K}, Kq={Kq}, beta={beta:.3f}, alpha={alpha:.3f}, p_query={p_query:.1f}"
    )
    logger.info(
        f"SuperGlobal timings: Smm={(t_smm1 - t_smm0):.4f}s, refine={(t_refine1 - t_refine0):.4f}s, "
        f"QE={(t_qe1 - t_qe0):.4f}s, score={(t_score1 - t_score0):.4f}s"
    )
    return final.astype(np.float32)


class SuperGlobalReranker:
    """
    SuperGlobal reranking using global feature aggregation and query expansion.
    Implements lightweight reranking without requiring additional models.
    """

    def __init__(self, model_service: Optional[Union[Any, Any]] = None):  # Support both ModelService and SigLIPModelService
        self.model_service = model_service

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_m: int = 400,
        qexp_k: int = 10,
        img_knn: int = 10,
        beta: float = 1.8,
        alpha: float = 0.5,
        p_query: float | None = None,
        # Backward-compat alias; if provided and p_query is None, use it
        gem_p: float | None = None,
    ) -> List[Tuple[Any, float]]:
        """
        Perform SuperGlobal reranking.

        Args:
            query: Original text query (unused, kept for API compatibility)
            candidates: List of candidate items
            query_embedding: Query embedding vector (D,)
            candidate_embeddings: List/array of candidate embedding vectors (M,D)
            top_m: Number of top candidates to consider (M cap)
            qexp_k: Number of top candidates for query expansion (Kq)
            img_knn: Number of nearest neighbors for DB-side refine (K)
            beta: Weight exponent for neighbor aggregation
            alpha: Blend between <q,gdr_i> and <g_qe, v_i>
            p_query: GeM power for query-side pooling (large ~ max)
            gem_p: Legacy alias for p_query

        Returns:
            List of (candidate, score) tuples sorted by score descending
        """
        if len(candidates) == 0:
            return []

        # Limit to top_m candidates
        n_candidates = min(len(candidates), int(top_m))
        candidates = candidates[:n_candidates]
        candidate_embeddings = candidate_embeddings[:n_candidates]

        # Convert to numpy arrays
        q_emb = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        V = np.asarray(candidate_embeddings, dtype=np.float32)

        # Resolve p_query
        if p_query is None:
            p_query = gem_p if gem_p is not None else 100.0

        # Build a lightweight options object for the functional core
        class _Opt:
            pass

        opt = _Opt()
        opt.img_knn = int(img_knn)
        opt.qexp_k = int(qexp_k)
        opt.beta = float(beta)
        opt.alpha = float(alpha)
        opt.p_query = float(p_query)

        # Compute scores
        scores = superglobal_rerank(q_emb, V, opt)

        # Create result tuples and sort
        results = list(zip(candidates, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
