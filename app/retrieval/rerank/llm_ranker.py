"""LLM-based reranking with direct image-query relevance scoring."""

import json
import hashlib
import os
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class LLMRanker:
    """
    LLM-based reranker using multimodal models for direct relevance scoring.
    Uses Vietnamese-capable MLLM to score image-query relevance.
    """

    def __init__(
        self,
        model_service=None,
        cache_dir: str = "./cache/llm_scores",
        model_name: str = "5CD-AI/Vintern-1B-v2",
        timeout: int = 15
    ):
        """
        Initialize LLM ranker.

        Args:
            model_service: Service for model access
            cache_dir: Directory for score cache
            model_name: Name of LLM model
            timeout: Default timeout for LLM calls
        """
        self.model_service = model_service
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.timeout = timeout

        # Mock model for now
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(f"LLMRanker initialized with cache_dir={cache_dir}")

    async def rerank_with_llm(
        self,
        query: str,
        candidates: List[Any],
        top_t: int = 5,
        timeout: Optional[int] = None
    ) -> List[Tuple[Any, float]]:
        """
        Rerank candidates using LLM relevance scoring.

        Args:
            query: Text query
            candidates: List of candidate items
            top_t: Number of top candidates to score with LLM
            timeout: Total timeout for LLM scoring

        Returns:
            List of (candidate, llm_score) tuples
        """
        try:
            if len(candidates) == 0:
                return []

            timeout = timeout or self.timeout

            # Limit to top_t candidates for LLM scoring
            n_candidates = min(len(candidates), top_t)
            selected_candidates = candidates[:n_candidates]

            logger.debug(f"LLM scoring top {n_candidates} candidates")

            # Score candidates with LLM
            start_time = time.time()
            scores = await self._score_candidates_batch(
                query, selected_candidates, timeout=timeout
            )

            elapsed = time.time() - start_time
            logger.debug(f"LLM scoring took {elapsed:.2f}s")

            # Create result tuples
            results = []
            for i, candidate in enumerate(selected_candidates):
                score = scores[i] if i < len(scores) else 0.0
                results.append((candidate, score))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"LLM reranking completed, "
                         f"score range: [{min(scores):.3f}, {max(scores):.3f}]")

            return results

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            # Fallback: return candidates with zero scores
            return [(candidate, 0.0) for candidate in candidates[:top_t]]

    async def _score_candidates_batch(
        self,
        query: str,
        candidates: List[Any],
        timeout: int = 15
    ) -> List[float]:
        """
        Score a batch of candidates using LLM.

        Args:
            query: Text query
            candidates: List of candidates to score
            timeout: Total timeout for batch

        Returns:
            List of relevance scores [0, 1]
        """
        tasks = []
        single_timeout = timeout / len(candidates) if candidates else timeout

        for candidate in candidates:
            task = self._get_or_compute_llm_score(
                query, candidate, single_timeout)
            tasks.append(task)

        try:
            scores = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )

            # Handle exceptions in results
            processed_scores = []
            for i, result in enumerate(scores):
                if isinstance(result, Exception):
                    logger.warning(
                        f"LLM scoring failed for candidate {i}: {result}")
                    processed_scores.append(0.0)
                else:
                    processed_scores.append(result)

            return processed_scores

        except asyncio.TimeoutError:
            logger.error(f"LLM scoring batch timeout after {timeout}s")
            return [0.0] * len(candidates)

    async def _get_or_compute_llm_score(
        self,
        query: str,
        candidate: Any,
        timeout: float = 10.0
    ) -> float:
        """
        Get cached LLM score or compute new one.

        Args:
            query: Text query
            candidate: Candidate item
            timeout: Timeout for single scoring

        Returns:
            Relevance score [0, 1]
        """
        try:
            # Create cache key from query and candidate
            cache_key = self._create_cache_key(query, candidate)
            cache_path = self.cache_dir / f"{cache_key}.json"

            # Check cache first
            if cache_path.exists():
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return float(data.get('relevance', 0.0))
                except Exception as e:
                    logger.warning(
                        f"Failed to read LLM cache {cache_path}: {e}")

            # Compute new score
            score = await self._compute_llm_score(query, candidate, timeout)

            # Cache the result
            try:
                cache_data = {
                    'query': query,
                    'image_id': self._get_image_id(candidate),
                    'relevance': float(score),
                    'model': self.model_name,
                    'timestamp': time.time()
                }
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write LLM cache {cache_path}: {e}")

            return score

        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return 0.0

    async def _compute_llm_score(
        self,
        query: str,
        candidate: Any,
        timeout: float = 10.0
    ) -> float:
        """
        Compute LLM relevance score for query-candidate pair.

        Args:
            query: Text query
            candidate: Candidate item
            timeout: Scoring timeout

        Returns:
            Relevance score [0, 1]
        """
        try:
            # Create Vietnamese prompt for relevance scoring
            prompt = self._create_relevance_prompt(query)

            # Mock LLM inference - replace with actual model call
            # In real implementation, this would:
            # 1. Load the image from candidate
            # 2. Pass image + prompt to MLLM
            # 3. Parse JSON response with relevance score

            # For now, use simple heuristic based on query and candidate
            score = await self._mock_llm_inference(query, candidate, prompt, timeout)

            # Ensure score is in [0, 1] range
            score = max(0.0, min(1.0, float(score)))

            logger.debug(f"LLM score for query '{query[:50]}...': {score:.3f}")
            return score

        except Exception as e:
            logger.error(f"LLM score computation error: {e}")
            return 0.0

    def _create_relevance_prompt(self, query: str) -> str:
        """
        Create Vietnamese prompt for relevance scoring.

        Args:
            query: User query

        Returns:
            Formatted prompt string
        """
        prompt = f"""
Hãy đánh giá mức độ liên quan giữa hình ảnh này và câu truy vấn của người dùng.

Truy vấn: "{query}"

Hãy phân tích hình ảnh và trả lời bằng định dạng JSON:
{{"relevance": float}}

Trong đó relevance là điểm từ 0.0 đến 1.0:
- 0.0: Hoàn toàn không liên quan
- 0.3: Có ít liên quan
- 0.5: Liên quan vừa phải  
- 0.7: Khá liên quan
- 1.0: Rất liên quan, khớp hoàn toàn

Chỉ trả lời JSON, không thêm giải thích.
""".strip()

        return prompt

    async def _mock_llm_inference(
        self,
        query: str,
        candidate: Any,
        prompt: str,
        timeout: float
    ) -> float:
        """
        Mock LLM inference for development/testing.

        Args:
            query: Text query
            candidate: Candidate item
            prompt: LLM prompt
            timeout: Inference timeout

        Returns:
            Mock relevance score
        """
        try:
            # Simulate processing time
            await asyncio.sleep(0.2)

            # Simple scoring heuristic based on string similarity
            query_lower = query.lower()
            candidate_str = str(candidate).lower()

            # Mock scoring logic
            score = 0.5  # Base score

            # Simple keyword matching
            query_words = set(query_lower.split())
            candidate_words = set(candidate_str.split())

            if query_words & candidate_words:
                score += 0.3  # Bonus for word overlap

            # Mock random variation
            import random
            random.seed(abs(hash(query + str(candidate))))
            score += random.uniform(-0.2, 0.2)

            score = max(0.0, min(1.0, score))

            return score

        except Exception as e:
            logger.error(f"Mock LLM inference error: {e}")
            return 0.0

    def _create_cache_key(self, query: str, candidate: Any) -> str:
        """
        Create cache key from query and candidate.

        Args:
            query: Text query
            candidate: Candidate item

        Returns:
            Cache key string
        """
        try:
            image_id = self._get_image_id(candidate)
            query_hash = hashlib.sha1(query.encode('utf-8')).hexdigest()[:16]
            return f"{image_id}_{query_hash}"
        except Exception:
            # Fallback
            combined = f"{query}_{str(candidate)}"
            return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def _get_image_id(self, candidate: Any) -> Optional[str]:
        """
        Extract image identifier from candidate.

        Args:
            candidate: Candidate object

        Returns:
            Image ID string or None
        """
        try:
            # Try different attribute names
            for attr in ['image_id', 'id', '_id', 'keyframe_id']:
                if hasattr(candidate, attr):
                    value = getattr(candidate, attr)
                    if value:
                        return str(value)

            # Try path-based ID
            for attr in ['image_path', 'path', 'file_path']:
                if hasattr(candidate, attr):
                    path = getattr(candidate, attr)
                    if path:
                        return Path(path).stem

            # Fallback
            return hashlib.md5(str(candidate).encode()).hexdigest()[:16]

        except Exception:
            return "unknown"
