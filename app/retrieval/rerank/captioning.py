"""Caption-based reranking service with caching."""

import json
import hashlib
import os
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CaptionService:
    """
    Caption generation and reranking service with file-based caching.
    Uses Vietnamese-capable multimodal models for caption generation.
    """

    def __init__(
        self,
        model_service=None,
        cache_dir: str = "./cache/captions",
        model_name: str = "5CD-AI/Vintern-1B-v2",
        max_new_tokens: int = 64,
        temperature: float = 0.0
    ):
        """
        Initialize caption service.

        Args:
            model_service: Service for text embedding computation
            cache_dir: Directory for caption cache
            model_name: Name of caption generation model
            max_new_tokens: Maximum tokens for caption generation
            temperature: Generation temperature
        """
        self.model_service = model_service
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Mock model for now - in real implementation would load actual model
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info(f"CaptionService initialized with cache_dir={cache_dir}")

    async def rerank_with_captions(
        self,
        query: str,
        candidates: List[Any],
        top_t: int = 20,
        timeout: float = 30.0
    ) -> List[Tuple[Any, float]]:
        """
        Rerank candidates using caption-query similarity.

        Args:
            query: Text query
            candidates: List of candidate items
            top_t: Number of top candidates to caption and rerank
            timeout: Total timeout for caption generation

        Returns:
            List of (candidate, caption_score) tuples
        """
        try:
            if len(candidates) == 0:
                return []

            # Limit to top_t candidates
            n_candidates = min(len(candidates), top_t)
            selected_candidates = candidates[:n_candidates]

            logger.debug(
                f"Generating captions for top {n_candidates} candidates")

            # Generate captions for selected candidates
            start_time = time.time()
            captions = await self._generate_captions_batch(
                selected_candidates, timeout=timeout
            )

            elapsed = time.time() - start_time
            logger.debug(f"Caption generation took {elapsed:.2f}s")

            # Compute caption-query similarities
            scores = await self._compute_caption_similarities(query, captions)

            # Create result tuples
            results = []
            for i, candidate in enumerate(selected_candidates):
                score = scores[i] if i < len(scores) else 0.0
                results.append((candidate, score))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(f"Caption reranking completed, "
                         f"score range: [{min(scores):.3f}, {max(scores):.3f}]")

            return results

        except Exception as e:
            logger.error(f"Caption reranking failed: {e}")
            # Fallback: return candidates with zero scores
            return [(candidate, 0.0) for candidate in candidates[:top_t]]

    async def _generate_captions_batch(
        self,
        candidates: List[Any],
        timeout: float = 30.0
    ) -> List[str]:
        """
        Generate captions for a batch of candidates with caching.

        Args:
            candidates: List of candidates to caption
            timeout: Timeout for generation

        Returns:
            List of caption strings
        """
        tasks = []
        for candidate in candidates:
            task = self._get_or_generate_caption(
                candidate, timeout / len(candidates))
            tasks.append(task)

        try:
            captions = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )

            # Handle exceptions in results
            processed_captions = []
            for i, result in enumerate(captions):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Caption generation failed for candidate {i}: {result}")
                    processed_captions.append("")
                else:
                    processed_captions.append(result)

            return processed_captions

        except asyncio.TimeoutError:
            logger.error(f"Caption generation batch timeout after {timeout}s")
            return [""] * len(candidates)

    async def _get_or_generate_caption(
        self,
        candidate: Any,
        timeout: float = 10.0
    ) -> str:
        """
        Get cached caption or generate new one.

        Args:
            candidate: Candidate item (should have image_id or path attribute)
            timeout: Timeout for single caption generation

        Returns:
            Caption string (empty if failed)
        """
        try:
            # Get image identifier
            image_id = self._get_image_id(candidate)
            if not image_id:
                return ""

            # Check cache first
            cache_path = self.cache_dir / f"{image_id}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return data.get('caption', '')
                except Exception as e:
                    logger.warning(f"Failed to read cache {cache_path}: {e}")

            # Generate new caption
            caption = await self._generate_caption(candidate, timeout)

            # Cache the result
            if caption:
                try:
                    cache_data = {
                        'image_id': image_id,
                        'caption': caption,
                        'model': self.model_name,
                        'timestamp': time.time()
                    }
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to write cache {cache_path}: {e}")

            return caption

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return ""

    async def _generate_caption(self, candidate: Any, timeout: float = 10.0) -> str:
        """
        Generate caption for a single candidate.

        Args:
            candidate: Candidate with image data
            timeout: Generation timeout

        Returns:
            Generated caption string
        """
        try:
            # Mock implementation - replace with actual model inference
            # In real implementation, this would:
            # 1. Load the image from candidate.image_path or candidate data
            # 2. Run the multimodal model to generate Vietnamese caption
            # 3. Return the generated text

            # For now, return a mock caption based on candidate info
            image_id = self._get_image_id(candidate)
            mock_captions = [
                "Một người đàn ông đang đi bộ trên đường phố",
                "Cảnh giao thông trong thành phố với nhiều xe cộ",
                "Khu vườn xanh mát với cây cối và hoa lá",
                "Tòa nhà cao tầng hiện đại trong khu đô thị",
                "Người phụ nữ đang mua sắm tại chợ truyền thống"
            ]

            # Simple hash-based selection for consistent mock results
            caption_index = abs(hash(image_id)) % len(mock_captions)
            caption = mock_captions[caption_index]

            # Simulate processing delay
            await asyncio.sleep(0.1)

            logger.debug(f"Generated caption for {image_id}: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return ""

    async def _compute_caption_similarities(
        self,
        query: str,
        captions: List[str]
    ) -> List[float]:
        """
        Compute similarities between query and captions.

        Args:
            query: Text query
            captions: List of caption strings

        Returns:
            List of similarity scores [0, 1]
        """
        try:
            if not self.model_service:
                logger.warning(
                    "No model service available for caption similarity")
                return [0.0] * len(captions)

            # Get query embedding
            query_embedding = self.model_service.embedding(query)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            query_emb = np.array(query_embedding).reshape(1, -1)

            # Get caption embeddings
            caption_embeddings = []
            for caption in captions:
                if caption.strip():
                    cap_emb = self.model_service.embedding(caption)
                    if hasattr(cap_emb, 'tolist'):
                        cap_emb = cap_emb.tolist()
                    caption_embeddings.append(cap_emb)
                else:
                    # Zero embedding for empty captions
                    caption_embeddings.append([0.0] * len(query_embedding))

            if not caption_embeddings:
                return [0.0] * len(captions)

            cap_embs = np.array(caption_embeddings)

            # Compute cosine similarities
            similarities = cosine_similarity(query_emb, cap_embs)[0]

            # Normalize to [0, 1] range
            similarities = np.clip((similarities + 1) / 2, 0, 1)

            return similarities.tolist()

        except Exception as e:
            logger.error(f"Caption similarity computation failed: {e}")
            return [0.0] * len(captions)

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
                        # Use filename without extension as ID
                        return Path(path).stem

            # Fallback: use hash of string representation
            return hashlib.md5(str(candidate).encode()).hexdigest()[:16]

        except Exception as e:
            logger.warning(f"Failed to extract image ID: {e}")
            return None
