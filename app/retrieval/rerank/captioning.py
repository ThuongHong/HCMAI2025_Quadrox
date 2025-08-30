"""Caption-based reranking service with synthetic captions from metadata."""

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


class CaptionRanker:
    """
    Caption-based reranking using synthetic captions generated from metadata.
    Uses lightweight metadata synthesis instead of heavy multimodal models.
    """

    def __init__(
        self,
        model_service=None,
        cache_dir: str = "./cache/captions",
        model_name: str = "synthetic",
        max_workers: int = 2
    ):
        """
        Initialize caption ranker.

        Args:
            model_service: Service for text embedding computation
            cache_dir: Directory for caption cache
            model_name: Caption generation model name ("synthetic" for metadata-based)
            max_workers: Max worker threads for parallel processing
        """
        self.model_service = model_service
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Cache for captions with metadata hash validation
        self._cache_file = self.cache_dir / "captions.json"
        self._cache = self._load_cache()

        logger.info(
            f"CaptionRanker initialized with model={model_name}, cache_dir={cache_dir}")

    def _load_cache(self) -> Dict[str, Any]:
        """Load caption cache from file."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Failed to load caption cache: {e}")
            return {}

    def _save_cache(self):
        """Save caption cache to file."""
        try:
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save caption cache: {e}")

    async def rerank_with_captions(
        self,
        query: str,
        candidates: List[Any],
        top_t: int = 20,
        cache_enabled: bool = True,
        fallback_enabled: bool = True
    ) -> List[Tuple[Any, float]]:
        """
        Rerank candidates using caption-query similarity.

        Args:
            query: Text query
            candidates: List of candidate items
            top_t: Number of top candidates to caption and rerank
            cache_enabled: Whether to use cache for captions
            fallback_enabled: Whether to use fallback on errors

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
                f"Generating synthetic captions for top {n_candidates} candidates")

            # Generate captions for selected candidates
            start_time = time.time()
            captions = await self._generate_captions_batch(
                selected_candidates, cache_enabled=cache_enabled, fallback_enabled=fallback_enabled
            )

            elapsed = time.time() - start_time
            logger.debug(f"Caption generation took {elapsed:.2f}s")

            # Compute caption-query similarities
            scores = await self._compute_caption_similarities(
                query, captions, fallback_enabled=fallback_enabled
            )

            # Create result tuples
            results = []
            for i, candidate in enumerate(selected_candidates):
                score = scores[i] if i < len(scores) else 0.0
                results.append((candidate, score))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            if scores:
                logger.debug(f"Caption reranking completed, "
                             f"score range: [{min(scores):.3f}, {max(scores):.3f}]")

            return results

        except Exception as e:
            logger.error(f"Caption reranking failed: {e}")
            if fallback_enabled:
                # Fallback: return candidates with zero scores
                return [(candidate, 0.0) for candidate in candidates[:top_t]]
            else:
                logger.warning("Fallback disabled, returning empty results")
                return []
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
        cache_enabled: bool = True,
        fallback_enabled: bool = True
    ) -> List[str]:
        """
        Generate synthetic captions for a batch of candidates.

        Args:
            candidates: List of candidates to caption
            cache_enabled: Whether to use caching
            fallback_enabled: Whether to use fallback on errors

        Returns:
            List of caption strings
        """
        tasks = []
        for candidate in candidates:
            task = self._get_or_generate_caption(
                candidate, cache_enabled=cache_enabled, fallback_enabled=fallback_enabled
            )
            tasks.append(task)

        try:
            captions = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in results
            processed_captions = []
            for i, result in enumerate(captions):
                if isinstance(result, Exception):
                    if fallback_enabled:
                        logger.warning(
                            f"Caption generation failed for candidate {i}: {result}")
                        processed_captions.append(
                            "objects: scene | scene: generic frame")
                    else:
                        logger.error(
                            f"Caption generation failed for candidate {i}: {result} (no-fallback mode)")
                        processed_captions.append("")
                else:
                    processed_captions.append(result)

            return processed_captions

        except Exception as e:
            logger.error(f"Caption generation batch failed: {e}")
            if fallback_enabled:
                return ["objects: scene | scene: generic frame"] * len(candidates)
            else:
                logger.warning("Fallback disabled, returning empty captions")
                return [""] * len(candidates)

    async def _get_or_generate_caption(
        self,
        candidate: Any,
        cache_enabled: bool = True,
        fallback_enabled: bool = True
    ) -> str:
        """
        Get cached caption or generate new synthetic one.

        Args:
            candidate: Candidate item with metadata
            cache_enabled: Whether to use caching
            fallback_enabled: Whether to use fallback on errors

        Returns:
            Caption string (fallback if failed and enabled)
        """
        try:
            # Get image identifier
            image_id = self._get_image_id(candidate)
            if not image_id:
                if fallback_enabled:
                    return "objects: scene | scene: generic frame"
                else:
                    return ""

            # Extract metadata for hashing
            metadata = self._extract_metadata(candidate)
            meta_hash = self._compute_metadata_hash(metadata)

            # Check cache first (if enabled)
            if cache_enabled and image_id in self._cache:
                cached_entry = self._cache[image_id]
                if cached_entry.get('meta_hash') == meta_hash:
                    return cached_entry.get('caption', '')

            # Generate new synthetic caption
            caption = await self._generate_synthetic_caption(metadata, fallback_enabled)

            # Cache the result (if enabled)
            if cache_enabled and caption:
                self._cache[image_id] = {
                    'caption': caption,
                    'meta_hash': meta_hash,
                    'model': self.model_name,
                    'timestamp': time.time()
                }
                # Save cache periodically
                if len(self._cache) % 10 == 0:
                    self._save_cache()

            return caption

        except Exception as e:
            if fallback_enabled:
                logger.warning(f"Caption generation failed: {e}")
                return "objects: scene | scene: generic frame"
            else:
                logger.error(
                    f"Caption generation failed: {e} (no-fallback mode)")
                return ""

    def _extract_metadata(self, candidate: Any) -> Dict[str, Any]:
        """Extract relevant metadata from candidate for caption generation."""
        metadata = {}

        # Try to extract common metadata fields
        fields_to_extract = [
            'objects', 'title', 'description', 'keywords', 'author',
            'tags', 'categories', 'scene_type', 'activity'
        ]

        for field in fields_to_extract:
            value = None
            # Try different attribute access patterns
            if hasattr(candidate, field):
                value = getattr(candidate, field)
            elif hasattr(candidate, '_asdict') and field in candidate._asdict():
                value = candidate._asdict()[field]
            elif isinstance(candidate, dict) and field in candidate:
                value = candidate[field]

            if value is not None:
                metadata[field] = value

        return metadata

    def _compute_metadata_hash(self, metadata: Dict[str, Any]) -> str:
        """Compute hash of metadata for cache validation."""
        # Create deterministic string representation of metadata
        sorted_items = sorted(metadata.items())
        metadata_str = json.dumps(
            sorted_items, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(metadata_str.encode()).hexdigest()[:16]

    async def _generate_synthetic_caption(
        self,
        metadata: Dict[str, Any],
        fallback_enabled: bool = True
    ) -> str:
        """
        Generate synthetic caption from metadata.

        Args:
            metadata: Extracted metadata dictionary
            fallback_enabled: Whether to use fallback on errors

        Returns:
            Synthetic caption string
        """
        try:
            caption_parts = []

            # Objects section (limit to 3 most relevant)
            objects = metadata.get('objects', [])
            if isinstance(objects, str):
                objects = [obj.strip()
                           for obj in objects.split(',') if obj.strip()]
            elif not isinstance(objects, list):
                objects = []

            if objects:
                objects_limited = objects[:3]  # Limit to first 3 objects
                caption_parts.append(f"objects: {', '.join(objects_limited)}")

            # Title section (truncated)
            title = metadata.get('title', '')
            if isinstance(title, str) and title.strip():
                title_words = title.strip().split()[:8]  # Limit to 8 words
                caption_parts.append(f"title: {' '.join(title_words)}")

            # Keywords section (limit to 3)
            keywords = metadata.get('keywords', [])
            if isinstance(keywords, str):
                keywords = [kw.strip()
                            for kw in keywords.split(',') if kw.strip()]
            elif not isinstance(keywords, list):
                keywords = []

            if keywords:
                keywords_limited = keywords[:3]  # Limit to first 3 keywords
                caption_parts.append(
                    f"keywords: {', '.join(keywords_limited)}")

            # Author section
            author = metadata.get('author', '')
            if isinstance(author, str) and author.strip():
                author_words = author.strip().split()[:3]  # Limit to 3 words
                caption_parts.append(f"author: {' '.join(author_words)}")

            # Scene section (generic description based on metadata)
            scene_type = "indoor" if any(obj in str(objects).lower() for obj in
                                         ['chair', 'table', 'bed', 'sofa', 'tv', 'computer']) else "outdoor"
            caption_parts.append(f"scene: {scene_type}")

            # Combine parts with separator
            if caption_parts:
                caption = " | ".join(caption_parts)
                # Ensure caption is not too long (limit to ~100 characters)
                if len(caption) > 100:
                    caption = caption[:97] + "..."
                return caption
            else:
                # Minimal fallback
                return "objects: scene | scene: generic frame"

        except Exception as e:
            logger.error(f"Synthetic caption generation failed: {e}")
            if fallback_enabled:
                return "objects: scene | scene: generic frame"
            else:
                return ""

    async def _compute_caption_similarities(
        self,
        query: str,
        captions: List[str],
        fallback_enabled: bool = True
    ) -> List[float]:
        """
        Compute similarities between query and captions.

        Args:
            query: Text query
            captions: List of caption strings
            fallback_enabled: Whether to use fallback on errors

        Returns:
            List of similarity scores [0, 1]
        """
        try:
            if not self.model_service:
                if fallback_enabled:
                    logger.warning(
                        "No model service available for caption similarity")
                    return [0.0] * len(captions)
                else:
                    logger.error(
                        "No model service available for caption similarity (no-fallback mode)")
                    return [0.0] * len(captions)

            # Get query embedding
            try:
                query_embedding = self.model_service.embedding(query)
                if hasattr(query_embedding, 'tolist'):
                    query_embedding = query_embedding.tolist()
                query_emb = np.array(query_embedding).reshape(1, -1)
            except Exception as e:
                if fallback_enabled:
                    logger.warning(f"Failed to compute query embedding: {e}")
                    return [0.0] * len(captions)
                else:
                    logger.error(
                        f"Failed to compute query embedding: {e} (no-fallback mode)")
                    return [0.0] * len(captions)

            # Get caption embeddings
            caption_embeddings = []
            for caption in captions:
                if caption.strip():
                    try:
                        cap_emb = self.model_service.embedding(caption)
                        if hasattr(cap_emb, 'tolist'):
                            cap_emb = cap_emb.tolist()
                        caption_embeddings.append(cap_emb)
                    except Exception as e:
                        if fallback_enabled:
                            logger.debug(
                                f"Failed to embed caption '{caption[:30]}...': {e}")
                            # Zero embedding for failed captions
                            caption_embeddings.append(
                                [0.0] * len(query_embedding))
                        else:
                            logger.warning(
                                f"Failed to embed caption '{caption[:30]}...': {e} (no-fallback mode)")
                            caption_embeddings.append(
                                [0.0] * len(query_embedding))
                else:
                    # Zero embedding for empty captions
                    caption_embeddings.append([0.0] * len(query_embedding))

            if not caption_embeddings:
                return [0.0] * len(captions)

            cap_embs = np.array(caption_embeddings)

            # Compute cosine similarities
            similarities = cosine_similarity(query_emb, cap_embs)[0]

            # Normalize to [0, 1] range: (cosine + 1) / 2
            similarities = np.clip((similarities + 1) / 2, 0, 1)

            return similarities.tolist()

        except Exception as e:
            if fallback_enabled:
                logger.error(f"Caption similarity computation failed: {e}")
                return [0.0] * len(captions)
            else:
                logger.error(
                    f"Caption similarity computation failed: {e} (no-fallback mode)")
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

            # Try video/frame based ID
            if hasattr(candidate, 'group_num') and hasattr(candidate, 'video_num') and hasattr(candidate, 'keyframe_num'):
                return f"L{candidate.group_num:02d}_V{candidate.video_num:03d}_{candidate.keyframe_num:03d}"

            # Fallback: use hash of string representation
            return hashlib.md5(str(candidate).encode()).hexdigest()[:16]

        except Exception as e:
            logger.warning(f"Failed to extract image ID: {e}")
            return None

    def _try_llm_caption_backend(self, image_path: str) -> Optional[str]:
        """
        Hook for real LLM caption backend (not implemented).

        Args:
            image_path: Path to image file

        Returns:
            None (not implemented)
        """
        # Future implementation could load and use actual caption model
        if self.model_name != "synthetic":
            logger.info(
                f"Real caption model {self.model_name} requested but not implemented, using synthetic")
        return None

    def __del__(self):
        """Cleanup: save cache on destruction."""
        try:
            if hasattr(self, '_cache') and self._cache:
                self._save_cache()
        except:
            pass
