"""Multi-stage reranking pipeline orchestrator."""

import time
import asyncio
from typing import List, Tuple, Any, Dict, Optional
import logging

from .options import RerankOptions
from .superglobal import SuperGlobalReranker
from .captioning import CaptionRanker
from .llm_ranker import LLMRanker
from .ensemble import EnsembleScorer

logger = logging.getLogger(__name__)


class RerankPipeline:
    """
    Multi-stage reranking pipeline that orchestrates SuperGlobal, Caption, and LLM reranking.
    Implements timeout management, graceful degradation, and result ensemble.
    """

    def __init__(
        self,
        model_service=None,
        cache_base_dir: str = "./cache",
        # Caption ranker configuration
        caption_model_name: str = "synthetic",
        caption_vintern_model_path: str = "./models/Vintern-1B-v3_5",
        caption_style: str = "dense",
        caption_max_new_tokens: int = 64,
        caption_allow_on_demand: bool = False,
        caption_alpha: float = 1.0,
        caption_beta: float = 0.25,
        caption_workers: int = 2,
        multilingual_model_path: str = "./models/clip-multilingual/clip-ViT-B-32-multilingual-v1"
    ):
        """
        Initialize reranking pipeline.

        Args:
            model_service: Service for embeddings and model access
            cache_base_dir: Base directory for caching
            caption_model_name: Caption model type (synthetic, vintern_cpu)
            caption_vintern_model_path: Path to Vintern model
            caption_style: Caption style for Vintern
            caption_max_new_tokens: Max tokens for caption generation
            caption_allow_on_demand: Allow on-demand caption generation
            caption_alpha: CLIP score weight
            caption_beta: Caption score weight
            caption_workers: Max workers for caption generation
            multilingual_model_path: Path to multilingual text embedding model
        """
        self.model_service = model_service

        # Initialize reranking components
        self.superglobal = SuperGlobalReranker(model_service)
        self.caption_ranker = CaptionRanker(
            model_service=model_service,
            cache_dir=f"{cache_base_dir}/captions",
            model_name=caption_model_name,
            max_workers=caption_workers,
            vintern_model_path=caption_vintern_model_path,
            caption_style=caption_style,
            max_new_tokens=caption_max_new_tokens,
            allow_on_demand=caption_allow_on_demand,
            alpha=caption_alpha,
            beta=caption_beta,
            multilingual_model_path=multilingual_model_path
        )
        self.llm_ranker = LLMRanker(
            model_service,
            cache_dir=f"{cache_base_dir}/llm_scores"
        )
        self.ensemble = EnsembleScorer()

        logger.info(
            f"RerankPipeline initialized with caption_model={caption_model_name}")

    async def rerank_textual_kis(
        self,
        query: str,
        base_candidates: List[Any],
        base_embeddings: Optional[List] = None,
        query_embedding: Optional[List] = None,
        options: Optional[RerankOptions] = None
    ) -> List[Any]:
        """
        Perform multi-stage reranking for textual keyframe search.

        Args:
            query: Original text query
            base_candidates: Initial candidate list from CLIP/metadata search
            base_embeddings: Embeddings for base candidates (optional)
            query_embedding: Query embedding (optional)
            options: Reranking configuration options

        Returns:
            Reranked list of candidates
        """
        try:
            start_time = time.time()

            # Use default options if none provided
            if options is None:
                options = RerankOptions()

            # Skip reranking if disabled
            if not options.enable:
                logger.debug(
                    "Reranking disabled, returning original candidates")
                return base_candidates[:options.final_top_k]

            logger.info(f"Starting rerank pipeline: {options.to_dict()}")

            # Handle auto mode
            if options.mode == "auto":
                options = self._adjust_auto_mode(
                    query, base_candidates, options)

            # Validate that at least one method is enabled
            if not (options.use_sg or options.use_caption or options.use_llm):
                logger.warning(
                    "No rerank methods enabled, falling back to SuperGlobal")
                options.use_sg = True

            # Prepare data
            query_emb = query_embedding
            if query_emb is None and self.model_service:
                query_emb = self.model_service.embedding(query)
                if hasattr(query_emb, 'tolist'):
                    query_emb = query_emb.tolist()

            cand_embs = base_embeddings
            if cand_embs is None and self.model_service and options.use_sg:
                # Generate embeddings for SuperGlobal if needed
                logger.debug("Generating embeddings for SuperGlobal reranking")
                cand_embs = []
                for candidate in base_candidates[:options.sg_top_m]:
                    # This would require image loading - simplified for now
                    # In real implementation: load image and get embedding
                    cand_embs.append([0.0] * 512)  # Mock embedding

            # Execute reranking stages
            method_results = {}
            stage_times = {}

            # Stage 1: SuperGlobal (if enabled)
            if options.use_sg:
                stage_start = time.time()
                try:
                    sg_results = await self._run_superglobal_stage(
                        query, base_candidates, query_emb, cand_embs, options
                    )
                    method_results['superglobal'] = sg_results
                    stage_times['superglobal'] = time.time() - stage_start
                    logger.debug(
                        f"SuperGlobal completed in {stage_times['superglobal']:.2f}s")
                except Exception as e:
                    logger.error(f"SuperGlobal stage failed: {e}")
                    options.w_sg = 0.0  # Disable weight

            # Stage 2: Caption reranking (if enabled)
            if options.use_caption:
                stage_start = time.time()
                try:
                    cap_results = await self._run_caption_stage(
                        query, base_candidates, options
                    )
                    method_results['caption'] = cap_results
                    stage_times['caption'] = time.time() - stage_start
                    logger.debug(
                        f"Caption reranking completed in {stage_times['caption']:.2f}s")
                except Exception as e:
                    logger.error(f"Caption stage failed: {e}")
                    options.w_cap = 0.0  # Disable weight

            # Stage 3: LLM reranking (if enabled)
            if options.use_llm:
                stage_start = time.time()
                try:
                    llm_results = await self._run_llm_stage(
                        query, base_candidates, options
                    )
                    method_results['llm'] = llm_results
                    stage_times['llm'] = time.time() - stage_start
                    logger.debug(
                        f"LLM reranking completed in {stage_times['llm']:.2f}s")
                except Exception as e:
                    logger.error(f"LLM stage failed: {e}")
                    options.w_llm = 0.0  # Disable weight

            # Ensemble and final ranking
            if method_results:
                weights = {
                    'superglobal': options.w_sg,
                    'caption': options.w_cap,
                    'llm': options.w_llm
                }

                final_results = self.ensemble.combine_scores(
                    base_candidates,
                    method_results,
                    weights,
                    final_top_k=None  # Let ensemble handle unlimited results
                )

                # Apply final_top_k limit: None or 0 = no limit, positive = limit
                if options.final_top_k is None or options.final_top_k <= 0:
                    final_candidates = [
                        candidate for candidate, score in final_results]
                    logger.debug(
                        f"No final_top_k limit applied, returning {len(final_candidates)} candidates")
                else:
                    final_candidates = [candidate for candidate,
                                        score in final_results[:options.final_top_k]]
                    logger.debug(
                        f"Applied final_top_k={options.final_top_k}, returning {len(final_candidates)} candidates")
            else:
                logger.warning(
                    "All reranking stages failed, returning original candidates")
                # Apply same logic for fallback
                if options.final_top_k is None or options.final_top_k <= 0:
                    final_candidates = base_candidates
                else:
                    final_candidates = base_candidates[:options.final_top_k]

            total_time = time.time() - start_time

            # Log pipeline summary
            self._log_pipeline_summary(
                query, len(base_candidates), len(final_candidates),
                method_results, stage_times, total_time, options
            )

            return final_candidates

        except Exception as e:
            logger.error(f"Reranking pipeline failed: {e}")
            # Fallback to original candidates with same final_top_k logic
            if options and (options.final_top_k is None or options.final_top_k <= 0):
                return base_candidates
            else:
                limit = options.final_top_k if options else 100
                return base_candidates[:limit]

    async def _run_superglobal_stage(
        self,
        query: str,
        candidates: List[Any],
        query_embedding: Optional[List],
        candidate_embeddings: Optional[List],
        options: RerankOptions
    ) -> List[Tuple[Any, float]]:
        """Run SuperGlobal reranking stage."""
        if not query_embedding or not candidate_embeddings:
            logger.warning("Missing embeddings for SuperGlobal, skipping")
            return []

        results = self.superglobal.rerank(
            query=query,
            candidates=candidates[:options.sg_top_m],
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings[:options.sg_top_m],
            top_m=options.sg_top_m,
            qexp_k=options.sg_qexp_k,
            img_knn=options.sg_img_knn,
            gem_p=options.sg_gem_p
        )

        return results

    async def _run_caption_stage(
        self,
        query: str,
        candidates: List[Any],
        options: RerankOptions
    ) -> List[Tuple[Any, float]]:
        """Run Caption reranking stage."""
        results = await self.caption_ranker.rerank_with_captions(
            query=query,
            candidates=candidates[:options.cap_top_t],
            top_t=options.cap_top_t,
            cache_enabled=options.cache_enabled,
            fallback_enabled=options.fallback_enabled
        )

        return results

    async def _run_llm_stage(
        self,
        query: str,
        candidates: List[Any],
        options: RerankOptions
    ) -> List[Tuple[Any, float]]:
        """Run LLM reranking stage."""
        results = await self.llm_ranker.rerank_with_llm(
            query=query,
            candidates=candidates[:options.llm_top_t],
            top_t=options.llm_top_t,
            timeout=options.llm_timeout,
            cache_enabled=options.cache_enabled,
            fallback_enabled=options.fallback_enabled
        )

        return results

    def _adjust_auto_mode(
        self,
        query: str,
        candidates: List[Any],
        options: RerankOptions
    ) -> RerankOptions:
        """
        Adjust reranking options for auto mode based on query and candidates.

        Args:
            query: Text query
            candidates: Base candidates
            options: Current options

        Returns:
            Adjusted options
        """
        # Auto mode heuristics
        query_len = len(query.split())

        # Always enable SuperGlobal (fast)
        options.use_sg = True

        # Enable Caption for longer/complex queries
        if query_len > 5 or any(keyword in query.lower() for keyword in ['người', 'xe', 'nhà', 'cây']):
            options.use_caption = True
            options.cap_top_t = min(15, len(candidates))

        # Enable LLM for very specific queries or when we have few top candidates
        if query_len > 8 or len(candidates) < 10:
            options.use_llm = True
            options.llm_top_t = min(3, len(candidates))

        logger.debug(
            f"Auto mode adjusted: sg={options.use_sg}, cap={options.use_caption}, llm={options.use_llm}")

        return options

    def _log_pipeline_summary(
        self,
        query: str,
        n_input: int,
        n_output: int,
        method_results: Dict[str, List],
        stage_times: Dict[str, float],
        total_time: float,
        options: RerankOptions
    ):
        """Log pipeline execution summary."""
        methods_used = [name for name,
                        results in method_results.items() if results]

        summary = {
            'query_preview': query[:50] + '...' if len(query) > 50 else query,
            'input_candidates': n_input,
            'output_candidates': n_output,
            'methods_used': methods_used,
            'stage_times': stage_times,
            'total_time': total_time,
            'options': {
                'mode': options.mode,
                'weights': f"sg={options.w_sg}, cap={options.w_cap}, llm={options.w_llm}"
            }
        }

        logger.info(f"Rerank pipeline summary: {summary}")

        # Detailed per-method statistics
        for method_name, results in method_results.items():
            if results:
                scores = [score for _, score in results]
                logger.debug(f"{method_name}: {len(scores)} scores, "
                             f"mean={sum(scores)/len(scores):.3f}, "
                             f"range=[{min(scores):.3f}, {max(scores):.3f}]")
