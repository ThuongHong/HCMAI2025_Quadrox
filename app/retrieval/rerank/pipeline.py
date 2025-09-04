"""
SuperGlobal-only reranking pipeline.
"""

import time
from typing import List, Tuple, Any, Dict, Optional
import logging

from .options import RerankOptions
from .superglobal import SuperGlobalReranker
from .ensemble import EnsembleScorer

logger = logging.getLogger(__name__)


class RerankPipeline:
    """Reranking pipeline using only SuperGlobal reranking."""

    def __init__(
        self,
        model_service=None,
        cache_base_dir: str = "./cache",
    ):
        self.model_service = model_service
        self.superglobal = SuperGlobalReranker(model_service)
        self.ensemble = EnsembleScorer()
        logger.info("RerankPipeline initialized (SuperGlobal only)")

    async def rerank_textual_kis(
        self,
        query: str,
        base_candidates: List[Any],
        base_embeddings: Optional[List] = None,
        query_embedding: Optional[List] = None,
        options: Optional[RerankOptions] = None
    ) -> List[Any]:
        """Perform SuperGlobal-only reranking for textual keyframe search."""
        try:
            start_time = time.time()

            if options is None:
                options = RerankOptions()

            if not options.enable:
                logger.debug("Reranking disabled, returning original candidates")
                return base_candidates[: options.final_top_k]

            logger.info(f"Starting rerank pipeline (SG-only): {options.to_dict()}")

            if options.mode == "auto":
                options = self._adjust_auto_mode(query, base_candidates, options)

            if not options.use_sg:
                logger.warning("SuperGlobal disabled; enabling by default")
                options.use_sg = True

            # Prepare embeddings
            query_emb = query_embedding
            if query_emb is None and self.model_service:
                query_emb = self.model_service.embedding(query)
                if hasattr(query_emb, "tolist"):
                    query_emb = query_emb.tolist()

            cand_embs = base_embeddings
            if cand_embs is None and self.model_service and options.use_sg:
                logger.debug("Generating embeddings for SuperGlobal reranking")
                cand_embs = []
                for _ in base_candidates[: options.sg_top_m]:
                    cand_embs.append([0.0] * 512)  # Mock embedding

            # Run SuperGlobal stage
            method_results: Dict[str, List[Tuple[Any, float]]] = {}
            stage_times: Dict[str, float] = {}

            if options.use_sg:
                stage_start = time.time()
                try:
                    sg_results = await self._run_superglobal_stage(
                        query, base_candidates, query_emb, cand_embs, options
                    )
                    method_results["superglobal"] = sg_results
                    stage_times["superglobal"] = time.time() - stage_start
                    logger.debug(
                        f"SuperGlobal completed in {stage_times['superglobal']:.2f}s"
                    )
                except Exception as e:
                    logger.error(f"SuperGlobal stage failed: {e}")
                    options.w_sg = 0.0

            # Ensemble and final ranking
            if method_results:
                weights = {"superglobal": options.w_sg}
                final_results = self.ensemble.combine_scores(
                    base_candidates,
                    method_results,
                    weights,
                    final_top_k=None,
                )

                if options.final_top_k is None or options.final_top_k <= 0:
                    final_candidates = [cand for cand, _ in final_results]
                else:
                    final_candidates = [
                        cand for cand, _ in final_results[: options.final_top_k]
                    ]
            else:
                logger.warning(
                    "Reranking produced no results, returning original candidates"
                )
                if options.final_top_k is None or options.final_top_k <= 0:
                    final_candidates = base_candidates
                else:
                    final_candidates = base_candidates[: options.final_top_k]

            total_time = time.time() - start_time
            self._log_pipeline_summary(
                query,
                len(base_candidates),
                len(final_candidates),
                method_results,
                stage_times,
                total_time,
                options,
            )

            return final_candidates

        except Exception as e:
            logger.error(f"Reranking pipeline failed: {e}")
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
        options: RerankOptions,
    ) -> List[Tuple[Any, float]]:
        """Run SuperGlobal reranking stage."""
        if not query_embedding or not candidate_embeddings:
            logger.warning("Missing embeddings for SuperGlobal, skipping")
            return []

        results = self.superglobal.rerank(
            query=query,
            candidates=candidates[: options.sg_top_m],
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings[: options.sg_top_m],
            top_m=options.sg_top_m,
            qexp_k=options.sg_qexp_k,
            img_knn=options.sg_img_knn,
            gem_p=options.sg_gem_p,
        )
        return results

    def _adjust_auto_mode(
        self,
        query: str,
        candidates: List[Any],
        options: RerankOptions,
    ) -> RerankOptions:
        """Auto mode: enable SuperGlobal only."""
        options.use_sg = True
        logger.debug("Auto mode: SuperGlobal only")
        return options

    def _log_pipeline_summary(
        self,
        query: str,
        n_input: int,
        n_output: int,
        method_results: Dict[str, List],
        stage_times: Dict[str, float],
        total_time: float,
        options: RerankOptions,
    ):
        methods_used = [name for name, results in method_results.items() if results]
        summary = {
            "query_preview": query[:50] + "..." if len(query) > 50 else query,
            "input_candidates": n_input,
            "output_candidates": n_output,
            "methods_used": methods_used,
            "stage_times": stage_times,
            "total_time": total_time,
            "options": {"mode": options.mode, "weights": f"sg={options.w_sg}"},
        }
        logger.info(f"Rerank pipeline summary: {summary}")

