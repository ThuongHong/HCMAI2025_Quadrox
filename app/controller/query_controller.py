from schema.request import MetadataFilter, ObjectFilter
# Agent
from agent.agent import VisualEventExtractor
from llama_index.core.llms import LLM

from schema.response import KeyframeServiceReponse
from service import ModelService, KeyframeQueryService
from pathlib import Path
import json
from typing import Optional, Dict, Any
# Import rerank components
from retrieval.rerank import RerankPipeline, RerankOptions
from core.settings import RerankSettings

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

class QueryController:

    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService,
        llm: LLM,
        rerank_config: Optional[RerankSettings] = None,
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service
        self.llm = llm
        self.visual_extractor = VisualEventExtractor(
            llm) if llm is not None else None

        # Initialize rerank pipeline
        self.rerank_config = rerank_config or RerankSettings()
        self.rerank_pipeline = RerankPipeline(
            model_service=model_service,
            cache_base_dir="./cache",
        )

    def convert_model_to_path(
        self,
        model: KeyframeServiceReponse
    ) -> tuple[str, float]:
        return os.path.join(self.data_folder, f"L{model.group_num:02d}/L{model.group_num:02d}_V{model.video_num:03d}/{model.keyframe_num:03d}.jpg"), model.confidence_score

    def convert_model_to_display(
        self,
        keyframe_data: tuple  # (keyframe_model, confidence_score)
    ):
        """Convert keyframe model to SingleKeyframeDisplay with full metadata"""
        keyframe, score = keyframe_data

        # Build the path
        path = os.path.join(
            self.data_folder, f"L{keyframe.group_num:02d}/L{keyframe.group_num:02d}_V{keyframe.video_num:03d}/{keyframe.keyframe_num:03d}.jpg")

        return {
            'path': path,
            'score': score,
            'video_id': keyframe.video_num,
            'group_id': keyframe.group_num,
            'author': keyframe.author,
            'channel_id': keyframe.channel_id,
            'title': keyframe.title,
            'description': keyframe.description,
            'keywords': keyframe.keywords,
            'length': keyframe.length,
            'publish_date': keyframe.publish_date,
            'thumbnail_url': keyframe.thumbnail_url,
            'watch_url': keyframe.watch_url,
            'objects': keyframe.objects,  # Include detected objects
        }

    def _extract_rerank_params(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract rerank parameters from request and convert to proper format."""
        rerank_params = {}

        # Map request parameter names to internal names
        param_mapping = {
            'rerank': 'enable',
            'rerank_mode': 'mode',
            'rr_superglobal': 'enable_superglobal',
            'sg_top_m': 'sg_top_m',
            'sg_qexp_k': 'sg_qexp_k',
            'sg_img_knn': 'sg_img_knn',
            # New SuperGlobal params (accept both new and legacy names)
            'sg_alpha': 'sg_alpha',
            'sg_beta': 'sg_beta',
            'sg_p_query': 'sg_p_query',
            'sg_gem_p': 'sg_gem_p',
            'w_sg': 'w_sg',
            'final_top_k': 'final_top_k',
            # Add new cache and fallback flags
            'rerank_cache_enabled': 'cache_enabled',
            'rerank_fallback_enabled': 'fallback_enabled'
        }

        for req_param, internal_param in param_mapping.items():
            if req_param in request_params and request_params[req_param] is not None:
                rerank_params[internal_param] = request_params[req_param]

        return rerank_params

    def _build_rerank_options(self, request_params: Dict[str, Any]) -> RerankOptions:
        """Build RerankOptions from request parameters and config."""
        # Extract rerank parameters from request
        rerank_params = self._extract_rerank_params(request_params)

        # Convert config to dict format
        config_defaults = {}
        if self.rerank_config:
            config_defaults = {
                'RERANK_ENABLE': self.rerank_config.RERANK_ENABLE,
                'RERANK_MODE': self.rerank_config.RERANK_MODE,
                'RERANK_ENABLE_SUPERGLOBAL': self.rerank_config.RERANK_ENABLE_SUPERGLOBAL,
                'RERANK_SG_TOP_M': self.rerank_config.RERANK_SG_TOP_M,
                'RERANK_SG_QEXP_K': self.rerank_config.RERANK_SG_QEXP_K,
                'RERANK_SG_IMG_KNN': self.rerank_config.RERANK_SG_IMG_KNN,
                'RERANK_SG_ALPHA': getattr(self.rerank_config, 'RERANK_SG_ALPHA', 0.5),
                'RERANK_SG_BETA': getattr(self.rerank_config, 'RERANK_SG_BETA', 1.8),
                'RERANK_SG_P_QUERY': getattr(self.rerank_config, 'RERANK_SG_P_QUERY', 100.0),
                'RERANK_SG_GEM_P': getattr(self.rerank_config, 'RERANK_SG_GEM_P', 3.0),
                'RERANK_SG_SCORE_WEIGHT': self.rerank_config.RERANK_SG_SCORE_WEIGHT,
                'RERANK_FINAL_TOP_K': self.rerank_config.RERANK_FINAL_TOP_K,
                'RERANK_CACHE_ENABLED': getattr(self.rerank_config, 'RERANK_CACHE_ENABLED', True),
                'RERANK_FALLBACK_ENABLED': getattr(self.rerank_config, 'RERANK_FALLBACK_ENABLED', True),
            }

        # Create options with precedence
        return RerankOptions.from_request_and_config(rerank_params, config_defaults)

    async def search_text(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """Search text with optional reranking."""
        refined_query, objects = await self._refine_query(query)

        # Build rerank options
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)

        # Get base search results
        embedding = self.model_service.embedding(refined_query).tolist()[0]

        # Adjust initial search size for reranking
        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)

        result = await self.keyframe_service.search_by_text_with_full_metadata(embedding, initial_top_k, score_threshold)

        # Apply reranking if enabled
        if rerank_options and rerank_options.enable and result:
            try:
                # Extract candidates and their embeddings (from vector search)
                candidates = [item[0] for item in result]  # keyframe objects
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)

                # Run rerank pipeline
                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=refined_query,
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=embedding,
                    options=rerank_options
                )

                # Reconstruct result strictly following reranked order,
                # keeping original base similarity scores when available
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Reranking failed, using original results: {e}")

        return result[:top_k]

    async def search_text_with_exlude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[int],
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """Search text with group exclusion and optional reranking."""
        exclude_ids = [
            int(k) for k, v in self.id2index.items()
            if int(v.split('/')[0]) in list_group_exlude
        ]

        refined_query, objects = await self._refine_query(query)

        # Build rerank options
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)

        embedding = self.model_service.embedding(refined_query).tolist()[0]

        # Adjust initial search size for reranking
        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)

        result = await self.keyframe_service.search_by_text_exclude_ids_with_metadata(embedding, initial_top_k, score_threshold, exclude_ids)

        # Apply reranking if enabled
        if rerank_options and rerank_options.enable and result:
            try:
                candidates = [item[0] for item in result]
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)

                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=refined_query,
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=embedding,
                    options=rerank_options
                )

                # Reconstruct result strictly following reranked order
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Reranking failed: {e}")

        return result[:top_k]

    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int],
        list_of_include_videos: list[int],
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """Search with selected video groups and optional reranking."""
        exclude_ids = None
        if len(list_of_include_groups) > 0 and len(list_of_include_videos) == 0:
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[0]) not in list_of_include_groups
            ]

        elif len(list_of_include_groups) == 0 and len(list_of_include_videos) > 0:
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[1]) not in list_of_include_videos
            ]

        elif len(list_of_include_groups) == 0 and len(list_of_include_videos) == 0:
            exclude_ids = []
        else:
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if (
                    int(v.split('/')[0]) not in list_of_include_groups or
                    int(v.split('/')[1]) not in list_of_include_videos
                )
            ]

        refined_query, objects = await self._refine_query(query)

        # Build rerank options
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)

        embedding = self.model_service.embedding(refined_query).tolist()[0]

        # Adjust initial search size for reranking
        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)

        result = await self.keyframe_service.search_by_text_exclude_ids_with_metadata(embedding, initial_top_k, score_threshold, exclude_ids)

        # Apply reranking if enabled
        if rerank_options and rerank_options.enable and result:
            try:
                candidates = [item[0] for item in result]
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)

                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=refined_query,
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=embedding,
                    options=rerank_options
                )

                # Reconstruct result strictly following reranked order
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Reranking failed: {e}")

        return result[:top_k]

    async def search_text_with_metadata_filter(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        metadata_filter: Optional[MetadataFilter] = None,
        object_filter: Optional[ObjectFilter] = None,
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """
        Search for keyframes with metadata and object filtering plus optional reranking
        """
        refined_query, objects = await self._refine_query(query)

        # Build rerank options
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)

        embedding = self.model_service.embedding(refined_query).tolist()[0]

        # Convert MetadataFilter to dict format for the service
        metadata_dict = None
        if metadata_filter:
            metadata_dict = {}
            if metadata_filter.authors:
                metadata_dict["authors"] = metadata_filter.authors
            if metadata_filter.keywords:
                metadata_dict["keywords"] = metadata_filter.keywords
            if metadata_filter.keywords_mode:
                metadata_dict["keywords_mode"] = metadata_filter.keywords_mode
            if metadata_filter.min_length is not None:
                metadata_dict["min_length"] = metadata_filter.min_length
            if metadata_filter.max_length is not None:
                metadata_dict["max_length"] = metadata_filter.max_length
            if metadata_filter.title_contains:
                metadata_dict["title_contains"] = metadata_filter.title_contains
            if metadata_filter.title_terms:
                metadata_dict["title_terms"] = metadata_filter.title_terms
            if metadata_filter.title_mode:
                metadata_dict["title_mode"] = metadata_filter.title_mode
            if metadata_filter.description_contains:
                metadata_dict["description_contains"] = metadata_filter.description_contains
            if metadata_filter.description_terms:
                metadata_dict["description_terms"] = metadata_filter.description_terms
            if metadata_filter.description_mode:
                metadata_dict["description_mode"] = metadata_filter.description_mode
            if metadata_filter.date_from:
                metadata_dict["date_from"] = metadata_filter.date_from
            if metadata_filter.date_to:
                metadata_dict["date_to"] = metadata_filter.date_to

        # Convert ObjectFilter to dict format for the service
        object_dict = None
        if object_filter:
            # Normalize and validate object list
            normalized_objects = [obj.lower().strip()
                                  for obj in object_filter.objects if obj.strip()]
            if normalized_objects:
                object_dict = {
                    # Limit to 20 objects max
                    "objects": normalized_objects[:20],
                    "mode": object_filter.mode
                }

        # Adjust initial search size for reranking
        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)

        result = await self.keyframe_service.search_by_text_with_metadata_filter_full(
            embedding, initial_top_k, score_threshold, metadata_dict, object_dict
        )

        # Apply reranking if enabled
        if rerank_options and rerank_options.enable and result:
            try:
                candidates = [item[0] for item in result]
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)

                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=refined_query,
                    base_candidates=candidates,
                    query_embedding=embedding,
                    options=rerank_options
                )

                # Reconstruct result strictly following reranked order
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Reranking failed: {e}")

        return result[:top_k]

    async def search_image(
        self,
        image,
        top_k: int,
        score_threshold: float,
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """Search keyframes using image query with optional reranking"""
        # Build rerank options
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)

        embedding = self.model_service.image_embedding(image).tolist()[0]

        # Adjust initial search size for reranking
        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)

        result = await self.keyframe_service.search_by_text_with_full_metadata(embedding, initial_top_k, score_threshold)

        # Apply reranking if enabled (for image search, use a generic query)
        if rerank_options and rerank_options.enable and result:
            try:
                candidates = [item[0] for item in result]

                # For image search, use a generic query for reranking
                generic_query = "visual content search"

                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=generic_query,
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=embedding,
                    options=rerank_options
                )

                # Reconstruct result maintaining original scores
                result = [(cand, score) for (cand, score), new_cand in
                          zip(result, reranked_candidates) if cand == new_cand]

                original_ids = {id(cand) for cand, _ in result}
                for new_cand in reranked_candidates:
                    if id(new_cand) not in original_ids:
                        result.append((new_cand, 0.5))

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Reranking failed: {e}")

        return result[:top_k]

    async def _refine_query(self, query: str) -> tuple[str, list[str]]:
        """
        translate + enhance
        """
        return await self.keyframe_service._refine_query(query, self.llm, self.visual_extractor)
