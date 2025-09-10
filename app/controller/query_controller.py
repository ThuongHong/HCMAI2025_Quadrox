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
from core.settings import RerankSettings, AppSettings

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from core.logger import SimpleLogger

logger = SimpleLogger(__name__)
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

    # --- Internal: make CLIP-friendly embedding text without altering outputs/logs ---
    def _normalize_for_embedding(self, text: str) -> str:
        """Insert a neutral descriptor before quoted phrases to help CLIP understand it's visible text.
        Keeps quoted content verbatim; affects only the string used for embedding.
        """
        if not isinstance(text, str) or not text:
            return text
        out = text
        # Add "phrase " before ASCII quotes if not already preceded by common descriptors
        out = __import__("re").sub(
            r"(?<!phrase )(?<!text )(?<!quote )(?<!slogan )(?<!caption )(?<!reading )(\"[^\"]+\")",
            r"phrase \g<0>",
            out,
        )
        # Add "phrase " before curly quotes if not already preceded
        out = __import__("re").sub(
            r"(?<!phrase )(?<!text )(?<!quote )(?<!slogan )(?<!caption )(?<!reading )([\u201C][^\u201D]+[\u201D])",
            r"phrase \g<0>",
            out,
        )
        return out

    def _embedding_for_query(self, text: str):
        return self.model_service.embedding(self._normalize_for_embedding(text)).tolist()[0]

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

    def _get_qexp_params(self, request_params: Dict[str, Any]) -> dict:
        """Return QExp runtime parameters with AppSettings defaults."""
        cfg = AppSettings()
        return {
            "enable": request_params.get("qexp_enable", cfg.QEXP_ENABLE),
            "top_variants": int(request_params.get("qexp_top_variants", cfg.QEXP_MAX_VARIANTS)),
            "fusion": request_params.get("qexp_fusion", cfg.QEXP_FUSION),
            "use_objects": request_params.get("qexp_use_objects", cfg.QEXP_OBJECT_FILTER_AUTO),
        }

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

    # Removed duplicate _get_qexp_params to avoid confusion

    def _stable_key(self, kf) -> str:
        try:
            db_id = getattr(kf, "db_id", None)
            if db_id is not None:
                return f"db:{db_id}"
            vec_id = getattr(kf, "vector_id", None)
            if vec_id is not None:
                return f"vec:{vec_id}"
            return f"{kf.group_num:02d}-{kf.video_num:03d}-{kf.keyframe_num:03d}"
        except Exception:
            return getattr(kf, "path", None) or getattr(kf, "rel_path", None) or str(id(kf))

    async def search_text(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """Search text with optional reranking. New QExp engine is gated by qexp_enable."""
        qexp = self._get_qexp_params(rerank_params or {})
        if not qexp["enable"]:
            # Legacy path (previous default)
            refined_query, _objects = await self.keyframe_service._refine_query(query, self.llm, self.visual_extractor)
            # Build rerank options
            rerank_options = None
            if rerank_params:
                rerank_options = self._build_rerank_options(rerank_params)
            embedding = self._embedding_for_query(refined_query)
            initial_top_k = top_k
            if rerank_options and rerank_options.enable:
                initial_top_k = max(top_k, rerank_options.sg_top_m)
            result = await self.keyframe_service.search_by_text_with_full_metadata(embedding, initial_top_k, score_threshold)
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
                    orig_map = {id(c): s for c, s in result}
                    result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Reranking failed, using original results: {e}")
            return result[:top_k]

        # New engine path
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)
        selected_query, obj_list, variants = await self.keyframe_service._refine_query_qexp(query, self.llm, self.visual_extractor)
        # Variant selection: keep only strong ones (score >= 7.5), capped by config
        queries = [selected_query]
        weights = [1.0]
        if qexp["enable"] and variants:
            vv = sorted(variants, key=lambda v: (v.get("score") or 0.0), reverse=True)
            vv = [v for v in vv if (v.get("score") or 0.0) >= 7.5][: qexp["top_variants"]]
            for v in vv:
                qv = v.get("query")
                if qv:
                    queries.append(qv)
                    weights.append(1.0)
        logger.info(f"QExp enabled: queries={len(queries)} (kept_variants={len(queries)-1}), fusion={qexp['fusion']}")
        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)
        per_query_k = initial_top_k if not qexp["enable"] else max(initial_top_k, min(200, int(initial_top_k * 1.5)))
        embeddings = [self._embedding_for_query(q) for q in queries]
        if qexp["fusion"] == "rrf":
            weights = [1.0] * len(queries)
        result, best_qmap = await self.keyframe_service.search_multi_and_fuse(
            embeddings=embeddings,
            weights=weights,
            top_k=per_query_k,
            score_threshold=score_threshold,
            fusion=qexp["fusion"],
        )
        # Soft object boost (configurable), skip if generic-only or explicit filter present
        cfg = AppSettings()
        GENERIC = {o.lower() for o in getattr(cfg, 'QEXP_OBJECT_GENERIC', {"person","people","man","woman","car","chair","table","phone","laptop","tv"})}
        valid_targets = [o for o in (obj_list or []) if o and o.lower() not in GENERIC]
        if qexp["use_objects"] and valid_targets:
            target = {o.lower() for o in valid_targets}
            before_n = len(result)
            boost_val = float(getattr(cfg, 'QEXP_OBJECT_BOOST', 0.08))
            boosted = []
            for kf, sc in result:
                kf_objs = set(map(str.lower, getattr(kf, "objects", []) or []))
                bonus = boost_val if (target & kf_objs) else 0.0
                boosted.append((kf, sc + bonus))
            result = sorted(boosted, key=lambda x: x[1], reverse=True)
            logger.debug(f"Object boost applied: targets={len(target)}, candidates={before_n}")
        if rerank_options and rerank_options.enable and result:
            try:
                candidates = [item[0] for item in result]
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)
                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=selected_query,
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=self._embedding_for_query(selected_query),
                    options=rerank_options
                )
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]
            except Exception as e:
                logger.error(f"Reranking failed, using fused results: {e}")
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
        qexp = self._get_qexp_params(rerank_params or {})
        if not qexp["enable"]:
            refined_query, _ = await self.keyframe_service._refine_query(query, self.llm, self.visual_extractor)
            rerank_options = None
            if rerank_params:
                rerank_options = self._build_rerank_options(rerank_params)
            embedding = self._embedding_for_query(refined_query)
            initial_top_k = top_k
            if rerank_options and rerank_options.enable:
                initial_top_k = max(top_k, rerank_options.sg_top_m)
            result = await self.keyframe_service.search_by_text_exclude_ids_with_metadata(embedding, initial_top_k, score_threshold, exclude_ids)
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
                    orig_map = {id(c): s for c, s in result}
                    result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Reranking failed: {e}")
            return result[:top_k]

        # New engine path
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)
        selected_query, obj_list, variants = await self.keyframe_service._refine_query_qexp(query, self.llm, self.visual_extractor)
        queries = [selected_query]
        weights = [1.0]
        if qexp["enable"] and variants:
            vv = sorted(variants, key=lambda v: (v.get("score") or 0.0), reverse=True)
            vv = [v for v in vv if (v.get("score") or 0.0) >= 7.5][: qexp["top_variants"]]
            for v in vv:
                qv = v.get("query")
                if qv:
                    queries.append(qv)
                    weights.append(1.0)
        logger.info(f"QExp enabled (exclude groups): queries={len(queries)}, fusion={qexp['fusion']}")

        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)
        per_query_k = initial_top_k if not qexp["enable"] else max(initial_top_k, min(200, int(initial_top_k * 1.5)))

        embeddings = [self._embedding_for_query(q) for q in queries]
        if qexp["fusion"] == "rrf":
            weights = [1.0] * len(queries)
        result, best_qmap = await self.keyframe_service.search_multi_and_fuse(
            embeddings=embeddings,
            weights=weights,
            top_k=per_query_k,
            score_threshold=score_threshold,
            fusion=qexp["fusion"],
            exclude_ids=exclude_ids,
        )

        cfg = AppSettings()
        GENERIC = {o.lower() for o in getattr(cfg, 'QEXP_OBJECT_GENERIC', {"person","people","man","woman","car","chair","table","phone","laptop","tv"})}
        valid_targets = [o for o in (obj_list or []) if o and o.lower() not in GENERIC]
        if qexp["use_objects"] and valid_targets:
            target = {o.lower() for o in valid_targets}
            boost_val = float(getattr(cfg, 'QEXP_OBJECT_BOOST', 0.08))
            boosted = []
            for kf, sc in result:
                kf_objs = set(map(str.lower, getattr(kf, "objects", []) or []))
                bonus = boost_val if (target & kf_objs) else 0.0
                boosted.append((kf, sc + bonus))
            result = sorted(boosted, key=lambda x: x[1], reverse=True)

        if rerank_options and rerank_options.enable and result:
            try:
                candidates = [item[0] for item in result]
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)
                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=selected_query,
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=self._embedding_for_query(selected_query),
                    options=rerank_options
                )
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]

            except Exception as e:
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

        # Build rerank + qexp
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)
        qexp = self._get_qexp_params(rerank_params or {})

        selected_query, obj_list, variants = await self.keyframe_service._refine_query_qexp(query, self.llm, self.visual_extractor)
        queries = [selected_query]
        weights = [1.0]
        if qexp["enable"] and variants:
            vv = sorted(variants, key=lambda v: (v.get("score") or 0.0), reverse=True)
            vv = [v for v in vv if (v.get("score") or 0.0) >= 7.5][: qexp["top_variants"]]
            for v in vv:
                qv = v.get("query")
                if qv:
                    queries.append(qv)
                    weights.append(1.0)
        logger.info(f"QExp enabled (selected groups/videos): queries={len(queries)}, fusion={qexp['fusion']}")

        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)
        per_query_k = initial_top_k if not qexp["enable"] else max(initial_top_k, min(200, int(initial_top_k * 1.5)))

        embeddings = [self._embedding_for_query(q) for q in queries]
        if qexp["fusion"] == "rrf":
            weights = [1.0] * len(queries)
        result, best_qmap = await self.keyframe_service.search_multi_and_fuse(
            embeddings=embeddings,
            weights=weights,
            top_k=per_query_k,
            score_threshold=score_threshold,
            fusion=qexp["fusion"],
            exclude_ids=exclude_ids,
        )

        cfg = AppSettings()
        GENERIC = {o.lower() for o in getattr(cfg, 'QEXP_OBJECT_GENERIC', {"person","people","man","woman","car","chair","table","phone","laptop","tv"})}
        valid_targets = [o for o in (obj_list or []) if o and o.lower() not in GENERIC]
        if qexp["use_objects"] and valid_targets:
            target = {o.lower() for o in valid_targets}
            boost_val = float(getattr(cfg, 'QEXP_OBJECT_BOOST', 0.08))
            boosted = []
            for kf, sc in result:
                kf_objs = set(map(str.lower, getattr(kf, "objects", []) or []))
                bonus = boost_val if (target & kf_objs) else 0.0
                boosted.append((kf, sc + bonus))
            result = sorted(boosted, key=lambda x: x[1], reverse=True)

        if rerank_options and rerank_options.enable and result:
            try:
                candidates = [item[0] for item in result]
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)
                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=selected_query,
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=self._embedding_for_query(selected_query),
                    options=rerank_options
                )
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]
            except Exception as e:
                logger.error(f"Reranking failed: {e}")

        return result[:top_k]

    async def search_with_video_names(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        video_names: list[str],
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """Search within specific videos by their names (e.g., L21_V026)."""
        # Convert video names to video IDs
        video_ids = []
        for video_name in video_names:
            try:
                # Parse video name format like "L21_V026" to extract video numbers
                if video_name.startswith('L') and '_V' in video_name:
                    parts = video_name.split('_V')
                    if len(parts) == 2:
                        video_num = int(parts[1])  # Extract video number (e.g., 026 -> 26)
                        video_ids.append(video_num)
                else:
                    # Try to parse as just a number
                    video_ids.append(int(video_name))
            except (ValueError, IndexError):
                # Skip invalid video names
                continue
        
        if not video_ids:
            return []  # No valid video IDs found
        
        # Use existing method with the converted video IDs
        return await self.search_with_selected_video_group(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            list_of_include_groups=[],  # Empty groups list
            list_of_include_videos=video_ids,
            rerank_params=rerank_params
        )

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
        # Build rerank/QExp options
        rerank_options = None
        if rerank_params:
            rerank_options = self._build_rerank_options(rerank_params)
        qexp = self._get_qexp_params(rerank_params or {})
        selected_query, obj_list, variants = await self.keyframe_service._refine_query_qexp(query, self.llm, self.visual_extractor)

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

        # Build queries and fuse with filters applied
        queries = [selected_query]
        weights = [1.0]
        if qexp["enable"] and variants:
            vv = sorted(variants, key=lambda v: (v.get("score") or 0.0), reverse=True)
            vv = [v for v in vv if (v.get("score") or 0.0) >= 7.5][: qexp["top_variants"]]
            for v in vv:
                if v.get("query"):
                    queries.append(v.get("query"))
                    weights.append(1.0)

        initial_top_k = top_k
        if rerank_options and rerank_options.enable:
            initial_top_k = max(top_k, rerank_options.sg_top_m)
        per_query_k = initial_top_k if not qexp["enable"] else max(initial_top_k, min(200, int(initial_top_k * 1.5)))

        embeddings = [self._embedding_for_query(q) for q in queries]
        if qexp["fusion"] == "rrf":
            weights = [1.0] * len(queries)
        result, best_qmap = await self.keyframe_service.search_multi_and_fuse(
            embeddings=embeddings,
            weights=weights,
            top_k=per_query_k,
            score_threshold=score_threshold,
            fusion=qexp["fusion"],
            metadata_filter=metadata_dict,
            object_filter=object_dict,
        )
        # Optional object soft-boost only if no explicit object_filter
        cfg = AppSettings()
        GENERIC = {o.lower() for o in getattr(cfg, 'QEXP_OBJECT_GENERIC', {"person","people","man","woman","car","chair","table","phone","laptop","tv"})}
        valid_targets = [o for o in (obj_list or []) if o and o.lower() not in GENERIC]
        if qexp["use_objects"] and valid_targets and object_filter is None:
            target = {o.lower() for o in valid_targets}
            boost_val = float(getattr(cfg, 'QEXP_OBJECT_BOOST', 0.08))
            before_n = len(result)
            boosted = []
            for kf, sc in result:
                kf_objs = set(map(str.lower, getattr(kf, "objects", []) or []))
                bonus = boost_val if (target & kf_objs) else 0.0
                boosted.append((kf, sc + bonus))
            result = sorted(boosted, key=lambda x: x[1], reverse=True)
            logger.debug(f"Object boost applied (metadata-filter): targets={len(target)}, candidates={before_n}")

        # Apply reranking if enabled
        if rerank_options and rerank_options.enable and result:
            try:
                candidates = [item[0] for item in result]
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)

                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=selected_query,  # TODO: accept per-candidate query embeddings using best_qmap
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=self._embedding_for_query(selected_query),
                    options=rerank_options
                )

                # Reconstruct result strictly following reranked order
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]

            except Exception as e:
                logger.error(f"Reranking failed: {e}")

        return result[:top_k]

    # --- Compatibility wrappers for router calls (do not change REST contracts) ---
    async def search_text_exclude_groups(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        exclude_groups: list[int],
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """Router alias -> search_text_with_exlude_group (typo preserved in original)."""
        return await self.search_text_with_exlude_group(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            list_group_exlude=exclude_groups,
            rerank_params=rerank_params,
        )

    async def search_text_include_groups_and_videos(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        include_groups: list[int],
        include_videos: list[int],
        rerank_params: Optional[Dict[str, Any]] = None
    ):
        """Router alias -> search_with_selected_video_group."""
        return await self.search_with_selected_video_group(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            list_of_include_groups=include_groups,
            list_of_include_videos=include_videos,
            rerank_params=rerank_params,
        )

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
                base_embeddings = self.keyframe_service.get_embeddings_for_candidates(candidates)

                # For image search, use a generic query for reranking
                generic_query = "visual content search"

                reranked_candidates = await self.rerank_pipeline.rerank_textual_kis(
                    query=generic_query,
                    base_candidates=candidates,
                    base_embeddings=base_embeddings,
                    query_embedding=embedding,
                    options=rerank_options
                )

                # Reconstruct result strictly following reranked order
                orig_map = {id(c): s for c, s in result}
                result = [(c, orig_map.get(id(c), 0.5)) for c in reranked_candidates]

            except Exception as e:
                logger.error(f"Reranking failed: {e}")

        return result[:top_k]

    async def _refine_query(self, query: str) -> tuple[str, list[str], list[dict]]:
        """
        translate + enhance
        """
        return await self.keyframe_service._refine_query_qexp(query, self.llm, self.visual_extractor)
