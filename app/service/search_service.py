from core.logger import SimpleLogger
from schema.response import KeyframeServiceReponse
from repository.mongo import KeyframeRepository
from repository.milvus import MilvusSearchRequest
from repository.milvus import KeyframeVectorRepository
from typing import Optional, Dict, Any
import json, hashlib
from pathlib import Path
from schema.agent import AgentResponse
import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)
from agent.agent import _preserve_verbatim_quoted, _restore_verbatim_tokens


logger = SimpleLogger(__name__)


class KeyframeQueryService:
    def __init__(
            self,
            keyframe_vector_repo: KeyframeVectorRepository,
            keyframe_mongo_repo: KeyframeRepository,

    ):

        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo = keyframe_mongo_repo
        # Keep embedding map from last vector search (id -> embedding)
        self._last_embedding_map: dict[int, list[float]] = {}
        # Cache for deterministic query refinement: raw query -> (refined, objects)
        self._refine_cache: dict[str, tuple[str, list[str]]] = {}
        # Cache for QExp refinement: raw query -> (refined, objects, variants)
        self._refine_cache_qexp: dict[str, tuple[str, list[str], list[dict]]] = {}

    def _stable_key(self, kf) -> str:
        """Build a stable key for fusion across queries."""
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

    async def _retrieve_keyframes(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys(ids)
        logger.debug(
            f"Retrieved {len(keyframes)} keyframes: {[k.key for k in keyframes[:5]]}")

        keyframe_map = {k.key: k for k in keyframes}
        return_keyframe = [
            keyframe_map[k] for k in ids
        ]
        return return_keyframe

    async def _retrieve_keyframes_with_metadata(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys_with_metadata(ids)
        logger.debug(
            f"Retrieved {len(keyframes)} keyframes with metadata: {[k.key for k in keyframes[:5]]}")

        keyframe_map = {k.key: k for k in keyframes}
        return_keyframe = [
            keyframe_map[k] for k in ids
        ]
        return return_keyframe

    async def _search_keyframes(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = None,
        exclude_indices: list[int] | None = None
    ) -> list[KeyframeServiceReponse]:

        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k,
            exclude_ids=exclude_indices
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        # Build embedding map for rerank consumption
        try:
            self._last_embedding_map = {r.id_: r.embedding for r in sorted_results}
        except Exception:
            self._last_embedding_map = {}

        sorted_ids = [result.id_ for result in sorted_results]

        keyframes = await self._retrieve_keyframes(sorted_ids)

        keyframe_map = {k.key: k for k in keyframes}
        response = []

        for result in sorted_results:
            keyframe = keyframe_map.get(result.id_)
            if keyframe is not None:
                response.append(
                    KeyframeServiceReponse(
                        key=keyframe.key,
                        video_num=keyframe.video_num,
                        group_num=keyframe.group_num,
                        keyframe_num=keyframe.keyframe_num,
                        confidence_score=result.distance
                    )
                )
        return response

    async def _search_keyframes_with_metadata(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = None,
        exclude_indices: list[int] | None = None
    ) -> list[tuple]:
        """Search keyframes and return full keyframe objects with scores"""

        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k,
            exclude_ids=exclude_indices
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        # Build embedding map for rerank consumption
        try:
            self._last_embedding_map = {r.id_: r.embedding for r in sorted_results}
        except Exception:
            self._last_embedding_map = {}

        sorted_ids = [result.id_ for result in sorted_results]

        keyframes = await self._retrieve_keyframes_with_metadata(sorted_ids)

        keyframe_map = {k.key: k for k in keyframes}
        response = []

        for result in sorted_results:
            keyframe = keyframe_map.get(result.id_)
            if keyframe is not None:
                response.append((keyframe, result.distance))
        return response

    async def search_by_text(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = 0.5,
    ):
        return await self._search_keyframes(text_embedding, top_k, score_threshold, None)

    async def search_by_text_with_full_metadata(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = 0.5,
    ):
        """Search and return full keyframe objects with metadata"""
        return await self._search_keyframes_with_metadata(text_embedding, top_k, score_threshold, None)

    async def search_by_text_exclude_ids_with_metadata(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        exclude_ids: list[int] | None
    ):
        """Search excluding IDs and return full keyframe objects with metadata"""
        return await self._search_keyframes_with_metadata(text_embedding, top_k, score_threshold, exclude_ids)

    async def search_by_text_range(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        range_queries: list[tuple[int, int]]
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """

        all_ids = self.keyframe_vector_repo.get_all_id()
        allowed_ids = set()
        for start, end in range_queries:
            allowed_ids.update(range(start, end + 1))

        exclude_ids = [id_ for id_ in all_ids if id_ not in allowed_ids]

        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)

    async def search_by_text_exclude_ids(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        exclude_ids: list[int] | None
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """
        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)

    async def search_by_text_with_metadata_filter(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        object_filter: Optional[Dict[str, Any]] = None
    ) -> list[KeyframeServiceReponse]:
        """
        Search for keyframes with metadata and object filtering
        """
        # First, perform the vector search
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k * 3,  # Get more results to account for filtering
            exclude_ids=None
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        # Apply score threshold filter
        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        # Build embedding map for rerank consumption
        try:
            self._last_embedding_map = {r.id_: r.embedding for r in sorted_results}
        except Exception:
            self._last_embedding_map = {}

        sorted_ids = [result.id_ for result in sorted_results]

        # Apply metadata and object filtering
        if metadata_filter or object_filter:
            keyframes = await self.keyframe_mongo_repo.get_keyframes_with_metadata_filter(
                sorted_ids, metadata_filter, object_filter
            )
        else:
            keyframes = await self._retrieve_keyframes(sorted_ids)

        # Create response mapping
        keyframe_map = {k.key: k for k in keyframes}
        response = []

        for result in sorted_results:
            keyframe = keyframe_map.get(result.id_)
            if keyframe is not None:
                response.append(
                    KeyframeServiceReponse(
                        key=keyframe.key,
                        video_num=keyframe.video_num,
                        group_num=keyframe.group_num,
                        keyframe_num=keyframe.keyframe_num,
                        confidence_score=result.distance
                    )
                )
                # Stop when we have enough results
                if len(response) >= top_k:
                    break

        return response

    async def search_by_text_with_metadata_filter_full(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        object_filter: Optional[Dict[str, Any]] = None
    ) -> list[tuple]:
        """
        Search for keyframes with metadata and object filtering and return full keyframe objects
        """
        # First, perform the vector search
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k * 3,  # Get more results to account for filtering
            exclude_ids=None
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        # Apply score threshold filter
        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        # Build embedding map for rerank consumption
        try:
            self._last_embedding_map = {r.id_: r.embedding for r in sorted_results}
        except Exception:
            self._last_embedding_map = {}

        sorted_ids = [result.id_ for result in sorted_results]

        # Apply metadata and object filtering
        if metadata_filter or object_filter:
            keyframes = await self.keyframe_mongo_repo.get_keyframes_with_metadata_filter(
                sorted_ids, metadata_filter, object_filter
            )
        else:
            keyframes = await self._retrieve_keyframes_with_metadata(sorted_ids)

        # Create response mapping
        keyframe_map = {k.key: k for k in keyframes}
        response = []

        for result in sorted_results:
            keyframe = keyframe_map.get(result.id_)
            if keyframe is not None:
                response.append((keyframe, result.distance))
                # Stop when we have enough results
                if len(response) >= top_k:
                    break

        return response

    async def _refine_query_qexp(self, query: str, llm=None, visual_extractor=None) -> tuple[str, list[str], list[dict]]:
        """Translate+Enhance with LLM, JSON cache, return (selected_query, objects, variants).

        Notes:
        - Uses disk cache to keep QExp deterministic across repeats.
        - Falls back safely to original query when LLM/caching fails.
        """
        # In-memory cache hit
        if query in self._refine_cache_qexp:
            cached = self._refine_cache_qexp[query]
            return cached[0], cached[1], cached[2]

        # Disk cache path
        cache_dir = Path("./cache/qexp"); cache_dir.mkdir(parents=True, exist_ok=True)
        qhash = hashlib.md5(query.encode("utf-8")).hexdigest()
        cpath = cache_dir / f"{qhash}.json"

        data: Optional[dict] = None
        if cpath.exists():
            try:
                data = json.load(open(cpath, "r", encoding="utf-8"))
                logger.debug(f"QExp cache hit for query -> {cpath.name}")
            except Exception:
                data = None

        if data is None and visual_extractor is not None and llm is not None:
            try:
                agent_resp = await visual_extractor.extract_visual_events(query)
                data = agent_resp.model_dump()
                try:
                    json.dump(data, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                    logger.debug(f"QExp cache saved: {cpath.name}")
                except Exception:
                    pass
            except Exception:
                data = None

        if data is None:
            self._refine_cache_qexp[query] = (query, [], [])
            return query, [], []

        agent = AgentResponse(**data)
        selected_query = agent.refined_query.strip() or query

        variants: list[dict] = []
        for v in agent.query_variants or []:
            try:
                item = v if isinstance(v, dict) else v.model_dump()
            except Exception:
                item = {
                    "query": getattr(v, "query", None),
                    "score": getattr(v, "score", None),
                    "rationale": getattr(v, "rationale", None),
                }
            if item.get("query") and str(item.get("query")).strip():
                variants.append(item)

        self._refine_cache_qexp[query] = (selected_query, agent.list_of_objects or [], variants)
        return selected_query, agent.list_of_objects or [], variants

    async def search_multi_and_fuse(
        self,
        embeddings: list[list[float]],
        weights: list[float],
        top_k: int,
        score_threshold: float,
        fusion: str = "rrf",
        exclude_ids: Optional[list[int]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        object_filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[list[tuple], dict[str, int]]:
        """
        For each embedding: vector search (per-query top_k), then fuse results.

        Returns:
          - list[(Keyframe, fused_score)] sorted by fused score desc
          - best_query_for: mapping stable_key -> query index (qi) where candidate achieved max base score

        Fusion modes:
          - "rrf": Reciprocal Rank Fusion, ignores weights; tie-break by max base score
          - "max": Max of base scores across queries, ignores weights
        """
        assert len(embeddings) == len(weights)

        per_query_results: list[list[tuple]] = []
        collected_maps: list[dict[int, list[float]]] = []

        for emb in embeddings:
            if metadata_filter is not None or object_filter is not None:
                res = await self.search_by_text_with_metadata_filter_full(
                    emb, top_k, score_threshold, metadata_filter, object_filter
                )
            elif exclude_ids is not None:
                res = await self.search_by_text_exclude_ids_with_metadata(
                    emb, top_k, score_threshold, exclude_ids
                )
            else:
                res = await self.search_by_text_with_full_metadata(
                    emb, top_k, score_threshold
                )
            per_query_results.append(res)
            try:
                collected_maps.append(dict(self._last_embedding_map))
            except Exception:
                pass

        # Merge embedding maps across all queries for downstream rerank use
        merged_map: dict[int, list[float]] = {}
        for m in collected_maps:
            merged_map.update(m)
        if merged_map:
            self._last_embedding_map = merged_map

        best_query_for: dict[str, int] = {}
        fused: dict[str, list] = {}

        if fusion == "rrf":
            K = 60
            for qi, res in enumerate(per_query_results):
                for rank, (cand, score) in enumerate(res, start=1):
                    skey = self._stable_key(cand)
                    cur = fused.setdefault(skey, [cand, 0.0, 0.0])  # [obj, fused_rrf, max_base]
                    # RRF ignores provided weights to avoid biasing by query count
                    cur[1] += (1.0 / (K + rank))
                    if score > cur[2]:
                        cur[2] = score
                        best_query_for[skey] = qi
            out = sorted(fused.values(), key=lambda x: (x[1], x[2]), reverse=True)
            logger.debug(f"Fusion=RRF, queries={len(embeddings)}, candidates={sum(len(r) for r in per_query_results)} -> fused={len(out)}")
            return [(obj, _fscore) for obj, _fscore, _ in out[:top_k]], best_query_for

        # MAX fusion: use max base score only (no weighting)
        for qi, res in enumerate(per_query_results):
            for cand, score in res:
                skey = self._stable_key(cand)
                cur = fused.setdefault(skey, [cand, 0.0])
                if score > cur[1]:
                    cur[1] = score
                    best_query_for[skey] = qi
        out = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        logger.debug(f"Fusion=MAX, queries={len(embeddings)}, candidates={sum(len(r) for r in per_query_results)} -> fused={len(out)}")
        return [(obj, sc) for obj, sc in out[:top_k]], best_query_for

    async def _refine_query(self, query: str, llm=None, visual_extractor=None) -> tuple[str, list[str]]:
        """
        Use LLM to:
        1) Translate Vietnamese→English (or keep original if English)
        2) Enhance the query for visual retrieval
        3) Optionally extract relevant objects via VisualEventExtractor
        Fallback to original on any error or if LLM unavailable.
        """
        # Deterministic caching for identical queries
        if query in self._refine_cache:
            cached = self._refine_cache[query]
            return cached[0], cached[1]

        if llm is None:
            return query, []

        # Step 1: Translation + enhancement via structured schema
        # Protect any quoted substrings to prevent translation/modification inside quotes
        _protected_query, _verb_map = _preserve_verbatim_quoted(query)

        translation_prompt = (
            "You are a retrieval query optimizer.\n"
            "1) Detect language; if Vietnamese, translate to English. If already English, keep text.\n"
            "2) Produce an enhanced English query optimized for semantic video/keyframe retrieval:\n"
            "   - Use concrete visual nouns, actions, colors, settings, spatial relations\n"
            "   - Remove filler; keep core visual concepts\n"
            "IMPORTANT: If the input contains placeholders like [[VERBATIM_1]], [[VERBATIM_2]], etc., copy them EXACTLY\n"
            "as-is into both fields; do not translate or modify them—they already encapsulate original quoted text.\n"
            "Return strict JSON: {\"translated_query\":\"<english>\", \"enhanced_query\":\"<optimized>\"}.\n\n"
            f"Input: \"\"\"{_protected_query}\"\"\""
        )

        refined_text = query
        try:
            from schema.agent import QueryRefineResponse
            # Favor deterministic generation when supported by provider
            try:
                resp = await llm.as_structured_llm(QueryRefineResponse, temperature=0.0).acomplete(translation_prompt)
            except Exception:
                resp = await llm.as_structured_llm(QueryRefineResponse).acomplete(translation_prompt)
            obj = resp.raw  # pydantic object
            translated_text = (obj.translated_query or query).strip()
            refined_text = (obj.enhanced_query or translated_text or query).strip()
            # Restore any protected quoted substrings
            translated_text = _restore_verbatim_tokens(translated_text, _verb_map)
            refined_text = _restore_verbatim_tokens(refined_text, _verb_map)
            logger.debug(
                f"Query refined: '{query}' -> '{refined_text}' (translated: '{translated_text}')")
        except Exception:
            refined_text = query
            logger.debug(
                f"Query refinement failed, using original: '{refined_text}'")

        # Step 2: Optional object suggestions via VisualEventExtractor
        objects: list[str] = []
        try:
            if visual_extractor is not None:
                agent_resp = await visual_extractor.extract_visual_events(refined_text)
                refined_from_extractor = (
                    agent_resp.refined_query or refined_text).strip()
                if refined_from_extractor != refined_text:
                    logger.debug(
                        f"Agent refined: '{refined_text}' -> '{refined_from_extractor}'")
                    refined_text = refined_from_extractor
                objects = agent_resp.list_of_objects or []
        except Exception:
            pass

        # Store in cache for deterministic repetition
        self._refine_cache[query] = (refined_text, objects)
        return refined_text, objects

    # (Removed duplicate _refine_query_qexp implementation to avoid ambiguity)


    def get_embeddings_for_candidates(self, candidates: list[Any]) -> list[list[float]]:
        """Return embeddings for given candidate keyframes from last vector search.

        Raises if any embedding is missing to keep rerank strict and explicit.
        """
        if not self._last_embedding_map:
            raise ValueError("No embeddings captured from last vector search")
        embs: list[list[float]] = []
        missing: list[int] = []
        for cand in candidates:
            key = getattr(cand, 'key', None)
            e = self._last_embedding_map.get(key)
            if e is None:
                missing.append(key)
            else:
                embs.append(e)
        if missing:
            raise ValueError(f"Missing embeddings for candidate ids: {missing[:5]} ...")
        return embs
