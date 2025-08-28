import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)


from repository.milvus import KeyframeVectorRepository
from repository.milvus import MilvusSearchRequest
from repository.mongo import KeyframeRepository

from schema.response import KeyframeServiceReponse

class KeyframeQueryService:
    def __init__(
            self, 
            keyframe_vector_repo: KeyframeVectorRepository,
            keyframe_mongo_repo: KeyframeRepository,
            
        ):

        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo= keyframe_mongo_repo


    async def _retrieve_keyframes(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys(ids)
        print(keyframes[:5])
  
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
    

    async def search_by_text(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = 0.5,
    ):
        return await self._search_keyframes(text_embedding, top_k, score_threshold, None)   
    

    async def search_by_text_range(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        range_queries: list[tuple[int,int]]
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
    
    async def _refine_query(self, query: str, llm=None, visual_extractor=None) -> tuple[str, list[str]]:
            """
            Use LLM to:
            1) Translate Vietnamese→English (or keep original if English)
            2) Enhance the query for visual retrieval
            3) Optionally extract relevant objects via VisualEventExtractor
            Fallback to original on any error or if LLM unavailable.
            """
            if llm is None:
                return query, []

            # Step 1: Translation + enhancement via structured schema
            translation_prompt = (
                "You are a retrieval query optimizer.\n"
                "1) Detect language; if Vietnamese, translate to English. If already English, keep text.\n"
                "2) Produce an enhanced English query optimized for semantic video/keyframe retrieval:\n"
                "   - Use concrete visual nouns, actions, colors, settings, spatial relations\n"
                "   - Remove filler; keep core visual concepts\n"
                "Return strict JSON: {\"translated_query\":\"<english>\", \"enhanced_query\":\"<optimized>\"}.\n\n"
                f"Input: \"\"\"{query}\"\"\""
            )

            refined_text = query
            try:
                from schema.agent import QueryRefineResponse
                resp = await llm.as_structured_llm(QueryRefineResponse).acomplete(translation_prompt)
                obj = resp.raw  # pydantic object
                translated_text = (obj.translated_query or query).strip()
                refined_text = (obj.enhanced_query or translated_text or query).strip()
                print(f"Final refined query: '{refined_text}' | translated: '{translated_text}'")
            except Exception:
                refined_text = query
                print(f"Final refined query: '{refined_text}' (fallback)")

            # Step 2: Optional object suggestions via VisualEventExtractor
            objects: list[str] = []
            try:
                if visual_extractor is not None:
                    agent_resp = await visual_extractor.extract_visual_events(refined_text)
                    refined_from_extractor = (agent_resp.refined_query or refined_text).strip()
                    if refined_from_extractor != refined_text:
                        print(f"Agent rephrase: '{refined_text}' -> '{refined_from_extractor}'")
                        refined_text = refined_from_extractor
                    objects = agent_resp.list_of_objects or []
            except Exception:
                pass

            return refined_text, objects