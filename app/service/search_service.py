import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)

from typing import Optional, Dict, Any

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

    async def _retrieve_keyframes_with_metadata(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys_with_metadata(ids)
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
    
    async def search_by_text_with_metadata_filter(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> list[KeyframeServiceReponse]:
        """
        Search for keyframes with metadata filtering
        """
        # First, perform the vector search
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k * 3,  # Get more results to account for metadata filtering
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

        sorted_ids = [result.id_ for result in sorted_results]

        # Apply metadata filtering
        if metadata_filter:
            keyframes = await self.keyframe_mongo_repo.get_keyframes_with_metadata_filter(
                sorted_ids, metadata_filter
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
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> list[tuple]:
        """
        Search for keyframes with metadata filtering and return full keyframe objects
        """
        # First, perform the vector search
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k * 3,  # Get more results to account for metadata filtering
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

        sorted_ids = [result.id_ for result in sorted_results]

        # Apply metadata filtering
        if metadata_filter:
            keyframes = await self.keyframe_mongo_repo.get_keyframes_with_metadata_filter(
                sorted_ids, metadata_filter
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
    


    




    
        



        

        

        
        
        


        

        







