from pathlib import Path
import json

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from service import ModelService, KeyframeQueryService
from schema.response import KeyframeServiceReponse
from llama_index.core.llms import LLM
from agent.agent import VisualEventExtractor

class QueryController:
    
    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService,
        llm: LLM,
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service
        self.llm = llm
        self.visual_extractor = VisualEventExtractor(llm) if llm is not None else None
    
    def convert_model_to_path(
        self,
        model: KeyframeServiceReponse
    ) -> tuple[str, float]:
        return os.path.join(self.data_folder, f"L{model.group_num:02d}/L{model.group_num:02d}_V{model.video_num:03d}/{model.keyframe_num:03d}.jpg"), model.confidence_score
    
        
    async def search_text(
        self, 
        query: str,
        top_k: int,
        score_threshold: float
    ):

        refined_query, objects = await self._refine_query(query)

        embedding = self.model_service.embedding(refined_query).tolist()[0]

        result = await self.keyframe_service.search_by_text(embedding, top_k, score_threshold)
        return result


    async def search_text_with_exlude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[int]
    ):
        exclude_ids = [
            int(k) for k, v in self.id2index.items()
            if int(v.split('/')[0]) in list_group_exlude
        ]

        refined_query, objects = await self._refine_query(query)
                
        embedding = self.model_service.embedding(refined_query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)
        return result


    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int]  ,
        list_of_include_videos: list[int]  
    ):     


        exclude_ids = None
        if len(list_of_include_groups) > 0   and len(list_of_include_videos) == 0:
            print("hi")
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[0]) not in list_of_include_groups
            ]
        
        elif len(list_of_include_groups) == 0   and len(list_of_include_videos) >0 :
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[1]) not in list_of_include_videos
            ]

        elif len(list_of_include_groups) == 0  and len(list_of_include_videos) == 0 :
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

        embedding = self.model_service.embedding(refined_query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)
        return result

    async def _refine_query(self, query: str) -> tuple[str, list[str]]:
        """
        Delegate query refinement to the search service.
        """
        return await self.keyframe_service._refine_query(query, self.llm, self.visual_extractor)