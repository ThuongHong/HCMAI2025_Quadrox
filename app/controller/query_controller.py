from pathlib import Path
import json
from typing import Optional, Dict, Any

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
from schema.request import MetadataFilter


class QueryController:
    
    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service

    
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
        path = os.path.join(self.data_folder, f"L{keyframe.group_num:02d}/L{keyframe.group_num:02d}_V{keyframe.video_num:03d}/{keyframe.keyframe_num:03d}.jpg")
        
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
            'watch_url': keyframe.watch_url
        }
    
        
    async def search_text(
        self, 
        query: str,
        top_k: int,
        score_threshold: float
    ):
        embedding = self.model_service.embedding(query).tolist()[0]
        result = await self.keyframe_service.search_by_text_with_full_metadata(embedding, top_k, score_threshold)
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

        embedding = self.model_service.embedding(query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids_with_metadata(embedding, top_k, score_threshold, exclude_ids)
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



        embedding = self.model_service.embedding(query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids_with_metadata(embedding, top_k, score_threshold, exclude_ids)
        return result
    
    async def search_text_with_metadata_filter(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        metadata_filter: Optional[MetadataFilter] = None
    ):
        """
        Search for keyframes with metadata filtering
        """
        embedding = self.model_service.embedding(query).tolist()[0]
        
        # Convert MetadataFilter to dict format for the service
        filter_dict = None
        if metadata_filter:
            filter_dict = {}
            if metadata_filter.authors:
                filter_dict["authors"] = metadata_filter.authors
            if metadata_filter.keywords:
                filter_dict["keywords"] = metadata_filter.keywords
            if metadata_filter.min_length is not None:
                filter_dict["min_length"] = metadata_filter.min_length
            if metadata_filter.max_length is not None:
                filter_dict["max_length"] = metadata_filter.max_length
            if metadata_filter.title_contains:
                filter_dict["title_contains"] = metadata_filter.title_contains
            if metadata_filter.description_contains:
                filter_dict["description_contains"] = metadata_filter.description_contains
            # Note: Date filtering would need additional implementation
        
        result = await self.keyframe_service.search_by_text_with_metadata_filter_full(
            embedding, top_k, score_threshold, filter_dict
        )
        return result

    

        

