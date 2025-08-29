from schema.request import MetadataFilter, ObjectFilter
from agent.agent import VisualEventExtractor
from llama_index.core.llms import LLM
from schema.response import KeyframeServiceReponse
from service import ModelService, KeyframeQueryService
from pathlib import Path
import json
from typing import Optional, Dict, Any
from PIL import Image
from typing import List, Dict, Tuple
from collections import defaultdict

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
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service
        self.llm = llm
        self.visual_extractor = VisualEventExtractor(
            llm) if llm is not None else None

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
            'objects': keyframe.objects  # Include detected objects
        }

    async def search_text(
        self,
        query: str,
        top_k: int,
        score_threshold: float
    ):

        refined_query, objects = await self._refine_query(query)

        embedding = self.model_service.embedding(refined_query).tolist()[0]
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

        refined_query, objects = await self._refine_query(query)
        embedding = self.model_service.embedding(refined_query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids_with_metadata(embedding, top_k, score_threshold, exclude_ids)
        return result

    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int],
        list_of_include_videos: list[int]
    ):

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

        embedding = self.model_service.embedding(refined_query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids_with_metadata(embedding, top_k, score_threshold, exclude_ids)
        return result

    async def search_text_with_metadata_filter(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        metadata_filter: Optional[MetadataFilter] = None,
        object_filter: Optional[ObjectFilter] = None
    ):
        """
        Search for keyframes with metadata and object filtering
        """
        refined_query, objects = await self._refine_query(query)

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

        result = await self.keyframe_service.search_by_text_with_metadata_filter_full(
            embedding, top_k, score_threshold, metadata_dict, object_dict
        )
        return result

    async def search_image(
        self,
        image,
        top_k: int,
        score_threshold: float
    ):
        """Search keyframes using image query"""
        embedding = self.model_service.image_embedding(image).tolist()[0]
        result = await self.keyframe_service.search_by_text_with_full_metadata(embedding, top_k, score_threshold)
        return result

    async def _refine_query(self, query: str) -> tuple[str, list[str]]:
        """
        translate + enhance
        """
        return await self.keyframe_service._refine_query(query, self.llm, self.visual_extractor)
    




    def _minmax(self, d: Dict[int, float]) -> Dict[int, float]:
        if not d: return {}
        lo, hi = min(d.values()), max(d.values())
        if hi <= lo: return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    async def _probe_ids_scores(
        self, text: str, top_k: int, thr: float
    ) -> Tuple[Dict[int, float], Dict[int, Tuple[int,int,int]]]:
        emb = self.model_service.embedding(text).tolist()
        res = await self.keyframe_service.search_by_text(
            text_embedding=emb, top_k=top_k, score_threshold=thr
        )
        id2s, id2meta = {}, {}
        for r in res:
            id2s[r.key] = max(id2s.get(r.key, 0.0), float(r.confidence_score))
            id2meta[r.key] = (int(r.group_num), int(r.video_num), int(r.keyframe_num))
        return id2s, id2meta

    async def search_text_tc(
        self,
        query: str,
        targets: List[str] | None,
        contexts: List[str] | None,
        top_k: int,
        score_threshold: float,
        top_k_full: int = 400,
        top_k_each: int = 150,
        alpha: float = 0.6,
        beta: float = 0.4
    ) -> list[tuple]:
        # 0) refine để có Q_full
        refined_query, _ = await self._refine_query(query)

        # 1) Q_full
        S_full_raw, id2meta = await self._probe_ids_scores(refined_query, top_k_full, score_threshold)

        # 2) T/C
        per_lbl_raw: Dict[str, Dict[int,float]] = {}
        for lab in (targets or []):
            d, m = await self._probe_ids_scores(lab, top_k_each, score_threshold)
            per_lbl_raw[lab] = d; id2meta.update(m)
        for lab in (contexts or []):
            d, m = await self._probe_ids_scores(lab, top_k_each, score_threshold)
            per_lbl_raw[lab] = d; id2meta.update(m)

        # 3) Chuẩn hoá từng kênh
        S_full = self._minmax(S_full_raw)
        per_lbl = {lab: self._minmax(d) for lab, d in per_lbl_raw.items()}

        # 4) obj_conf = max trên mọi nhãn
        obj_conf: Dict[int, float] = defaultdict(float)
        for d in per_lbl.values():
            for k, v in d.items():
                if v > obj_conf[k]: obj_conf[k] = v

        # 5) Hợp điểm nhẹ
        ids = set(S_full.keys()) | set(obj_conf.keys())
        merged = [(k, alpha*S_full.get(k, 0.0) + beta*obj_conf.get(k, 0.0)) for k in ids]
        merged.sort(key=lambda x: x[31], reverse=True)
        final_ids = [k for k, _ in merged[:top_k]]

        # 6) Nạp metadata chỉ cho top_k
        keyframes = await self.keyframe_service._retrieve_keyframes_with_metadata(final_ids)
        kmap = {k.key: k for k in keyframes}
        return [(kmap[k], float(s)) for k, s in merged[:top_k] if k in kmap]

