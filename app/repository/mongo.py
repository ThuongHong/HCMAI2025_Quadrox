"""
The implementation of Keyframe repositories. The following class is responsible for getting the keyframe by many ways
"""

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

# from typing import Any
from typing import Any, Dict, List, Optional
from models.keyframe import Keyframe
from common.repository import MongoBaseRepository
from schema.interface import KeyframeInterface




class KeyframeRepository(MongoBaseRepository[Keyframe]):
    async def get_keyframe_by_list_of_keys(
        self, keys: list[int]
    ):
        result = await self.find({"key": {"$in": keys}})
        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            ) for keyframe in result

        ]

    async def get_keyframe_by_list_of_keys_with_metadata(
        self, keys: list[int]
    ):
        """Get full keyframe objects with metadata"""
        result = await self.find({"key": {"$in": keys}})
        return result  # Return full Keyframe objects

    async def get_keyframe_by_video_num(
        self, 
        video_num: int,
    ):
        result = await self.find({"video_num": video_num})
        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            ) for keyframe in result
        ]

    async def get_keyframe_by_keyframe_num(
        self, 
        keyframe_num: int,
    ):
        result = await self.find({"keyframe_num": keyframe_num})
        return [
            KeyframeInterface(
                key=keyframe.key,
                video_num=keyframe.video_num,
                group_num=keyframe.group_num,
                keyframe_num=keyframe.keyframe_num
            ) for keyframe in result
        ]   

    async def get_keyframes_with_metadata_filter(
        self,
        keys: List[int],
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[KeyframeInterface]:
        """
        Get keyframes by keys with optional metadata filtering
        """
        query = {"key": {"$in": keys}}
        
        if metadata_filter:
            and_conditions = [{"key": {"$in": keys}}]
            
            # Add metadata filters to the query
            if metadata_filter.get("authors"):
                # Contains matching for authors (case-insensitive)
                authors = metadata_filter["authors"]
                author_conditions = []
                for author in authors:
                    # Use regex to find any author containing the search term
                    author_conditions.append({"author": {"$regex": author, "$options": "i"}})
                and_conditions.append({"$or": author_conditions})
            
            if metadata_filter.get("keywords"):
                # Case-insensitive keyword matching - any keyword in array contains search term
                keyword_conditions = []
                for keyword in metadata_filter["keywords"]:
                    keyword_conditions.append({"keywords": {"$elemMatch": {"$regex": keyword, "$options": "i"}}})
                and_conditions.append({"$or": keyword_conditions})
            
            if metadata_filter.get("min_length") is not None:
                and_conditions.append({"length": {"$gte": metadata_filter["min_length"]}})
            
            if metadata_filter.get("max_length") is not None:
                and_conditions.append({"length": {"$lte": metadata_filter["max_length"]}})
            
            if metadata_filter.get("title_contains"):
                and_conditions.append({"title": {"$regex": metadata_filter["title_contains"], "$options": "i"}})
            
            if metadata_filter.get("description_contains"):
                and_conditions.append({"description": {"$regex": metadata_filter["description_contains"], "$options": "i"}})
            
            if metadata_filter.get("date_from"):
                # Simple date filtering - convert DD/MM/YYYY to YYYYMMDD for string comparison
                date_from = metadata_filter["date_from"]
                try:
                    day, month, year = date_from.split('/')
                    yyyymmdd_from = f"{year}{month.zfill(2)}{day.zfill(2)}"
                    and_conditions.append({
                        "$expr": {
                            "$gte": [
                                {"$concat": [
                                    {"$arrayElemAt": [{"$split": ["$publish_date", "/"]}, 2]},
                                    {"$arrayElemAt": [{"$split": ["$publish_date", "/"]}, 1]}, 
                                    {"$arrayElemAt": [{"$split": ["$publish_date", "/"]}, 0]}
                                ]},
                                yyyymmdd_from
                            ]
                        }
                    })
                except:
                    # Fallback to string comparison
                    and_conditions.append({"publish_date": {"$gte": date_from}})
            
            if metadata_filter.get("date_to"):
                # Simple date filtering - convert DD/MM/YYYY to YYYYMMDD for string comparison
                date_to = metadata_filter["date_to"]
                try:
                    day, month, year = date_to.split('/')
                    yyyymmdd_to = f"{year}{month.zfill(2)}{day.zfill(2)}"
                    and_conditions.append({
                        "$expr": {
                            "$lte": [
                                {"$concat": [
                                    {"$arrayElemAt": [{"$split": ["$publish_date", "/"]}, 2]},
                                    {"$arrayElemAt": [{"$split": ["$publish_date", "/"]}, 1]}, 
                                    {"$arrayElemAt": [{"$split": ["$publish_date", "/"]}, 0]}
                                ]},
                                yyyymmdd_to
                            ]
                        }
                    })
                except:
                    # Fallback to string comparison
                    and_conditions.append({"publish_date": {"$lte": date_to}})
            
            # Use $and to combine all conditions
            query = {"$and": and_conditions}
        
        result = await self.find(query)
        return result  # Return full Keyframe objects with metadata


