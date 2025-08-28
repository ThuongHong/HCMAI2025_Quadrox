from pydantic import BaseModel, Field
from typing import Optional, List


class KeyframeServiceReponse(BaseModel):
    key: int = Field(..., description="Keyframe key")
    video_num: int = Field(..., description="Video ID")
    group_num: int = Field(..., description="Group ID")
    keyframe_num: int = Field(..., description="Keyframe number")
    confidence_score: float = Field(..., description="Keyframe number")
    


class SingleKeyframeDisplay(BaseModel):
    path: str
    score: float
    video_id: Optional[int] = None
    group_id: Optional[int] = None
    # Video metadata fields
    author: Optional[str] = None
    channel_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    length: Optional[int] = None
    publish_date: Optional[str] = None
    thumbnail_url: Optional[str] = None
    watch_url: Optional[str] = None
    # Object detection fields
    objects: Optional[List[str]] = None

class KeyframeDisplay(BaseModel):
    results: list[SingleKeyframeDisplay]