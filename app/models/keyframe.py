from beanie import Document, Indexed
from typing import Annotated, Optional, List
from pydantic import BaseModel, Field


class Keyframe(Document):
    key: Annotated[int, Indexed(unique=True)]
    video_num: Annotated[int, Indexed()]
    group_num: Annotated[int, Indexed()]
    keyframe_num: Annotated[int, Indexed()]
    
    # Metadata fields
    author: Optional[str] = Field(None, description="Video author/channel name")
    channel_id: Optional[str] = Field(None, description="YouTube channel ID")
    title: Optional[str] = Field(None, description="Video title")
    description: Optional[str] = Field(None, description="Video description")
    keywords: Optional[List[str]] = Field(None, description="Video keywords/tags")
    length: Optional[int] = Field(None, description="Video length in seconds")
    publish_date: Optional[str] = Field(None, description="Video publish date")
    thumbnail_url: Optional[str] = Field(None, description="Video thumbnail URL")
    watch_url: Optional[str] = Field(None, description="Video watch URL")

    class Settings:
        name = "keyframes"



    