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

    def __str__(self) -> str:
        """Compact string representation for logging"""
        return f"Keyframe(key={self.key}, L{self.group_num:02d}_V{self.video_num:03d}_{self.keyframe_num:03d}, author='{self.author[:20]}...', title='{self.title[:30]}...')"
    
    def __repr__(self) -> str:
        """Use compact representation"""
        return self.__str__()

    class Settings:
        name = "keyframes"