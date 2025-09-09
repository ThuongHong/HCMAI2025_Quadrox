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
    
    # Object detection fields
    objects: Annotated[Optional[List[str]], Indexed()] = Field(None, description="Detected objects in keyframe")

    def __str__(self) -> str:
        """Compact string representation for logging"""
        # Safe slicing to avoid TypeError when fields are None
        author = (self.author or "")[:20]
        title = (self.title or "")[:30]
        return (
            f"Keyframe(key={self.key}, "
            f"L{self.group_num:02d}_V{self.video_num:03d}_{self.keyframe_num:03d}, "
            f"author='{author}...', title='{title}...')"
        )
    
    def __repr__(self) -> str:
        """Use compact representation"""
        return self.__str__()

    class Settings:
        name = "keyframes"
