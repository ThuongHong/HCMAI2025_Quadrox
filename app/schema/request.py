from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import UploadFile


class BaseSearchRequest(BaseModel):
    """Base search request with common parameters"""
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=500, description="Number of top results to return")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold")


class BaseImageSearchRequest(BaseModel):
    """Base image search request with common parameters"""
    top_k: int = Field(default=10, ge=1, le=500, description="Number of top results to return")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold")


class TextSearchRequest(BaseSearchRequest):
    """Simple text search request"""
    pass


class ImageSearchRequest(BaseImageSearchRequest):
    """Simple image search request"""
    pass


class TextSearchWithExcludeGroupsRequest(BaseSearchRequest):
    """Text search request with group exclusion"""
    exclude_groups: List[int] = Field(
        default_factory=list,
        description="List of group IDs to exclude from search results",
    )


class TextSearchWithSelectedGroupsAndVideosRequest(BaseSearchRequest):
    """Text search request with specific group and video selection"""
    include_groups: List[int] = Field(
        default_factory=list,
        description="List of group IDs to include in search results",
    )
    include_videos: List[int] = Field(
        default_factory=list,
        description="List of video IDs to include in search results",
    )


class MetadataFilter(BaseModel):
    """Metadata filter criteria"""
    authors: Optional[List[str]] = Field(
        None, description="Filter by specific authors/channel names"
    )
    keywords: Optional[List[str]] = Field(
        None, description="Filter by keywords that must be present in video keywords"
    )
    min_length: Optional[int] = Field(
        None, ge=0, description="Minimum video length in seconds"
    )
    max_length: Optional[int] = Field(
        None, ge=0, description="Maximum video length in seconds"
    )
    date_from: Optional[str] = Field(
        None, description="Filter videos published from this date (DD/MM/YYYY format)"
    )
    date_to: Optional[str] = Field(
        None, description="Filter videos published until this date (DD/MM/YYYY format)"
    )
    title_contains: Optional[str] = Field(
        None, description="Filter by title containing specific text"
    )
    description_contains: Optional[str] = Field(
        None, description="Filter by description containing specific text"
    )


class TextSearchWithMetadataFilterRequest(BaseSearchRequest):
    """Text search request with metadata filtering"""
    metadata_filter: Optional[MetadataFilter] = Field(
        None, description="Metadata filter criteria"
    )


