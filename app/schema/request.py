from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from fastapi import UploadFile


class BaseSearchRequest(BaseModel):
    """Base search request with common parameters"""
    query: str = Field(..., description="Search query text",
                       min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=500,
                       description="Number of top results to return")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold")

    # Query Expansion (QExp) optional overrides
    qexp_enable: Optional[bool] = Field(default=None, description="Enable new QExp engine for this request")
    qexp_top_variants: Optional[int] = Field(default=None, description="Top-N variants to keep")
    qexp_fusion: Optional[str] = Field(default=None, description='Fusion method: "max" | "rrf"')
    qexp_use_objects: Optional[bool] = Field(default=None, description="Auto object soft-boost when objects present")

    # Rerank parameters
    rerank: Optional[int] = Field(
        default=None, description="Enable reranking (0=off, 1=on)")
    rerank_mode: Optional[Literal["auto", "custom"]] = Field(
        default=None, description="Reranking mode")

    # Rerank method switches (SuperGlobal only)
    rr_superglobal: Optional[int] = Field(
        default=None, description="Enable SuperGlobal rerank (0=off, 1=on)")

    # SuperGlobal parameters
    sg_top_m: Optional[int] = Field(
        default=None, ge=1, le=10000, description="SuperGlobal top-M candidates")
    sg_qexp_k: Optional[int] = Field(
        default=None, ge=1, le=100, description="Query expansion K")
    sg_img_knn: Optional[int] = Field(
        default=None, ge=1, le=100, description="Image KNN parameter")
    sg_gem_p: Optional[float] = Field(
        default=None, ge=0.1, le=10.0, description="GeM pooling parameter")
    w_sg: Optional[float] = Field(
        default=None, ge=0.0, le=5.0, description="SuperGlobal weight")

    # Final output
    final_top_k: Optional[int] = Field(
        default=None, ge=1, le=1000, description="Final top-K results")


class BaseImageSearchRequest(BaseModel):
    """Base image search request with common parameters"""
    top_k: int = Field(default=10, ge=1, le=500,
                       description="Number of top results to return")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold")


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


class TextSearchWithVideoNamesRequest(BaseSearchRequest):
    """Text search request with specific video names (e.g., L21_V026)"""
    video_names: List[str] = Field(
        default_factory=list,
        description="List of video names to search within (e.g., ['L21_V026', 'L22_V110'])",
        example=["L21_V026", "L22_V110"]
    )


class MetadataFilter(BaseModel):
    """Metadata filter criteria"""
    authors: Optional[List[str]] = Field(
        None, description="Filter by specific authors/channel names"
    )
    keywords: Optional[List[str]] = Field(
        None, description="Filter by keywords that must be present in video keywords"
    )
    keywords_mode: Optional[Literal["any", "all"]] = Field(
        "any", description="Keyword matching mode: 'any' for at least one, 'all' for all keywords"
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
        None, description="Filter by title containing specific text (deprecated, use title_terms)"
    )
    title_terms: Optional[List[str]] = Field(
        None, description="Filter by title containing specific terms"
    )
    title_mode: Optional[Literal["any", "all"]] = Field(
        "any", description="Title matching mode: 'any' for at least one term, 'all' for all terms"
    )
    description_contains: Optional[str] = Field(
        None, description="Filter by description containing specific text (deprecated, use description_terms)"
    )
    description_terms: Optional[List[str]] = Field(
        None, description="Filter by description containing specific terms"
    )
    description_mode: Optional[Literal["any", "all"]] = Field(
        "any", description="Description matching mode: 'any' for at least one term, 'all' for all terms"
    )


class ObjectFilter(BaseModel):
    """Object detection filter criteria"""
    objects: List[str] = Field(
        ..., description="List of object names to filter by", max_items=20
    )
    mode: Literal["any", "all"] = Field(
        default="any", description="Filter mode: 'any' for at least one match, 'all' for all objects present"
    )


class TextSearchWithMetadataFilterRequest(BaseSearchRequest):
    """Text search request with metadata filtering"""
    metadata_filter: Optional[MetadataFilter] = Field(
        None, description="Metadata filter criteria"
    )
    object_filter: Optional[ObjectFilter] = Field(
        None, description="Object detection filter criteria"
    )


# Optional per-request overrides for Query Expansion
class QueryParams(BaseModel):
    qexp_enable: Optional[bool] = None
    qexp_top_variants: Optional[int] = None
    qexp_fusion: Optional[str] = None         # "max" | "rrf"
    qexp_use_objects: Optional[bool] = None   # auto-apply object filter if possible


class TemporalEnrichRequest(BaseModel):
    """Temporal enrichment request for a given pivot keyframe.

    Provide either pivot_video_id (e.g., L01_V001) or (pivot_group_num, pivot_video_num).
    Provide at least one of: pivot_n, or (pivot_frame_idx & pivot_pts_time), or pivot_pts_time with pivot_n.
    """
    mode: Literal["auto", "interactive"] = Field(..., description="Temporal mode")
    # Identify video
    pivot_video_id: Optional[str] = Field(None, description="Video ID like L01_V001")
    pivot_group_num: Optional[int] = Field(None, description="Group number (e.g., 1 for L01)")
    pivot_video_num: Optional[int] = Field(None, description="Video number (e.g., 1 for V001)")
    # Pivot location
    pivot_n: Optional[int] = Field(None, description="Keyframe order within video (n)")
    pivot_frame_idx: Optional[int] = Field(None, description="Absolute video frame index")
    pivot_pts_time: Optional[float] = Field(None, description="Presentation timestamp in seconds")
    pivot_score: Optional[float] = Field(None, description="Similarity/confidence score of pivot (optional)")
    # Interactive controls
    delta: Optional[float] = Field(5.0, ge=1.0, le=30.0, description="Half window seconds for interactive mode")
