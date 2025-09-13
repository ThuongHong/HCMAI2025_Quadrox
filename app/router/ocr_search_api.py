"""
OCR Search API Router
Provides REST API endpoints for OCR text search functionality.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

from service.ocr_search_service import OCRSearchService, OCRSearchResult

logger = logging.getLogger(__name__)

# Initialize the router
router = APIRouter(prefix="/api/v1/ocr", tags=["OCR Search"])

# Initialize OCR search service
try:
    ocr_service = OCRSearchService()
    logger.info("OCR search service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OCR search service: {e}")
    ocr_service = None


# Request/Response Models
class OCRSearchRequest(BaseModel):
    """OCR search request model"""
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    limit: int = Field(default=50, ge=1, le=500, description="Maximum number of results")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")
    video_filters: Optional[List[str]] = Field(default=None, description="Video IDs to filter by (e.g., ['L01_V001', 'L02_V005'])")
    include_ocr_details: bool = Field(default=False, description="Include detailed OCR detection information")
    include_video_metadata: bool = Field(default=True, description="Include video metadata (title, URL, etc.)")


class OCRSearchResponse(BaseModel):
    """OCR search response model"""
    results: List[Dict[str, Any]]
    total_found: int
    query: str
    search_params: Dict[str, Any]


class OCRVideoSearchRequest(BaseModel):
    """OCR video search request model"""
    video_id: str = Field(..., description="Video ID (e.g., 'L01_V001')")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of results")


class OCRStatsResponse(BaseModel):
    """OCR statistics response model"""
    total_records: int
    unique_videos: int
    average_confidence: float
    records_with_text: int
    records_without_text: int


# API Endpoints
@router.post("/search", response_model=OCRSearchResponse)
async def search_ocr_text(request: OCRSearchRequest):
    """
    Search for text in OCR results using full-text search.
    
    This endpoint searches through OCR text extracted from video keyframes using SQLite FTS5.
    Optionally enriches results with video metadata including YouTube URLs and titles.
    
    **Parameters:**
    - **query**: The search text (supports FTS5 query syntax)
    - **limit**: Maximum number of results to return (1-500)
    - **min_confidence**: Minimum OCR confidence threshold (0.0-1.0)
    - **video_filters**: Optional list of video IDs to limit search scope
    - **include_ocr_details**: Include full OCR detection JSON data
    - **include_video_metadata**: Include video metadata (YouTube URLs, titles, etc.)
    
    **FTS5 Query Examples:**
    - Simple text: `"person walking"`
    - AND operator: `"person AND walking"`
    - OR operator: `"person OR people"`
    - Phrase search: `'"exact phrase"'`
    - Prefix search: `"walk*"`
    
    **Returns:**
    - List of matching OCR results with keyframe information
    - Each result includes frame path for image display
    - Video metadata including watch URLs when available
    """
    if ocr_service is None:
        raise HTTPException(status_code=500, detail="OCR search service not available")
    
    try:
        if request.include_ocr_details:
            results = ocr_service.search_with_context(
                query=request.query,
                limit=request.limit,
                min_confidence=request.min_confidence,
                video_filters=request.video_filters,
                include_ocr_details=True
            )
        elif request.include_video_metadata:
            # Use new async method with metadata enrichment
            search_results = await ocr_service.search_text_with_metadata(
                query=request.query,
                limit=request.limit,
                min_confidence=request.min_confidence,
                video_filters=request.video_filters,
                include_video_metadata=True
            )
            results = [result.dict() for result in search_results]
        else:
            search_results = ocr_service.search_text(
                query=request.query,
                limit=request.limit,
                min_confidence=request.min_confidence,
                video_filters=request.video_filters
            )
            results = [result.dict() for result in search_results]
        
        return OCRSearchResponse(
            results=results,
            total_found=len(results),
            query=request.query,
            search_params={
                "limit": request.limit,
                "min_confidence": request.min_confidence,
                "video_filters": request.video_filters,
                "include_ocr_details": request.include_ocr_details,
                "include_video_metadata": request.include_video_metadata
            }
        )
    
    except Exception as e:
        logger.error(f"OCR search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search")
async def search_ocr_text_get(
    query: str = Query(..., description="Search query text", min_length=1, max_length=1000),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    video_filters: Optional[str] = Query(None, description="Comma-separated video IDs (e.g., 'L01_V001,L02_V005')"),
    include_ocr_details: bool = Query(False, description="Include detailed OCR detection information"),
    include_video_metadata: bool = Query(True, description="Include video metadata (YouTube URLs, titles, etc.)")
):
    """
    Search for text in OCR results using GET request.
    
    Same functionality as POST /search but using query parameters for simple integration.
    """
    if ocr_service is None:
        raise HTTPException(status_code=500, detail="OCR search service not available")
    
    # Parse video filters
    video_filter_list = None
    if video_filters:
        video_filter_list = [v.strip() for v in video_filters.split(',') if v.strip()]
    
    request = OCRSearchRequest(
        query=query,
        limit=limit,
        min_confidence=min_confidence,
        video_filters=video_filter_list,
        include_ocr_details=include_ocr_details,
        include_video_metadata=include_video_metadata
    )
    
    return await search_ocr_text(request)


@router.post("/video", response_model=OCRSearchResponse)
async def get_video_ocr(request: OCRVideoSearchRequest):
    """
    Get all OCR results for a specific video.
    
    **Parameters:**
    - **video_id**: Video identifier (e.g., 'L01_V001')
    - **limit**: Maximum number of results to return
    
    **Returns:**
    - All OCR detections for the specified video, ordered by frame number
    """
    if ocr_service is None:
        raise HTTPException(status_code=500, detail="OCR search service not available")
    
    try:
        results = ocr_service.search_by_video(
            video_id=request.video_id,
            limit=request.limit
        )
        
        return OCRSearchResponse(
            results=[result.dict() for result in results],
            total_found=len(results),
            query=f"video:{request.video_id}",
            search_params={
                "video_id": request.video_id,
                "limit": request.limit
            }
        )
    
    except Exception as e:
        logger.error(f"Video OCR retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get video OCR: {str(e)}")


@router.get("/video/{video_id}")
async def get_video_ocr_get(
    video_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results")
):
    """
    Get all OCR results for a specific video using GET request.
    
    **Path Parameters:**
    - **video_id**: Video identifier (e.g., 'L01_V001')
    
    **Query Parameters:**
    - **limit**: Maximum number of results to return
    """
    request = OCRVideoSearchRequest(video_id=video_id, limit=limit)
    return await get_video_ocr(request)


@router.get("/stats", response_model=OCRStatsResponse)
async def get_ocr_statistics():
    """
    Get OCR database statistics.
    
    **Returns:**
    - Total number of OCR records
    - Number of unique videos
    - Average confidence score
    - Records with/without text
    """
    if ocr_service is None:
        raise HTTPException(status_code=500, detail="OCR search service not available")
    
    try:
        stats = ocr_service.get_video_statistics()
        return OCRStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get OCR statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/ocr-detail")
async def get_ocr_detail(
    ocr_path: str = Query(..., description="OCR file path (e.g., 'ocr\\L01_V001\\001.json')")
):
    """
    Get detailed OCR information from JSON file.
    
    **Parameters:**
    - **ocr_path**: Path to OCR JSON file
    
    **Returns:**
    - Full OCR detection data including bounding boxes and confidence scores
    """
    if ocr_service is None:
        raise HTTPException(status_code=500, detail="OCR search service not available")
    
    try:
        ocr_details = ocr_service.get_ocr_detail(ocr_path)
        if ocr_details is None:
            raise HTTPException(status_code=404, detail="OCR file not found")
        
        return {
            "ocr_path": ocr_path,
            "detections": ocr_details
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get OCR details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get OCR details: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for OCR search service.
    """
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR search service not available")
    
    try:
        stats = ocr_service.get_video_statistics()
        return {
            "status": "healthy",
            "service": "ocr_search",
            "database": "connected",
            "total_records": stats["total_records"],
            "unique_videos": stats["unique_videos"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")