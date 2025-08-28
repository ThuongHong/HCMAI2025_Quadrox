
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
import io

from schema.request import (
    TextSearchRequest,
    TextSearchWithExcludeGroupsRequest,
    TextSearchWithSelectedGroupsAndVideosRequest,
    TextSearchWithMetadataFilterRequest,
    ImageSearchRequest,
)
from schema.response import KeyframeServiceReponse, SingleKeyframeDisplay, KeyframeDisplay
from controller.query_controller import QueryController
from core.dependencies import get_query_controller
from core.logger import SimpleLogger


logger = SimpleLogger(__name__)


router = APIRouter(
    prefix="/keyframe",
    tags=["keyframe"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/search",
    response_model=KeyframeDisplay,
    summary="Simple text search for keyframes",
    description="""
    Perform a simple text-based search for keyframes using semantic similarity.
    
    This endpoint converts the input text query to an embedding and searches for 
    the most similar keyframes in the database.
    
    **Parameters:**
    - **query**: The search text (1-1000 characters)
    - **top_k**: Maximum number of results to return (1-100, default: 10)
    - **score_threshold**: Minimum confidence score (0.0-1.0, default: 0.0)
    
    **Returns:**
    List of keyframes with their metadata and confidence scores, ordered by similarity.
    
    **Example:**
    ```json
    {
        "query": "person walking in the park",
        "top_k": 5,
        "score_threshold": 0.7
    }
    ```
    """,
    response_description="List of matching keyframes with confidence scores"
)
async def search_keyframes(
    request: TextSearchRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes using text query with semantic similarity.
    """

    logger.info(
        f"Text search: '{request.query}' | top_k={request.top_k}, threshold={request.score_threshold}")

    results = await controller.search_text(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )

    logger.info(f"Found {len(results)} results")
    display_results = [
        SingleKeyframeDisplay(**controller.convert_model_to_display(result))
        for result in results
    ]
    return KeyframeDisplay(results=display_results)


@router.post(
    "/search/exclude-groups",
    response_model=KeyframeDisplay,
    summary="Text search with group exclusion",
    description="""
    Perform text-based search for keyframes while excluding specific groups.
    
    This endpoint allows you to search for keyframes while filtering out 
    results from specified groups (e.g., to avoid certain video categories).
    
    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **exclude_groups**: List of group IDs to exclude from results
    
    **Use Cases:**
    - Exclude specific video categories or datasets
    - Filter out content from certain time periods
    - Remove specific collections from search results
    
    **Example:**
    ```json
    {
        "query": "sunset landscape",
        "top_k": 15,
        "score_threshold": 0.6,
        "exclude_groups": [1, 3, 7]
    }
    ```
    """,
    response_description="List of matching keyframes excluding specified groups"
)
async def search_keyframes_exclude_groups(
    request: TextSearchWithExcludeGroupsRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes with group exclusion filtering.
    """

    logger.info(
        f"Text search with group exclusion: query='{request.query}', exclude_groups={request.exclude_groups}")

    results: list[KeyframeServiceReponse] = await controller.search_text_with_exlude_group(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_group_exlude=request.exclude_groups
    )

    logger.info(
        f"Found {len(results)} results excluding groups {request.exclude_groups}")

    display_results = [
        SingleKeyframeDisplay(**controller.convert_model_to_display(result))
        for result in results
    ]
    return KeyframeDisplay(results=display_results)


@router.post(
    "/search/selected-groups-videos",
    response_model=KeyframeDisplay,
    summary="Text search within selected groups and videos",
    description="""
    Perform text-based search for keyframes within specific groups and videos only.
    
    This endpoint allows you to limit your search to specific groups and videos,
    effectively creating a filtered search scope.
    
    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **include_groups**: List of group IDs to search within
    - **include_videos**: List of video IDs to search within
    
    **Behavior:**
    - Only keyframes from the specified groups AND videos will be searched
    - If a keyframe belongs to an included group OR an included video, it will be considered
    - Empty lists mean no filtering for that category
    
    **Use Cases:**
    - Search within specific video collections
    - Focus on particular time periods or datasets
    - Limit search to curated content sets
    
    **Example:**
    ```json
    {
        "query": "car driving on highway",
        "top_k": 20,
        "score_threshold": 0.5,
        "include_groups": [2, 4, 6],
        "include_videos": [101, 102, 203, 204]
    }
    ```
    """,
    response_description="List of matching keyframes from selected groups and videos"
)
async def search_keyframes_selected_groups_videos(
    request: TextSearchWithSelectedGroupsAndVideosRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes within selected groups and videos.
    """

    logger.info(
        f"Text search with selection: query='{request.query}', include_groups={request.include_groups}, include_videos={request.include_videos}")

    results = await controller.search_with_selected_video_group(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_of_include_groups=request.include_groups,
        list_of_include_videos=request.include_videos
    )

    logger.info(f"Found {len(results)} results within selected groups/videos")

    display_results = [
        SingleKeyframeDisplay(**controller.convert_model_to_display(result))
        for result in results
    ]
    return KeyframeDisplay(results=display_results)


@router.post(
    "/search/metadata-filter",
    response_model=KeyframeDisplay,
    summary="Text search with advanced filtering",
    description="""
    Perform text-based search for keyframes with advanced filtering capabilities.
    
    This endpoint allows you to search for keyframes while applying sophisticated filters
    based on video metadata and detected objects in keyframes.
    
    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **metadata_filter**: Advanced metadata filtering criteria including:
      - **authors**: Filter by specific channel/author names
      - **keywords**: Filter by video keywords/tags
      - **min_length/max_length**: Filter by video duration
      - **title_contains**: Filter by title content
      - **description_contains**: Filter by description content
      - **date_from/date_to**: Filter by publish date range (DD/MM/YYYY format)
    - **object_filter**: Object detection filtering criteria including:
      - **objects**: List of object names to filter by (max 20)
      - **mode**: Filter mode - "any" (at least one object) or "all" (all objects present)
    
    **Use Cases:**
    - Search within content from specific creators
    - Find videos with particular themes or tags
    - Filter by video length (short clips vs long videos)
    - Search for specific topics in titles or descriptions
    - Find keyframes containing specific objects (cars, people, buildings, etc.)
    - Combine metadata and object filters for precise results
    
    **Example:**
    ```json
    {
        "query": "city street scene",
        "top_k": 15,
        "score_threshold": 0.6,
        "metadata_filter": {
            "authors": ["Travel Channel"],
            "keywords": ["city", "urban"],
            "min_length": 300
        },
        "object_filter": {
            "objects": ["car", "building", "person"],
            "mode": "any"
        }
    }
    ```
    """,
    response_description="List of matching keyframes filtered by metadata and object criteria"
)
async def search_keyframes_with_metadata_filter(
    request: TextSearchWithMetadataFilterRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes with advanced metadata and object filtering.
    """
    filter_info = []
    if request.metadata_filter:
        filter_info.append("metadata")
    if request.object_filter:
        # Show first 5 objects
        objects_str = ", ".join(request.object_filter.objects[:5])
        if len(request.object_filter.objects) > 5:
            objects_str += f" (+{len(request.object_filter.objects) - 5} more)"
        filter_info.append(
            f"objects[{request.object_filter.mode}]: {objects_str}")

    logger.info(
        f"Advanced search: query='{request.query}', filters=[{', '.join(filter_info)}]")

    results = await controller.search_text_with_metadata_filter(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        metadata_filter=request.metadata_filter,
        object_filter=request.object_filter
    )

    logger.info(f"Found {len(results)} results with advanced filtering")

    display_results = [
        SingleKeyframeDisplay(**controller.convert_model_to_display(result))
        for result in results
    ]
    return KeyframeDisplay(results=display_results)


@router.post(
    "/search/image",
    response_model=KeyframeDisplay,
    summary="Image-based search for keyframes",
    description="""
    Perform an image-based search for keyframes using visual similarity.
    
    This endpoint converts the uploaded image to an embedding and searches for 
    the most visually similar keyframes in the database.
    
    **Parameters:**
    - **file**: The image file to search with (JPEG, PNG, etc.)
    - **top_k**: Maximum number of results to return (1-100, default: 10)
    - **score_threshold**: Minimum confidence score (0.0-1.0, default: 0.0)
    
    **Returns:**
    List of keyframes with their metadata and confidence scores, ordered by visual similarity.
    
    **Supported formats:**
    - JPEG, PNG, BMP, TIFF, WebP
    
    **Example usage:**
    Upload an image file and specify search parameters to find visually similar keyframes.
    """,
    response_description="List of matching keyframes with visual similarity scores"
)
async def search_keyframes_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=10, ge=1, le=500,
                       description="Number of top results to return"),
    score_threshold: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold"),
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes using image query with visual similarity.
    """

    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed (for PNG with transparency, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        logger.info(
            f"Image search request: filename='{file.filename}', top_k={top_k}, threshold={score_threshold}")

        results = await controller.search_image(
            image=image,
            top_k=top_k,
            score_threshold=score_threshold
        )

        logger.info(
            f"Found {len(results)} results for image: '{file.filename}'")
        display_results = [
            SingleKeyframeDisplay(
                **controller.convert_model_to_display(result))
            for result in results
        ]
        return KeyframeDisplay(results=display_results)

    except Exception as e:
        logger.error(f"Error processing image search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {str(e)}")
