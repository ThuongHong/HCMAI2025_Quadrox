
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

    # Extract rerank parameters from request
    rerank_params = {}
    for field, value in request.dict().items():
        if field.startswith(('rerank', 'rr_', 'sg_', 'cap_', 'llm_', 'w_', 'final_top_k')) and value is not None:
            rerank_params[field] = value

    results = await controller.search_text(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        rerank_params=rerank_params if rerank_params else None
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

    # Extract rerank parameters from request
    rerank_params = {}
    for field, value in request.dict().items():
        if field.startswith(('rerank', 'rr_', 'sg_', 'cap_', 'llm_', 'w_', 'final_top_k')) and value is not None:
            rerank_params[field] = value

    results: list[KeyframeServiceReponse] = await controller.search_text_with_exlude_group(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_group_exlude=request.exclude_groups,
        rerank_params=rerank_params if rerank_params else None
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

    # Extract rerank parameters from request
    rerank_params = {}
    for field, value in request.dict().items():
        if field.startswith(('rerank', 'rr_', 'sg_', 'cap_', 'llm_', 'w_', 'final_top_k')) and value is not None:
            rerank_params[field] = value

    results = await controller.search_with_selected_video_group(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        list_of_include_groups=request.include_groups,
        list_of_include_videos=request.include_videos,
        rerank_params=rerank_params if rerank_params else None
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

    # Extract rerank parameters from request
    rerank_params = {}
    for field, value in request.dict().items():
        if field.startswith(('rerank', 'rr_', 'sg_', 'cap_', 'llm_', 'w_', 'final_top_k')) and value is not None:
            rerank_params[field] = value

    results = await controller.search_text_with_metadata_filter(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        metadata_filter=request.metadata_filter,
        object_filter=request.object_filter,
        rerank_params=rerank_params if rerank_params else None
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
@router.get(
    "/search/advanced",
    response_model=KeyframeDisplay,
    summary="Advanced text search with reranking options",
    description="""
    Perform advanced text-based search for keyframes with comprehensive reranking options.
    
    This endpoint provides full control over the multi-stage reranking pipeline including:
    - **SuperGlobal reranking**: Fast global feature aggregation
    - **Caption reranking**: Vietnamese caption generation and matching
    - **LLM reranking**: Direct relevance scoring with multimodal LLM
    
    **Rerank Parameters:**
    - **rerank**: Enable reranking (0=off, 1=on)
    - **rerank_mode**: "auto" (system decides) or "custom" (manual control)
    - **rr_superglobal**, **rr_caption**, **rr_llm**: Enable individual methods
    - **sg_top_m**, **cap_top_t**, **llm_top_t**: Control processing scope
    - **w_sg**, **w_cap**, **w_llm**: Adjust method weights
    
    **Example URLs:**
    ```
    # SuperGlobal only (fast)
    /search/advanced?q=người đàn ông&rerank=1&rr_superglobal=1&rr_caption=0&rr_llm=0
    
    # SuperGlobal + Caption (moderate speed)
    /search/advanced?q=người đàn ông&rerank=1&rr_superglobal=1&rr_caption=1&cap_top_t=15
    
    # Full pipeline (slower but most accurate)
    /search/advanced?q=người đàn ông&rerank=1&rr_superglobal=1&rr_caption=1&rr_llm=1&llm_top_t=3
    ```
    """,
    response_description="List of reranked keyframes with confidence scores"
)
async def search_keyframes_advanced(
    q: str = Query(..., description="Search query text",
                   min_length=1, max_length=1000),
    top_k: int = Query(default=10, ge=1, le=500,
                       description="Number of top results to return"),
    score_threshold: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold"),

    # Rerank master controls
    rerank: Optional[int] = Query(
        default=None, description="Enable reranking (0=off, 1=on)"),
    rerank_mode: Optional[str] = Query(
        default=None, description="Reranking mode: auto or custom"),

    # Rerank method switches
    rr_superglobal: Optional[int] = Query(
        default=None, description="Enable SuperGlobal rerank (0=off, 1=on)"),
    rr_caption: Optional[int] = Query(
        default=None, description="Enable Caption rerank (0=off, 1=on)"),
    rr_llm: Optional[int] = Query(
        default=None, description="Enable LLM rerank (0=off, 1=on)"),

    # SuperGlobal parameters
    sg_top_m: Optional[int] = Query(
        default=None, ge=1, le=10000, description="SuperGlobal top-M candidates"),
    sg_qexp_k: Optional[int] = Query(
        default=None, ge=1, le=100, description="Query expansion K"),
    sg_img_knn: Optional[int] = Query(
        default=None, ge=1, le=100, description="Image KNN parameter"),
    sg_gem_p: Optional[float] = Query(
        default=None, ge=0.1, le=10.0, description="GeM pooling parameter"),
    w_sg: Optional[float] = Query(
        default=None, ge=0.0, le=5.0, description="SuperGlobal weight"),

    # Caption parameters
    cap_top_t: Optional[int] = Query(
        default=None, ge=1, le=100, description="Caption rerank top-T"),
    cap_model: Optional[str] = Query(
        default=None, description="Caption model name"),
    cap_max_tokens: Optional[int] = Query(
        default=None, ge=1, le=512, description="Caption max tokens"),
    cap_temp: Optional[float] = Query(
        default=None, ge=0.0, le=2.0, description="Caption temperature"),
    w_cap: Optional[float] = Query(
        default=None, ge=0.0, le=5.0, description="Caption weight"),

    # LLM parameters
    llm_top_t: Optional[int] = Query(
        default=None, ge=1, le=20, description="LLM rerank top-T"),
    llm_model: Optional[str] = Query(
        default=None, description="LLM model name"),
    llm_timeout: Optional[int] = Query(
        default=None, ge=1, le=300, description="LLM timeout seconds"),
    w_llm: Optional[float] = Query(
        default=None, ge=0.0, le=5.0, description="LLM weight"),

    # Final output
    final_top_k: Optional[int] = Query(
        default=None, ge=1, le=1000, description="Final top-K results"),

    controller: QueryController = Depends(get_query_controller)
):
    """
    Advanced search with full reranking control via query parameters.
    """

    logger.info(f"Advanced search: query='{q}', rerank={rerank}")

    # Build rerank params from query parameters
    rerank_params = {}
    local_vars = locals()
    for param_name in ['rerank', 'rerank_mode', 'rr_superglobal', 'rr_caption', 'rr_llm',
                       'sg_top_m', 'sg_qexp_k', 'sg_img_knn', 'sg_gem_p', 'w_sg',
                       'cap_top_t', 'cap_model', 'cap_max_tokens', 'cap_temp', 'w_cap',
                       'llm_top_t', 'llm_model', 'llm_timeout', 'w_llm', 'final_top_k']:
        if param_name in local_vars and local_vars[param_name] is not None:
            rerank_params[param_name] = local_vars[param_name]

    # Validate rerank parameters
    if rerank_params.get('rerank') == 1 and rerank_params.get('rerank_mode') == 'custom':
        methods_enabled = any([
            rerank_params.get('rr_superglobal') == 1,
            rerank_params.get('rr_caption') == 1,
            rerank_params.get('rr_llm') == 1
        ])
        if not methods_enabled:
            raise HTTPException(
                status_code=400,
                detail="When rerank=1 and rerank_mode=custom, at least one rerank method (rr_superglobal, rr_caption, rr_llm) must be enabled. Try: rr_superglobal=1"
            )

    results = await controller.search_text(
        query=q,
        top_k=top_k,
        score_threshold=score_threshold,
        rerank_params=rerank_params if rerank_params else None
    )

    logger.info(f"Found {len(results)} results with advanced reranking")

    display_results = [
        SingleKeyframeDisplay(**controller.convert_model_to_display(result))
        for result in results
    ]
    return KeyframeDisplay(results=display_results)


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
