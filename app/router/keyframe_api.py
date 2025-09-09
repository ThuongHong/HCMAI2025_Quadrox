
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
import io
import json

from schema.request import (
    TextSearchRequest,
    TextSearchWithExcludeGroupsRequest,
    TextSearchWithSelectedGroupsAndVideosRequest,
    TextSearchWithVideoNamesRequest,
    TextSearchWithMetadataFilterRequest,
    ImageSearchRequest,
    TemporalEnrichRequest,
)
from schema.response import KeyframeServiceReponse, SingleKeyframeDisplay, KeyframeDisplay
from controller.query_controller import QueryController
from core.dependencies import get_query_controller
from core.logger import SimpleLogger
from retrieval.temporal_search.service import temporal_enrich, video_id_from_nums


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

    # Extract rerank parameters (SuperGlobal only)
    rerank_params = {}
    for field, value in request.dict().items():
        if value is None:
            continue
        if field.startswith(('rerank', 'rr_', 'sg_', 'qexp_')):
            rerank_params[field] = value
        elif field in ('w_sg', 'final_top_k'):
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

    # Extract rerank parameters from request (SG only)
    rerank_params = {}
    for field, value in request.dict().items():
        if value is None:
            continue
        if field.startswith(('rerank', 'rr_', 'sg_', 'qexp_')):
            rerank_params[field] = value
        elif field in ('w_sg', 'final_top_k'):
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
        if field.startswith(('rerank', 'rr_', 'sg_', 'cap_', 'llm_', 'w_', 'final_top_k', 'qexp_')) and value is not None:
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
    "/search/video-names",
    response_model=KeyframeDisplay,
    summary="Text search within specific videos by name",
    description="""
    Perform text-based search for keyframes within specific videos identified by their names.
    
    This endpoint allows you to search only within specified videos using their string identifiers
    (e.g., "L21_V026", "L22_V110"). This is perfect for progressive filtering in multi-stage workflows.
    
    **Parameters:**
    - **query**: The search text
    - **top_k**: Maximum number of results to return  
    - **score_threshold**: Minimum confidence score
    - **video_names**: List of video names to search within (e.g., ["L21_V026", "L22_V110"])
    
    **Use Cases:**
    - **Progressive Search**: Use results from Stage 1 to narrow Stage 2 search
    - **Sequential Filtering**: "red hat man" → extract videos → "going home" → extract videos → "dog jumping"
    - **Multi-Scene Queries**: Complex scenarios with multiple sequential actions
    - **Result Refinement**: Narrow down search scope based on previous findings
    
    **Workflow Example:**
    1. Stage 1: Search "red hat man" → Get 100 results → Extract unique video names
    2. Stage 2: Search "going to house" within those videos → Get 30 results → Extract video names  
    3. Stage 3: Search "dog jumping" within Stage 2 videos → Get 10 refined results
    
    **Example:**
    ```json
    {
        "query": "dog jumping on man's hand",
        "top_k": 10,
        "score_threshold": 0.6,
        "video_names": ["L21_V026", "L22_V110", "L23_V045"]
    }
    ```
    """,
    response_description="List of matching keyframes from specified videos"
)
async def search_keyframes_by_video_names(
    request: TextSearchWithVideoNamesRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """
    Search for keyframes within specific videos by their names.
    """

    logger.info(
        f"Text search with video names: query='{request.query}', video_names={request.video_names}")

    # Extract rerank parameters from request
    rerank_params = {}
    for field, value in request.dict().items():
        if field.startswith(('rerank', 'rr_', 'sg_', 'w_', 'final_top_k', 'qexp_')) and value is not None:
            rerank_params[field] = value

    results = await controller.search_with_video_names(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        video_names=request.video_names,
        rerank_params=rerank_params if rerank_params else None
    )

    logger.info(f"Found {len(results)} results within specified videos")

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
        if field.startswith(('rerank', 'rr_', 'sg_', 'cap_', 'llm_', 'w_', 'final_top_k', 'qexp_')) and value is not None:
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


@router.get(
    "/search/advanced",
    response_model=KeyframeDisplay,
    summary="Advanced text search with reranking options",
    description="""
    Perform advanced text-based search for keyframes with comprehensive reranking options.
    
    This endpoint provides full control over the multi-stage reranking pipeline including:
    - **SuperGlobal reranking**: Fast global feature aggregation

    
    **Rerank Parameters:**
    - **rerank**: Enable reranking (0=off, 1=on)
    - **rerank_mode**: "auto" (system decides) or "custom" (manual control)

    **Example URLs:**
    ```
    # SuperGlobal only (fast)
    /search/advanced?q=người đàn ông&rerank=1&rr_superglobal=1&rr_caption=0&rr_llm=0
    
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

    # Rerank method switches (SG only)
    rr_superglobal: Optional[int] = Query(
        default=None, description="Enable SuperGlobal rerank (0=off, 1=on)"),

    # SuperGlobal parameters
    sg_top_m: Optional[int] = Query(
        default=None, ge=1, le=10000, description="SuperGlobal top-M candidates"),
    sg_qexp_k: Optional[int] = Query(
        default=None, ge=1, le=100, description="Query expansion K"),
    sg_img_knn: Optional[int] = Query(
        default=None, ge=1, le=100, description="Image KNN parameter"),
    sg_alpha: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="SuperGlobal alpha blend"),
    sg_beta: Optional[float] = Query(
        default=None, ge=0.0, le=5.0, description="SuperGlobal beta exponent"),
    sg_p_query: Optional[float] = Query(
        default=None, ge=1.0, le=1000.0, description="Query-side GeM p (~max)"),
    sg_gem_p: Optional[float] = Query(
        default=None, ge=0.1, le=1000.0, description="Legacy GeM parameter (compat)"),
    w_sg: Optional[float] = Query(
        default=None, ge=0.0, le=5.0, description="SuperGlobal weight"),

    # Final output
    final_top_k: Optional[int] = Query(
        default=None, ge=1, le=1000, description="Final top-K results"),

    # Advanced filtering options
    metadata_filter: Optional[str] = Query(
        default=None, description="JSON string containing metadata filters"),
    object_filter: Optional[str] = Query(
        default=None, description="JSON string containing object filters"),
    include_groups: Optional[str] = Query(
        default=None, description="Comma-separated list of group IDs to include"),
    include_videos: Optional[str] = Query(
        default=None, description="Comma-separated list of video IDs to include"),
    exclude_groups: Optional[str] = Query(
        default=None, description="Comma-separated list of group IDs to exclude"),

    controller: QueryController = Depends(get_query_controller)
):
    """
    Advanced search with full reranking control via query parameters.
    """

    logger.info(f"Advanced search: query='{q}', rerank={rerank}")

    # Build rerank params from query parameters (SG only)
    rerank_params = {}
    local_vars = locals()
    for param_name in ['rerank', 'rerank_mode', 'rr_superglobal',
                       'sg_top_m', 'sg_qexp_k', 'sg_img_knn',
                       'sg_alpha', 'sg_beta', 'sg_p_query', 'sg_gem_p', 'w_sg',
                       'final_top_k']:
        if param_name in local_vars and local_vars[param_name] is not None:
            rerank_params[param_name] = local_vars[param_name]

    # Validate rerank parameters (SG only)
    if rerank_params.get('rerank') == 1 and rerank_params.get('rerank_mode') == 'custom':
        if rerank_params.get('rr_superglobal') != 1:
            raise HTTPException(
                status_code=400,
                detail="When rerank=1 and rerank_mode=custom, rr_superglobal=1 must be enabled."
            )

    # Parse filters if provided
    parsed_metadata_filter = None
    parsed_object_filter = None
    
    if metadata_filter:
        try:
            parsed_metadata_filter = json.loads(metadata_filter)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata_filter JSON")
    
    if object_filter:
        try:
            parsed_object_filter = json.loads(object_filter)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid object_filter JSON")
    
    # Parse group/video filters
    parsed_include_groups = []
    parsed_include_videos = []
    parsed_exclude_groups = []
    
    if include_groups:
        try:
            parsed_include_groups = [int(x.strip()) for x in include_groups.split(',') if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid include_groups format")
    
    if include_videos:
        try:
            parsed_include_videos = [int(x.strip()) for x in include_videos.split(',') if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid include_videos format")
    
    if exclude_groups:
        try:
            parsed_exclude_groups = [int(x.strip()) for x in exclude_groups.split(',') if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid exclude_groups format")

    # Choose the appropriate search method based on filters provided
    if parsed_metadata_filter or parsed_object_filter:
        # Need to convert dict to Pydantic objects for existing controller method
        from schema.request import MetadataFilter, ObjectFilter
        
        metadata_obj = None
        if parsed_metadata_filter:
            metadata_obj = MetadataFilter(**parsed_metadata_filter)
        
        object_obj = None 
        if parsed_object_filter:
            object_obj = ObjectFilter(**parsed_object_filter)
            
        results = await controller.search_text_with_metadata_filter(
            query=q,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_obj,
            object_filter=object_obj,
            rerank_params=rerank_params if rerank_params else None
        )
    elif parsed_exclude_groups:
        results = await controller.search_text_exclude_groups(
            query=q,
            top_k=top_k,
            score_threshold=score_threshold,
            exclude_groups=parsed_exclude_groups,
            rerank_params=rerank_params if rerank_params else None
        )
    elif parsed_include_groups or parsed_include_videos:
        results = await controller.search_text_include_groups_and_videos(
            query=q,
            top_k=top_k,
            score_threshold=score_threshold,
            include_groups=parsed_include_groups,
            include_videos=parsed_include_videos,
            rerank_params=rerank_params if rerank_params else None
        )
    else:
        # Default search
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


@router.post(
    "/temporal/enrich",
    summary="Temporal enrichment around a pivot keyframe",
    description="""
    Compute temporal neighborhood and clusters around a pivot keyframe.

    Provide either `pivot_video_id` (e.g. L01_V001) or (`pivot_group_num`, `pivot_video_num`).
    Provide at least one of: `pivot_n`, or (`pivot_frame_idx` & `pivot_pts_time`).

    Modes:
    - auto: Expand window adaptively using edge confidence
    - interactive: Use ±delta seconds around pivot_pts_time
    """,
)
async def temporal_enrich_endpoint(request: TemporalEnrichRequest):
    try:
        # Resolve video_id
        if request.pivot_video_id and isinstance(request.pivot_video_id, str):
            video_id = request.pivot_video_id
        elif request.pivot_group_num is not None and request.pivot_video_num is not None:
            video_id = video_id_from_nums(int(request.pivot_group_num), int(request.pivot_video_num))
        else:
            raise HTTPException(status_code=400, detail="Missing video identifier: provide pivot_video_id or (pivot_group_num & pivot_video_num)")

        out = temporal_enrich(
            mode=request.mode,
            video_id=video_id,
            pivot_n=request.pivot_n,
            pivot_frame_idx=request.pivot_frame_idx,
            pivot_pts_time=request.pivot_pts_time,
            pivot_score=request.pivot_score,
            delta=request.delta or 5.0,
            gap_seconds=10.0,
        )
        return JSONResponse(content=out)
    except FileNotFoundError as e:
        logger.warning(f"Temporal enrich skipped: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Temporal enrich failed: {e}")
        raise HTTPException(status_code=500, detail="Temporal enrichment failed")


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
