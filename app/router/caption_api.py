from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.logger import SimpleLogger
from core.settings import AppSettings
from core.dependencies import get_model_service
from service.model_service import ModelService
from service.caption_search import hybrid_rrf


logger = SimpleLogger(__name__)


router = APIRouter(
    prefix="/caption",
    tags=["caption"],
    responses={404: {"description": "Not found"}},
)


class CaptionSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k_dense: Optional[int] = Field(default=200, ge=1, le=1000)
    top_k_bm25: Optional[int] = Field(default=200, ge=1, le=1000)
    rrf_k: Optional[int] = Field(default=60, ge=1, le=10000)
    return_k: Optional[int] = Field(default=100, ge=1, le=1000)


@router.get("/enabled")
async def caption_enabled(settings: AppSettings = Depends(lambda: AppSettings())):
    return {"enabled": bool(settings.CAPTION_SEARCH_ENABLED)}


@router.post("/search")
async def caption_search(
    request: CaptionSearchRequest,
    model_service: ModelService = Depends(get_model_service),
    settings: AppSettings = Depends(lambda: AppSettings()),
):
    if not settings.CAPTION_SEARCH_ENABLED:
        raise HTTPException(status_code=404, detail="Caption search disabled")
    try:
        results = hybrid_rrf(
            query=request.query,
            model_service=model_service,
            top_k_dense=int(request.top_k_dense or 200),
            top_k_bm25=int(request.top_k_bm25 or 200),
            rrf_k=int(request.rrf_k or 60),
            return_k=int(request.return_k or 100),
        )
        return results
    except FileNotFoundError as e:
        logger.warning(f"Caption search missing index: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Caption search failed: {e}")
        raise HTTPException(status_code=500, detail="Caption search failed")

