"""
Caption API Router
Provides image captioning functionality using Vietnamese captioning system
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import time
from typing import Optional
import tempfile
import os
from pathlib import Path

from core.logger import SimpleLogger
from retrieval.rerank.vintern_captioner import VinternCaptionerCPU
from core.settings import RerankSettings

# Initialize router
router = APIRouter(prefix="/api/caption", tags=["caption"])

# Global captioner instance
_captioner = None


def get_captioner():
    """Get or initialize the VinternCaptionerCPU instance"""
    global _captioner
    if _captioner is None:
        settings = RerankSettings()
        _captioner = VinternCaptionerCPU(
            model_path=settings.RERANK_CAPTION_VINTERN_MODEL_PATH,
            fallback_to_public=True
        )
    return _captioner


@router.post("/")
async def generate_caption(
    image: UploadFile = File(...),
    style: str = Form(default="dense"),
    allow_fallback: bool = Form(default=True),
    max_new_tokens: Optional[int] = Form(default=None),
):
    """
    Generate caption for uploaded image

    Parameters:
    - **image**: Image file (JPEG, PNG, etc.)
    - **style**: Caption style ('dense', 'short', 'tags', 'ocr')
    - **allow_fallback**: Allow fallback to BLIP if Vintern fails
    - **max_new_tokens**: Maximum tokens for generation

    Returns:
    - **caption**: Generated caption text
    - **style**: Caption style used
    - **source**: Model source (vintern, blip, etc.)
    - **processing_time_ms**: Processing time in milliseconds
    """
    logger = SimpleLogger(__name__)
    start_time = time.time()

    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            # Check file extension as fallback
            filename = image.filename or ""
            valid_extensions = ['.jpg', '.jpeg',
                                '.png', '.bmp', '.tiff', '.webp']
            if not any(filename.lower().endswith(ext) for ext in valid_extensions):
                raise HTTPException(
                    status_code=400,
                    detail="File must be an image (supported: JPG, PNG, BMP, TIFF, WebP)"
                )

        # Validate style
        valid_styles = ['dense', 'short', 'tags', 'ocr']
        if style not in valid_styles:
            raise HTTPException(
                status_code=400,
                detail=f"Style must be one of: {valid_styles}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await image.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Get captioner
            captioner = get_captioner()

            # Generate caption
            caption_params = {"style": style}
            if max_new_tokens is not None:
                caption_params["max_new_tokens"] = max_new_tokens

            result = captioner.caption_image(tmp_file_path, **caption_params)

            processing_time = (time.time() - start_time) * 1000

            # Defensive schema handling - safely get fields with defaults
            caption_text = result.get("caption", "")
            source = result.get("source", "unknown")
            result_style = result.get("style", style)
            success = result.get("success", True)
            error = result.get("error")

            logger.info(
                f"Caption generated: {source} model, "
                f"style={result_style}, time={processing_time:.0f}ms"
            )

            # Handle result based on success flag
            if success and caption_text:
                return {
                    "caption": caption_text,
                    "style": result_style,
                    "source": source,
                    "processing_time_ms": processing_time,
                    "success": True
                }
            else:
                # If captioner indicated failure or no caption
                return JSONResponse(
                    status_code=500,
                    content={
                        "caption": caption_text or "Caption generation failed",
                        "style": result_style,
                        "source": source,
                        "processing_time_ms": processing_time,
                        "success": False,
                        "error": error or "Unknown error"
                    }
                )

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")

        # Return fallback response
        processing_time = (time.time() - start_time) * 1000
        return JSONResponse(
            status_code=500,
            content={
                "caption": "Caption generation failed",
                "style": style,
                "source": "error",
                "processing_time_ms": processing_time,
                "success": False,
                "error": str(e)
            }
        )


@router.get("/health")
async def caption_health():
    """
    Check caption service health
    """
    try:
        captioner = get_captioner()
        return {
            "status": "healthy",
            "model_type": captioner.model_type,
            "fallback_available": captioner.fallback_to_public
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.get("/models")
async def get_available_models():
    """
    Get information about available caption models
    """
    try:
        captioner = get_captioner()
        return {
            "primary_model": "vintern-1b-v3.5",
            "fallback_models": ["blip-image-captioning-base", "git-base-coco"],
            "current_model_type": captioner.model_type,
            "supported_styles": ["dense", "short", "tags", "ocr"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
