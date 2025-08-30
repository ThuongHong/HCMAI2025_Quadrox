"""Vietnamese Caption Generation using Vintern-1B-v3.5 (CPU-only)."""

import json
import logging
import hashlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)


class VinternCaptionerCPU:
    """
    CPU-only Vietnamese image captioning using Vintern-1B-v3.5.
    Optimized for top-K reranking (10-100 images) with caching.
    Falls back to public models if Vintern is not available.
    """

    # Prompt templates for different styles
    PROMPTS = {
        "dense": "<image>\nMô tả ngắn gọn, chính xác nội dung chính (đối tượng, hành động, bối cảnh, chữ nếu có).",
        "short": "<image>\nMô tả ngắn gọn hình ảnh này.",
        "tags": "<image>\nLiệt kê các từ khóa mô tả hình ảnh:",
        "ocr": "<image>\nNhận dạng và đọc văn bản trong hình ảnh:"
    }

    # Fallback models if Vintern is not available
    FALLBACK_MODELS = [
        "Salesforce/blip-image-captioning-base",
        "microsoft/git-base-coco",
        "nlpconnect/vit-gpt2-image-captioning"
    ]

    def __init__(
        self,
        model_path: str = "./models/Vintern-1B-v3_5",
        cache_dir: str = "./resources/captions_cache",
        max_workers: int = 2,
        fallback_to_public: bool = True
    ):
        """
        Initialize VinternCaptionerCPU.

        Args:
            model_path: Path to Vintern model directory
            cache_dir: Directory for caption cache
            max_workers: Max workers for ThreadPoolExecutor
            fallback_to_public: Use public models if Vintern unavailable
        """
        # Normalize path for cross-platform compatibility
        from pathlib import Path
        norm_path = str(model_path).replace("\\", "/")
        self.model_path = Path(norm_path).expanduser().resolve()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.fallback_to_public = fallback_to_public

        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cpu"
        self.model_type = "vintern"  # or "blip", "git", etc.

        # Determine optimal dtype for CPU
        supports_bf16 = torch.cpu.is_bf16_supported() if hasattr(
            torch.cpu, "is_bf16_supported") else False
        self.dtype = torch.bfloat16 if supports_bf16 else torch.float32

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        logger.info(f"VinternCaptionerCPU initialized: model_path={self.model_path}, "
                    f"cache_dir={cache_dir}, max_workers={max_workers}, dtype={self.dtype}, "
                    f"fallback_enabled={fallback_to_public}")

        # Check model path existence early
        if not self.model_path.exists():
            msg = f"Vintern model not found at {self.model_path}"
            if fallback_to_public:
                logger.warning(f"{msg}, will use fallback models")
                self.model_type = "fallback"
            else:
                raise FileNotFoundError(msg)

    def _load_model(self) -> None:
        """Load model components if not already loaded."""
        if self.model is not None:
            return

        try:
            # Try to load Vintern first if path exists and not fallback type
            if self.model_type != "fallback" and self.model_path.exists():
                self._load_vintern_model()
            elif self.fallback_to_public:
                logger.warning(
                    f"Using fallback models (Vintern not available)")
                self._load_fallback_model()
            else:
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}")

        except Exception as e:
            if self.fallback_to_public and self.model_type == "vintern":
                logger.warning(f"Vintern model failed, trying fallback: {e}")
                self._load_fallback_model()
            else:
                logger.error(f"Failed to load any model: {e}")
                raise

    def _load_vintern_model(self) -> None:
        """Load Vintern model with CPU-only optimizations."""
        logger.info(f"Loading Vintern model from {self.model_path}")
        start_time = time.time()

        # Load tokenizer with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )

        # Load processor with optimizations
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )

        # Load model with CPU-only settings and corrected parameter name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=self.dtype,  # Changed from torch_dtype to dtype
            device_map=None,   # Force CPU-only
            low_cpu_mem_usage=True
        )

        # Ensure model is on CPU
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = "vintern"

        elapsed = time.time() - start_time
        logger.info(f"Vintern model loaded successfully in {elapsed:.2f}s")

    def _load_fallback_model(self) -> None:
        """Load fallback public model."""
        from transformers import BlipProcessor, BlipForConditionalGeneration

        model_name = "Salesforce/blip-image-captioning-base"
        logger.info(f"Loading fallback model: {model_name}")
        start_time = time.time()

        try:
            self.processor = BlipProcessor.from_pretrained(
                model_name, use_fast=True)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                dtype=self.dtype  # Changed from torch_dtype to dtype
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_type = "blip"

            elapsed = time.time() - start_time
            logger.info(
                f"Fallback BLIP model loaded successfully in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise

    def _preprocess_image(self, image_path: str) -> Union[Dict[str, torch.Tensor], Image.Image]:
        """
        Preprocess image based on model type.

        Args:
            image_path: Path to image file

        Returns:
            Dict with processed image tensors or PIL Image
        """
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')

            if self.model_type == "vintern":
                # Use processor to handle image preprocessing
                # This should handle resizing to 448x448 and normalization
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )

                # Move to CPU and convert dtype
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(
                            self.device, dtype=self.dtype)

                return inputs
            else:
                # For fallback models, return PIL Image
                return image

        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise

    def _generate_caption(
        self,
        image_data: Union[Dict[str, torch.Tensor], Image.Image],
        prompt: str,
        max_new_tokens: int = 64
    ) -> str:
        """
        Generate caption based on model type.

        Args:
            image_data: Preprocessed image tensors or PIL Image
            prompt: Text prompt for generation
            max_new_tokens: Max tokens to generate

        Returns:
            Generated caption text
        """
        try:
            if self.model_type == "vintern":
                return self._generate_caption_vintern(image_data, prompt, max_new_tokens)
            else:
                return self._generate_caption_blip(image_data, prompt, max_new_tokens)

        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            raise

    def _generate_caption_vintern(
        self,
        image_inputs: Dict[str, torch.Tensor],
        prompt: str,
        max_new_tokens: int = 64
    ) -> str:
        """Generate caption using Vintern model."""
        # Prepare text inputs
        text_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move text inputs to device
        for key in text_inputs:
            if torch.is_tensor(text_inputs[key]):
                text_inputs[key] = text_inputs[key].to(self.device)

        # Combine inputs
        model_inputs = {**image_inputs, **text_inputs}

        # Generate with conservative settings for CPU
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic
                num_beams=1,      # Fast single-beam search
                repetition_penalty=2.5,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Extract only the new generated text
        # Remove the original prompt
        if prompt in response:
            caption = response.replace(prompt, "").strip()
        else:
            caption = response.strip()

        return caption

    def _generate_caption_blip(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 64
    ) -> str:
        """Generate caption using BLIP model."""
        # Simple prompt adaptation for BLIP
        text_prompt = "a photo of" if "dense" in prompt else ""

        inputs = self.processor(
            image, text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.2
            )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()

    def _wrap_result(
        self,
        caption: str,
        style: str,
        max_new_tokens: int,
        image_path: str,
        start_time: float,
        success: bool = True,
        error: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """
        Wrap result in standardized schema to fix KeyError 'source'.

        Args:
            caption: Generated caption text
            style: Caption style used
            max_new_tokens: Max tokens parameter 
            image_path: Path to source image
            start_time: Generation start time
            success: Whether generation succeeded
            error: Error if failed

        Returns:
            Standardized result dict with all required fields
        """
        processing_time = time.time() - start_time

        return {
            "caption": (caption or "").strip(),
            "style": style,
            "max_new_tokens": max_new_tokens,
            "source": self.model_type or "unknown",  # Fix for KeyError 'source'
            "model": f"{self.model_type or 'unknown'}_cpu",
            "image_path": str(image_path),
            "generation_time": processing_time,
            "processing_time_ms": int(processing_time * 1000),
            "success": bool(success),
            "error": str(error) if error else None,
            "timestamp": time.time()
        }

    def _get_cache_key(self, image_path: str, style: str, max_new_tokens: int) -> str:
        """Generate cache key for image + generation params."""
        key_str = f"{image_path}|{style}|{max_new_tokens}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load caption from cache file."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Cache load failed for {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save caption to cache file."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Cache save failed for {cache_key}: {e}")

    def caption_image(
        self,
        image_path: str,
        style: str = "dense",
        max_new_tokens: int = 64
    ) -> Dict[str, Any]:
        """
        Generate caption for single image.

        Args:
            image_path: Path to image file
            style: Caption style (dense, short, tags, ocr)
            max_new_tokens: Max tokens to generate

        Returns:
            Dict with caption and metadata
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._get_cache_key(image_path, style, max_new_tokens)
            cached_result = self._load_from_cache(cache_key)

            if cached_result:
                logger.debug(f"Cache hit for {Path(image_path).name}")
                # Ensure cached result has all required fields
                if "source" not in cached_result:
                    cached_result["source"] = cached_result.get(
                        "model_type", "cached")
                if "success" not in cached_result:
                    cached_result["success"] = "error" not in cached_result
                return cached_result

            # Load model if needed
            self._load_model()

            # Get prompt
            prompt = self.PROMPTS.get(style, self.PROMPTS["dense"])

            # Preprocess image
            image_inputs = self._preprocess_image(image_path)

            # Generate caption
            caption = self._generate_caption(
                image_inputs, prompt, max_new_tokens)

            # Use standardized wrapper for result
            result = self._wrap_result(
                caption=caption,
                style=style,
                max_new_tokens=max_new_tokens,
                image_path=image_path,
                start_time=start_time,
                success=True
            )

            # Cache result
            self._save_to_cache(cache_key, result)

            logger.debug(
                f"Generated caption for {Path(image_path).name} in {result['generation_time']:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Caption generation failed for {image_path}: {e}")
            return self._wrap_result(
                caption="",
                style=style,
                max_new_tokens=max_new_tokens,
                image_path=image_path,
                start_time=start_time,
                success=False,
                error=e
            )

    def batch_caption(
        self,
        image_paths: List[str],
        style: str = "dense",
        max_new_tokens: int = 64
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate captions for multiple images in parallel.

        Args:
            image_paths: List of image file paths
            style: Caption style for all images
            max_new_tokens: Max tokens to generate

        Returns:
            Dict mapping image_path to caption result
        """
        if not image_paths:
            return {}

        logger.info(
            f"Starting batch caption for {len(image_paths)} images with {self.max_workers} workers")
        start_time = time.time()

        # Use ThreadPoolExecutor for parallel processing
        def caption_single(image_path: str) -> tuple[str, Dict[str, Any]]:
            result = self.caption_image(image_path, style, max_new_tokens)
            return image_path, result

        results = {}
        try:
            # Submit all tasks
            futures = []
            for image_path in image_paths:
                future = self.executor.submit(caption_single, image_path)
                futures.append(future)

            # Collect results
            for future in futures:
                try:
                    image_path, result = future.result(
                        timeout=120)  # 2 min timeout per image
                    results[image_path] = result
                except Exception as e:
                    logger.error(f"Batch caption task failed: {e}")

        except Exception as e:
            logger.error(f"Batch caption failed: {e}")

        elapsed = time.time() - start_time
        success_count = len([r for r in results.values() if "caption" in r])
        error_count = len(results) - success_count

        logger.info(f"Batch caption completed in {elapsed:.2f}s: "
                    f"{success_count} success, {error_count} errors")

        return results

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass
