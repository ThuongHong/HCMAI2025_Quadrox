#!/usr/bin/env python3
"""Precompute Vietnamese captions for keyframe images using VinternCaptionerCPU."""

from app.retrieval.rerank.vintern_captioner import VinternCaptionerCPU
import argparse
import logging
import sys
import time
from pathlib import Path
from glob import glob
from typing import List

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for caption precomputation."""
    parser = argparse.ArgumentParser(
        description="Precompute Vietnamese captions for keyframe images"
    )
    parser.add_argument(
        "--images_glob",
        type=str,
        default="resources/keyframes/**/*.jpg",
        help="Glob pattern for image files"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="dense",
        choices=["dense", "short", "tags", "ocr"],
        help="Caption style to use"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/Vintern-1B-v3_5",
        help="Path to Vintern model directory"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens for caption generation"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./resources/captions_cache",
        help="Directory for caption cache"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Maximum worker threads for parallel processing"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for processing images"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )

    args = parser.parse_args()

    logger.info(f"Starting caption precomputation with:")
    logger.info(f"  Images glob: {args.images_glob}")
    logger.info(f"  Style: {args.style}")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Max new tokens: {args.max_new_tokens}")
    logger.info(f"  Cache dir: {args.cache_dir}")
    logger.info(f"  Max workers: {args.max_workers}")
    logger.info(f"  Batch size: {args.batch_size}")

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        logger.info("Please download the model first:")
        logger.info(
            f"  huggingface-cli download 5CD-AI/Vintern-1B-v3_5 --local-dir {model_path}")
        return 1

    # Find image files
    logger.info(f"Searching for images with pattern: {args.images_glob}")
    image_paths = glob(args.images_glob, recursive=True)

    if not image_paths:
        logger.error(f"No images found with pattern: {args.images_glob}")
        return 1

    # Filter existing images
    existing_paths = [p for p in image_paths if Path(p).exists()]
    logger.info(f"Found {len(existing_paths)} existing image files")

    if args.max_images:
        existing_paths = existing_paths[:args.max_images]
        logger.info(f"Limited to {len(existing_paths)} images for processing")

    # Initialize captioner
    try:
        captioner = VinternCaptionerCPU(
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers
        )
        logger.info("VinternCaptionerCPU initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize captioner: {e}")
        return 1

    # Process images in batches
    total_processed = 0
    total_success = 0
    total_errors = 0
    start_time = time.time()

    try:
        for i in range(0, len(existing_paths), args.batch_size):
            batch_paths = existing_paths[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(existing_paths) +
                             args.batch_size - 1) // args.batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)")

            # Process batch
            batch_start = time.time()
            results = captioner.batch_caption(
                image_paths=batch_paths,
                style=args.style,
                max_new_tokens=args.max_new_tokens
            )
            batch_elapsed = time.time() - batch_start

            # Count results
            batch_success = len(
                [r for r in results.values() if "caption" in r])
            batch_errors = len(results) - batch_success

            total_processed += len(batch_paths)
            total_success += batch_success
            total_errors += batch_errors

            logger.info(f"Batch {batch_num} completed in {batch_elapsed:.2f}s: "
                        f"{batch_success} success, {batch_errors} errors")

            # Log some sample results
            if results:
                sample_path = next(iter(results.keys()))
                sample_result = results[sample_path]
                if "caption" in sample_result:
                    logger.info(
                        f"Sample caption: {sample_result['caption'][:100]}...")
                elif "error" in sample_result:
                    logger.warning(f"Sample error: {sample_result['error']}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    # Final statistics
    total_elapsed = time.time() - start_time
    success_rate = (total_success / total_processed *
                    100) if total_processed > 0 else 0

    logger.info("=" * 60)
    logger.info("CAPTION PRECOMPUTATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {total_processed}")
    logger.info(f"Successful captions: {total_success}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Total time: {total_elapsed:.2f}s")
    logger.info(
        f"Average time per image: {total_elapsed/total_processed:.2f}s" if total_processed > 0 else "N/A")
    logger.info(f"Cache directory: {args.cache_dir}")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
