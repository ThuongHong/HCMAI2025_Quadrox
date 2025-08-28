import os
import sys

# Add the app directory to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from app.core.settings import MongoDBSettings
from beanie import init_beanie
import motor.motor_asyncio
from app.models.keyframe import Keyframe
import asyncio
import json
import argparse
from pathlib import Path



mongo_settings = MongoDBSettings()


async def load_metadata_to_keyframes(metadata_folder_path: str, batch_size: int = 10, skip_existing: bool = True):
    """Load metadata from JSON files into keyframe documents with optimization"""

    # Initialize database connection
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_settings.MONGO_URI)
    database = client[mongo_settings.MONGO_DB]

    # Initialize Beanie
    await init_beanie(database=database, document_models=[Keyframe])

    # Load metadata files
    metadata_dir = Path(metadata_folder_path)

    if not metadata_dir.exists():
        print(f"Metadata directory not found: {metadata_dir}")
        return

    # Get all JSON files sorted for consistent processing
    json_files = sorted(list(metadata_dir.glob("*.json")))
    print(f"Found {len(json_files)} metadata files")

    updated_count = 0
    error_count = 0
    skipped_count = 0

    # Process files in batches to manage memory and quota
    for i in range(0, len(json_files), batch_size):
        batch_files = json_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(json_files) + batch_size - 1) // batch_size

        print(
            f"\n--- Processing Batch {batch_num}/{total_batches} ({len(batch_files)} files) ---")

        for json_file in batch_files:
            try:
                # Extract video information from filename (e.g., L21_V001.json)
                filename = json_file.stem  # L21_V001
                parts = filename.split('_')

                if len(parts) != 2:
                    print(
                        f"⚠️  Skipping file with unexpected format: {filename}")
                    skipped_count += 1
                    continue

                try:
                    group_part = parts[0][1:]  # Remove 'L' prefix
                    video_part = parts[1][1:]  # Remove 'V' prefix

                    group_num = int(group_part)
                    video_num = int(video_part)
                except ValueError:
                    print(f"❌ Invalid number format in filename: {filename}")
                    error_count += 1
                    continue

                # Check if keyframes exist first
                keyframe_count = await Keyframe.find(
                    Keyframe.group_num == group_num,
                    Keyframe.video_num == video_num
                ).count()

                if keyframe_count == 0:
                    print(
                        f"⚠️  No keyframes found for {filename} (Group: {group_num}, Video: {video_num})")
                    skipped_count += 1
                    continue

                # Check if metadata already exists (if skip_existing is True)
                if skip_existing:
                    existing_with_metadata = await Keyframe.find_one(
                        Keyframe.group_num == group_num,
                        Keyframe.video_num == video_num,
                        Keyframe.title != None
                    )
                    if existing_with_metadata:
                        print(
                            f"⏭️  Skipping {filename} - already has metadata")
                        skipped_count += 1
                        continue

                # Load metadata from JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Optimize metadata to reduce storage usage and prevent quota errors
                optimized_metadata = {}

                # Only include non-null values and limit sizes
                if metadata.get("author"):
                    author = str(metadata["author"])
                    optimized_metadata["author"] = author[:100] if len(
                        author) > 100 else author

                if metadata.get("channel_id"):
                    optimized_metadata["channel_id"] = str(
                        metadata["channel_id"])

                if metadata.get("title"):
                    title = str(metadata["title"])
                    optimized_metadata["title"] = title[:200] if len(
                        title) > 200 else title

                # Significantly limit description to save space
                if metadata.get("description"):
                    description = str(metadata["description"])
                    optimized_metadata["description"] = description[:300] + \
                        "..." if len(description) > 300 else description

                # Limit keywords to reduce storage
                if metadata.get("keywords") and isinstance(metadata["keywords"], list):
                    # Only first 5 keywords
                    keywords = metadata["keywords"][:5]
                    optimized_metadata["keywords"] = keywords

                if metadata.get("length"):
                    optimized_metadata["length"] = metadata["length"]

                if metadata.get("publish_date"):
                    optimized_metadata["publish_date"] = str(
                        metadata["publish_date"])

                if metadata.get("thumbnail_url"):
                    optimized_metadata["thumbnail_url"] = str(
                        metadata["thumbnail_url"])

                if metadata.get("watch_url"):
                    optimized_metadata["watch_url"] = str(
                        metadata["watch_url"])

                # Skip if no valid metadata found
                if not optimized_metadata:
                    print(f"⚠️  No valid metadata found in {filename}")
                    skipped_count += 1
                    continue

                # Update all keyframes for this video with optimized metadata
                result = await Keyframe.find(
                    Keyframe.group_num == group_num,
                    Keyframe.video_num == video_num
                ).update_many({"$set": optimized_metadata})

                if result.modified_count > 0:
                    print(
                        f"✅ Updated {result.modified_count} keyframes for {filename}")
                    updated_count += result.modified_count
                else:
                    print(f"⚠️  No updates made for {filename}")

                # Small delay to reduce database load
                await asyncio.sleep(0.05)

            except Exception as e:
                print(f"❌ Error processing {json_file.name}: {e}")
                error_count += 1
                continue

        # Pause between batches to manage memory and reduce quota pressure
        if batch_num < total_batches:
            print(f"⏳ Waiting 1 second before next batch...")
            await asyncio.sleep(1)

    print(f"\nMigration completed:")
    print(f"- Updated keyframes: {updated_count}")
    print(f"- Skipped files: {skipped_count}")
    print(f"- Errors: {error_count}")
    print(f"- Total files processed: {len(json_files)}")

    # Close the database connection
    client.close()


def main():
    """Main function to run the metadata migration"""
    parser = argparse.ArgumentParser(
        description="Migrate metadata from JSON files to MongoDB (optimized)")
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing metadata JSON files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of files to process in each batch (default: 5)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip files that already have metadata (default: True)"
    )

    args = parser.parse_args()

    print(f"🚀 Starting optimized metadata migration from: {args.folder_path}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"⏭️  Skip existing: {args.skip_existing}")

    asyncio.run(load_metadata_to_keyframes(
        args.folder_path,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing
    ))
    print("✨ Metadata migration finished!")


if __name__ == "__main__":
    main()
