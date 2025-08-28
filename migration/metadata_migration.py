import asyncio
import json
import os
import sys
import argparse
from pathlib import Path

# Add the app directory to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app"))
sys.path.insert(0, ROOT_DIR)

from models.keyframe import Keyframe
import motor.motor_asyncio
from beanie import init_beanie
from core.settings import MongoDBSettings

mongo_settings = MongoDBSettings()


async def load_metadata_to_keyframes(metadata_folder_path: str):
    """Load metadata from JSON files into keyframe documents"""
    
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
    
    # Get all JSON files
    json_files = list(metadata_dir.glob("*.json"))
    print(f"Found {len(json_files)} metadata files")
    
    updated_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            # Extract video information from filename (e.g., L21_V001.json)
            filename = json_file.stem  # L21_V001
            parts = filename.split('_')
            
            if len(parts) != 2:
                print(f"Skipping file with unexpected format: {filename}")
                continue
                
            group_part = parts[0][1:]  # Remove 'L' prefix
            video_part = parts[1][1:]  # Remove 'V' prefix
            
            group_num = int(group_part)
            video_num = int(video_part)
            
            # Load metadata from JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Update all keyframes for this video
            result = await Keyframe.find(
                Keyframe.group_num == group_num,
                Keyframe.video_num == video_num
            ).update_many({
                "$set": {
                    "author": metadata.get("author"),
                    "channel_id": metadata.get("channel_id"),
                    "title": metadata.get("title"),
                    "description": metadata.get("description"),
                    "keywords": metadata.get("keywords", []),
                    "length": metadata.get("length"),
                    "publish_date": metadata.get("publish_date"),
                    "thumbnail_url": metadata.get("thumbnail_url"),
                    "watch_url": metadata.get("watch_url")
                }
            })
            
            if result.modified_count > 0:
                print(f"Updated {result.modified_count} keyframes for {filename}")
                updated_count += result.modified_count
            else:
                print(f"No keyframes found for {filename} (Group: {group_num}, Video: {video_num})")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            error_count += 1
            continue
    
    print(f"\nMigration completed:")
    print(f"- Updated keyframes: {updated_count}")
    print(f"- Errors: {error_count}")
    
    # Close the database connection
    client.close()


def main():
    """Main function to run the metadata migration"""
    parser = argparse.ArgumentParser(description="Migrate metadata from JSON files to MongoDB")
    parser.add_argument(
        "--folder_path", 
        type=str, 
        required=True,
        help="Path to the folder containing metadata JSON files"
    )
    
    args = parser.parse_args()
    
    print(f"Starting metadata migration from: {args.folder_path}")
    asyncio.run(load_metadata_to_keyframes(args.folder_path))
    print("Metadata migration finished!")


if __name__ == "__main__":
    main()
