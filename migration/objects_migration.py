"""
Object Detection Migration Script

This script migrates object detection data from JSON files to MongoDB keyframe documents.
It reads object detection results from resources/objects/{group}/{video}/{frame}.json
and updates the corresponding keyframe documents with the detected objects.
"""

import os
import sys

# Add the app directory to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from typing import Dict, List, Set
from pathlib import Path
import time
import argparse
import asyncio
import json
from app.core.settings import MongoDBSettings
from beanie import init_beanie
import motor.motor_asyncio
from app.models.keyframe import Keyframe


# Add the app directory to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


mongo_settings = MongoDBSettings()


class ObjectMigrator:
    """Handles object detection data migration to MongoDB"""

    def __init__(self, objects_root: Path, dry_run: bool = False):
        self.objects_root = objects_root
        self.dry_run = dry_run
        self.all_unique_objects = set()  # Store all unique objects found
        self.stats = {
            'processed_files': 0,
            'updated_keyframes': 0,
            'failed_files': 0,
            'missing_keyframes': 0,
            'total_objects': 0,
            'unique_objects_count': 0
        }

    def normalize_object_name(self, detection_class_entity: str) -> str:
        """Normalize object detection class entity to standard format"""
        if not detection_class_entity:
            return ""
        return detection_class_entity.lower().strip()

    def extract_objects_from_detection(self, detection_data: Dict) -> List[str]:
        """Extract unique normalized object names from detection JSON"""
        try:
            entities = detection_data.get('detection_class_entities', [])
            if not entities:
                return []

            # Normalize and deduplicate objects
            normalized_objects = set()
            for entity in entities:
                normalized = self.normalize_object_name(entity)
                if normalized:
                    normalized_objects.add(normalized)

            return sorted(list(normalized_objects))  # Sort for consistency
        except Exception as e:
            print(f"❌ Error extracting objects: {e}")
            return []

    def parse_object_path(self, object_file: Path) -> tuple[int, int, int] | None:
        """Parse object file path to extract group_num, video_num, keyframe_num"""
        try:
            # Path structure: resources/objects/{group}/{video}/{frame}.json
            # Example: resources/objects/L21_V001/001.json

            parts = object_file.parts
            if len(parts) < 2:
                return None

            # Extract video directory name (e.g., "L21_V001")
            video_dir = parts[-2]
            frame_file = parts[-1]

            # Parse video directory: L{group}_V{video}
            if not video_dir.startswith('L') or '_V' not in video_dir:
                return None

            group_part, video_part = video_dir.split('_V')
            group_num = int(group_part[1:])  # Remove 'L' prefix
            video_num = int(video_part)

            # Parse frame number from filename
            keyframe_num = int(frame_file.replace('.json', ''))

            return group_num, video_num, keyframe_num

        except (ValueError, IndexError) as e:
            print(f"⚠️  Invalid object file path format: {object_file} - {e}")
            return None

    async def process_object_file(self, object_file: Path) -> bool:
        """Process a single object detection JSON file"""
        try:
            # Parse file path to get keyframe coordinates
            parsed = self.parse_object_path(object_file)
            if not parsed:
                return False

            group_num, video_num, keyframe_num = parsed

            # Load object detection data
            with open(object_file, 'r', encoding='utf-8') as f:
                detection_data = json.load(f)

            # Extract and normalize objects
            objects = self.extract_objects_from_detection(detection_data)

            if not objects:
                print(f"⚠️  No objects found in {object_file}")
                return True  # Not an error, just empty detection

            # Add objects to the global unique set
            self.all_unique_objects.update(objects)

            # Find matching keyframe in database
            keyframe = await Keyframe.find_one(
                Keyframe.group_num == group_num,
                Keyframe.video_num == video_num,
                Keyframe.keyframe_num == keyframe_num
            )

            if not keyframe:
                print(
                    f"❌ No keyframe found for L{group_num:02d}_V{video_num:03d}_{keyframe_num:03d}")
                self.stats['missing_keyframes'] += 1
                return False

            # Update keyframe with objects (if not dry run)
            if not self.dry_run:
                await keyframe.update({"$set": {"objects": objects}})
                print(f"✅ Updated L{group_num:02d}_V{video_num:03d}_{keyframe_num:03d} with {len(objects)} objects: {', '.join(objects[:5])}" + (
                    f" (+{len(objects)-5} more)" if len(objects) > 5 else ""))
            else:
                print(
                    f"🔍 [DRY RUN] Would update L{group_num:02d}_V{video_num:03d}_{keyframe_num:03d} with {len(objects)} objects")

            self.stats['updated_keyframes'] += 1
            self.stats['total_objects'] += len(objects)
            return True

        except Exception as e:
            print(f"❌ Error processing {object_file}: {e}")
            self.stats['failed_files'] += 1
            return False

    def save_unique_objects_list(self, output_file: Path = None):
        """Save all unique objects found during migration to a JSON file"""
        if output_file is None:
            output_file = self.objects_root / "all_objects_found.json"

        try:
            # Convert set to sorted list for consistent output
            unique_objects_list = sorted(list(self.all_unique_objects))

            # Create comprehensive object data
            objects_data = {
                "metadata": {
                    "total_unique_objects": len(unique_objects_list),
                    "migration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_directory": str(self.objects_root),
                    "description": "All unique object names found during object detection migration"
                },
                "objects": unique_objects_list,
                "objects_by_category": self.categorize_objects(unique_objects_list)
            }

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(objects_data, f, indent=2, ensure_ascii=False)

            print(
                f"💾 Saved {len(unique_objects_list)} unique objects to: {output_file}")
            return True

        except Exception as e:
            print(f"❌ Error saving unique objects list: {e}")
            return False

    def categorize_objects(self, objects_list: List[str]) -> Dict[str, List[str]]:
        """Categorize objects into logical groups for better organization"""
        categories = {
            "vehicles": [],
            "people_and_animals": [],
            "buildings_and_structures": [],
            "furniture_and_household": [],
            "nature_and_outdoor": [],
            "technology_and_electronics": [],
            "food_and_drink": [],
            "sports_and_recreation": [],
            "other": []
        }

        # Define category keywords
        category_keywords = {
            "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat", "train", "vehicle"],
            "people_and_animals": ["person", "people", "human", "man", "woman", "child", "dog", "cat", "bird", "horse", "cow", "animal"],
            "buildings_and_structures": ["building", "house", "skyscraper", "tower", "bridge", "fence", "wall", "door", "window"],
            "furniture_and_household": ["chair", "table", "bed", "couch", "toilet", "sink", "refrigerator", "microwave", "oven"],
            "nature_and_outdoor": ["tree", "flower", "mountain", "water", "sky", "cloud", "potted plant"],
            "technology_and_electronics": ["tv", "laptop", "phone", "computer", "remote", "keyboard", "mouse"],
            "food_and_drink": ["bottle", "cup", "bowl", "food", "drink", "banana", "apple", "pizza"],
            "sports_and_recreation": ["ball", "racket", "skateboard", "surfboard", "skis", "kite"]
        }

        for obj in objects_list:
            categorized = False
            obj_lower = obj.lower()

            # Try to categorize based on keywords
            for category, keywords in category_keywords.items():
                if any(keyword in obj_lower for keyword in keywords):
                    categories[category].append(obj)
                    categorized = True
                    break

            # If not categorized, put in "other"
            if not categorized:
                categories["other"].append(obj)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    async def migrate_objects(self, batch_size: int = 100):
        """Migrate all object detection files to MongoDB"""

        print(
            f"🚀 Starting object detection migration from: {self.objects_root}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🔍 Dry run: {self.dry_run}")

        if not self.objects_root.exists():
            print(f"❌ Objects directory not found: {self.objects_root}")
            return

        # Collect all JSON files
        json_files = list(self.objects_root.rglob("*.json"))
        print(f"📁 Found {len(json_files)} object detection files")

        if not json_files:
            print("⚠️  No JSON files found in objects directory")
            return

        # Process files in batches
        for i in range(0, len(json_files), batch_size):
            batch_files = json_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(json_files) + batch_size - 1) // batch_size

            print(
                f"\n--- Processing Batch {batch_num}/{total_batches} ({len(batch_files)} files) ---")

            # Process batch concurrently (but limit concurrency)
            # Limit to 10 concurrent operations
            semaphore = asyncio.Semaphore(10)

            async def process_with_semaphore(file_path):
                async with semaphore:
                    return await self.process_object_file(file_path)

            # Process files in current batch
            tasks = [process_with_semaphore(file_path)
                     for file_path in batch_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful operations in this batch
            batch_success = sum(1 for r in results if r is True)
            self.stats['processed_files'] += len(batch_files)

            print(
                f"✅ Batch {batch_num} completed: {batch_success}/{len(batch_files)} successful")

            # Small delay between batches to reduce database load
            if batch_num < total_batches:
                await asyncio.sleep(0.1)

        # Print final statistics
        print(f"\n🎉 Object migration completed!")
        print(f"📊 Statistics:")
        print(f"   - Processed files: {self.stats['processed_files']}")
        print(f"   - Updated keyframes: {self.stats['updated_keyframes']}")
        print(f"   - Total objects migrated: {self.stats['total_objects']}")
        print(f"   - Failed files: {self.stats['failed_files']}")
        print(f"   - Missing keyframes: {self.stats['missing_keyframes']}")

        # Update stats with unique objects count
        self.stats['unique_objects_count'] = len(self.all_unique_objects)
        print(
            f"   - Unique objects found: {self.stats['unique_objects_count']}")

        if self.dry_run:
            print("🔍 This was a dry run - no actual changes were made to the database")

        # Display some sample objects found
        if self.all_unique_objects:
            sample_objects = sorted(list(self.all_unique_objects))[:10]
            print(f"\n📋 Sample objects found: {', '.join(sample_objects)}")
            if len(self.all_unique_objects) > 10:
                print(
                    f"   ... and {len(self.all_unique_objects) - 10} more objects")


async def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description="Migrate object detection data from JSON files to MongoDB"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default=str(Path(ROOT_DIR) / "resources" / "objects"),
        help="Path to the objects directory containing detection JSON files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of files to process in each batch (default: 100)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run migration in dry-run mode (no actual database changes)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file path for unique objects list (default: {objects_folder}/all_objects_found.json)"
    )

    args = parser.parse_args()

    # Initialize database connection
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_settings.MONGO_URI)
    database = client[mongo_settings.MONGO_DB]

    # Initialize Beanie
    await init_beanie(database=database, document_models=[Keyframe])

    # Create migrator and run migration
    migrator = ObjectMigrator(Path(args.folder_path), dry_run=args.dry_run)
    await migrator.migrate_objects(batch_size=args.batch_size)

    # Save unique objects list to specified output file
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path(args.folder_path) / "all_objects_found.json"

    migrator.save_unique_objects_list(output_path)

    # Close database connection
    client.close()


if __name__ == "__main__":
    asyncio.run(main())
