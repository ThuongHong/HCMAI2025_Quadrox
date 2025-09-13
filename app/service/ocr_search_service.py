"""
OCR Search Service
Provides OCR text search functionality using the ocr2_search.db database.
"""
import sqlite3
import json
import os
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
import asyncio

# Simple logging without core dependency
import logging
logger = logging.getLogger(__name__)


class OCRSearchResult(BaseModel):
    """OCR search result model"""
    frame_id: str
    video_id: str
    frame_number: str
    ocr_text: str
    confidence: float
    ocr_path: str
    frame_path: str
    # Video metadata (optional)
    watch_url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    publish_date: Optional[str] = None
    length: Optional[int] = None


class OCRSearchService:
    """Service for searching OCR text in videos"""
    
    def __init__(self, db_path: str = None):
        """Initialize OCR search service
        
        Args:
            db_path: Path to ocr2_search.db file
        """
        if db_path is None:
            # Default to resources folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            db_path = os.path.join(project_root, "resources", "ocr2_search.db")
        
        self.db_path = db_path
        self._validate_database()
    
    async def _get_video_metadata_simple(self, video_id: str) -> Optional[Dict]:
        """Get video metadata directly from keyframe using same logic as text search
        
        Args:
            video_id: Video ID in format L01_V001
            
        Returns:
            Dictionary with video metadata or None if not found
        """
        try:
            # Parse video_id: L01_V001 -> group_num=1, video_num=1
            parts = video_id.split('_')
            if len(parts) != 2:
                return None
                
            group_str = parts[0]  # L01
            video_str = parts[1]  # V001
            
            if not (group_str.startswith('L') and video_str.startswith('V')):
                return None
                
            group_num = int(group_str[1:])  # 1
            video_num = int(video_str[1:])  # 1
            
            # Use exact same approach as text search - import and use Beanie model
            from app.models.keyframe import Keyframe
            
            # Find keyframe with matching group_num and video_num (same as text search)
            keyframe = await Keyframe.find_one({
                "group_num": group_num,
                "video_num": video_num
            })
            
            if keyframe:
                # Return exact same fields as text search query_controller.py line 133-147
                return {
                    "watch_url": keyframe.watch_url,
                    "title": keyframe.title, 
                    "author": keyframe.author,
                    "description": keyframe.description,
                    "thumbnail_url": keyframe.thumbnail_url,
                    "publish_date": keyframe.publish_date,
                    "length": keyframe.length,
                    "channel_id": keyframe.channel_id,
                    "keywords": keyframe.keywords
                }
            else:
                logger.warning(f"No keyframe found for group_num={group_num}, video_num={video_num}")
                
        except Exception as e:
            logger.error(f"Failed to get video metadata for {video_id}: {e}")
        
        return None
    
    def _validate_database(self):
        """Validate that the database exists and has the expected structure"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"OCR database not found at: {self.db_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ocr_index'")
                if not cursor.fetchone():
                    raise ValueError("ocr_index table not found in database")
                logger.info(f"OCR database validated: {self.db_path}")
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            raise
    
    async def search_text_with_metadata(
        self, 
        query: str,
        limit: int = 50,
        min_confidence: float = 0.0,
        video_filters: Optional[List[str]] = None,
        include_video_metadata: bool = True
    ) -> List[OCRSearchResult]:
        """Search for text in OCR results and enrich with video metadata
        
        Args:
            query: Search text query
            limit: Maximum number of results to return
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            video_filters: Optional list of video IDs to filter by (e.g., ['L01_V001', 'L02_V005'])
            include_video_metadata: Whether to include video metadata (requires MongoDB)
        
        Returns:
            List of OCRSearchResult objects with optional video metadata
        """
        logger.info(f"OCR search_text_with_metadata called: include_video_metadata={include_video_metadata}")
        
        # Get basic search results
        results = self.search_text(query, limit, min_confidence, video_filters)
        
        if not include_video_metadata:
            logger.info("Skipping video metadata enrichment")
            return results
        
        # Enrich with video metadata
        enriched_results = []
        seen_videos = {}  # Cache metadata per video
        
        logger.info(f"Enriching {len(results)} OCR results with video metadata")
        
        for result in results:
            # Check if we already have metadata for this video
            if result.video_id not in seen_videos:
                logger.info(f"Getting metadata for video_id: {result.video_id}")
                metadata = await self._get_video_metadata_simple(result.video_id)
                seen_videos[result.video_id] = metadata
                logger.info(f"Metadata for {result.video_id}: {metadata}")
            else:
                metadata = seen_videos[result.video_id]
            
            # Create enriched result
            if metadata:
                result.watch_url = metadata.get("watch_url")
                result.title = metadata.get("title")
                result.author = metadata.get("author")
                result.description = metadata.get("description")
                result.thumbnail_url = metadata.get("thumbnail_url")
                result.publish_date = metadata.get("publish_date")
                result.length = metadata.get("length")
            
            enriched_results.append(result)
        
        logger.info(f"Enriched {len(enriched_results)} OCR results with video metadata")
        return enriched_results

    def search_text(
        self, 
        query: str,
        limit: int = 50,
        min_confidence: float = 0.0,
        video_filters: Optional[List[str]] = None
    ) -> List[OCRSearchResult]:
        """Search for text in OCR results using FTS5
        
        Args:
            query: Search text query
            limit: Maximum number of results to return
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            video_filters: Optional list of video IDs to filter by (e.g., ['L01_V001', 'L02_V005'])
        
        Returns:
            List of OCRSearchResult objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build the query
                base_query = """
                SELECT frame_id, video_id, frame_number, ocr_text, avg_conf, ocr_path, frame_path
                FROM ocr_index 
                WHERE ocr_index MATCH ? 
                AND avg_conf >= ?
                """
                
                params = [query, min_confidence]
                
                # Add video filters if provided
                if video_filters:
                    placeholders = ','.join(['?' for _ in video_filters])
                    base_query += f" AND video_id IN ({placeholders})"
                    params.extend(video_filters)
                
                base_query += " ORDER BY avg_conf DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(base_query, params)
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    result = OCRSearchResult(
                        frame_id=row[0],
                        video_id=row[1],
                        frame_number=row[2],
                        ocr_text=row[3],
                        confidence=float(row[4]),
                        ocr_path=row[5],
                        frame_path=row[6]
                    )
                    results.append(result)
                
                logger.info(f"OCR search returned {len(results)} results for query: '{query}'")
                return results
                
        except Exception as e:
            logger.error(f"OCR search failed: {e}")
            raise
    
    def get_video_statistics(self) -> Dict[str, int]:
        """Get statistics about videos in the database
        
        Returns:
            Dictionary with video statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total records
                cursor.execute("SELECT COUNT(*) FROM ocr_index")
                total_records = cursor.fetchone()[0]
                
                # Unique videos
                cursor.execute("SELECT COUNT(DISTINCT video_id) FROM ocr_index")
                unique_videos = cursor.fetchone()[0]
                
                # Average confidence
                cursor.execute("SELECT AVG(avg_conf) FROM ocr_index WHERE avg_conf > 0")
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                # Records with text (non-empty ocr_text)
                cursor.execute("SELECT COUNT(*) FROM ocr_index WHERE ocr_text != ''")
                records_with_text = cursor.fetchone()[0]
                
                return {
                    "total_records": total_records,
                    "unique_videos": unique_videos,
                    "average_confidence": round(float(avg_confidence), 3),
                    "records_with_text": records_with_text,
                    "records_without_text": total_records - records_with_text
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
    
    def search_by_video(self, video_id: str, limit: int = 100) -> List[OCRSearchResult]:
        """Get all OCR results for a specific video
        
        Args:
            video_id: Video ID (e.g., 'L01_V001')
            limit: Maximum number of results
        
        Returns:
            List of OCRSearchResult objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT frame_id, video_id, frame_number, ocr_text, avg_conf, ocr_path, frame_path
                FROM ocr_index 
                WHERE video_id = ?
                ORDER BY CAST(frame_number AS INTEGER) ASC
                LIMIT ?
                """
                
                cursor.execute(query, [video_id, limit])
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    result = OCRSearchResult(
                        frame_id=row[0],
                        video_id=row[1],
                        frame_number=row[2],
                        ocr_text=row[3],
                        confidence=float(row[4]),
                        ocr_path=row[5],
                        frame_path=row[6]
                    )
                    results.append(result)
                
                logger.info(f"Found {len(results)} OCR results for video: {video_id}")
                return results
                
        except Exception as e:
            logger.error(f"Failed to get OCR results for video {video_id}: {e}")
            raise
    
    def get_ocr_detail(self, ocr_path: str) -> Optional[List[Dict]]:
        """Get detailed OCR information from JSON file
        
        Args:
            ocr_path: Path to OCR JSON file (e.g., 'ocr\\L01_V001\\001.json')
        
        Returns:
            List of OCR detection objects or None if file not found
        """
        try:
            # Build full path relative to resources folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            full_ocr_path = os.path.join(project_root, "resources", ocr_path.replace("\\", os.sep))
            
            if not os.path.exists(full_ocr_path):
                logger.warning(f"OCR file not found: {full_ocr_path}")
                return None
            
            with open(full_ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            return ocr_data
            
        except Exception as e:
            logger.error(f"Failed to read OCR file {ocr_path}: {e}")
            return None
    
    def search_with_context(
        self, 
        query: str,
        limit: int = 50,
        min_confidence: float = 0.0,
        video_filters: Optional[List[str]] = None,
        include_ocr_details: bool = False
    ) -> List[Dict]:
        """Search with additional context information
        
        Args:
            query: Search text query
            limit: Maximum number of results
            min_confidence: Minimum confidence threshold
            video_filters: Optional video filters
            include_ocr_details: Whether to include full OCR detection details
        
        Returns:
            List of result dictionaries with additional context
        """
        results = self.search_text(query, limit, min_confidence, video_filters)
        
        enhanced_results = []
        for result in results:
            result_dict = result.dict()
            
            # Add OCR details if requested
            if include_ocr_details:
                ocr_details = self.get_ocr_detail(result.ocr_path)
                result_dict["ocr_details"] = ocr_details
            
            # Parse video information
            video_parts = result.video_id.split('_')
            if len(video_parts) == 2:
                result_dict["group_number"] = video_parts[0]  # L01, L02, etc.
                result_dict["video_number"] = video_parts[1]  # V001, V002, etc.
            
            enhanced_results.append(result_dict)
        
        return enhanced_results