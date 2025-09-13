"""
Local OCR Search for GUI
Path-aware OCR search service that works from gui/ folder
"""
import sqlite3
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class GuiOCRSearch:
    """OCR Search service for GUI - handles path resolution"""
    
    def __init__(self):
        """Initialize with path detection"""
        self.setup_paths()
        self._validate_database()
    
    def setup_paths(self):
        """Setup paths based on current working directory"""
        current_dir = Path.cwd()
        
        if current_dir.name == 'gui':
            # Running from gui folder
            self.project_root = current_dir.parent
            self.resources_path = self.project_root / "resources"
        else:
            # Running from project root
            self.project_root = current_dir
            self.resources_path = current_dir / "resources"
        
        # Database paths to try
        self.db_paths = [
            self.resources_path / "ocr2_search.db",

        ]
        
        # Find first existing database
        self.db_path = None
        for db_path in self.db_paths:
            if db_path.exists():
                self.db_path = db_path
                break
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Resources path: {self.resources_path}")
        logger.info(f"Selected database: {self.db_path}")
    
    def _validate_database(self):
        """Validate database exists"""
        if not self.db_path or not self.db_path.exists():
            available_dbs = [str(p) for p in self.db_paths if p.exists()]
            raise FileNotFoundError(
                f"No OCR database found. Tried: {[str(p) for p in self.db_paths]}. "
                f"Available: {available_dbs}"
            )
    
    def search_text(self, 
                   query: str, 
                   limit: int = 50, 
                   min_confidence: float = 0.0,
                   video_filters: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search OCR text in database
        
        Args:
            query: Search query
            limit: Max results
            min_confidence: Min confidence threshold
            video_filters: List of video IDs to filter
            
        Returns:
            Dict with results and metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build search query
                base_query = """
                SELECT frame_id, video_id, frame_number, ocr_text, avg_conf, ocr_path, frame_path
                FROM ocr_index 
                WHERE ocr_index MATCH ?
                AND avg_conf >= ?
                """
                
                params = [query, min_confidence]
                
                # Add video filters
                if video_filters:
                    placeholders = ','.join(['?' for _ in video_filters])
                    base_query += f" AND video_id IN ({placeholders})"
                    params.extend(video_filters)
                
                base_query += " ORDER BY avg_conf DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(base_query, params)
                rows = cursor.fetchall()
                
                # Convert to results format
                results = []
                for row in rows:
                    result = {
                        "frame_id": row[0],
                        "video_id": row[1], 
                        "frame_number": row[2],
                        "ocr_text": row[3],
                        "confidence": float(row[4]),
                        "ocr_path": row[5],
                        "frame_path": row[6]
                    }
                    results.append(result)
                
                return {
                    "results": results,
                    "total_found": len(results),
                    "query": query,
                    "search_params": {
                        "limit": limit,
                        "min_confidence": min_confidence,
                        "video_filters": video_filters
                    }
                }
                
        except Exception as e:
            logger.error(f"OCR search failed: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM ocr_index")
                total_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT video_id) FROM ocr_index")
                unique_videos = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(avg_conf) FROM ocr_index WHERE avg_conf > 0")
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT COUNT(*) FROM ocr_index WHERE LENGTH(ocr_text) > 0")
                records_with_text = cursor.fetchone()[0]
                
                return {
                    "total_records": total_records,
                    "unique_videos": unique_videos,
                    "average_confidence": round(float(avg_confidence), 3),
                    "records_with_text": records_with_text,
                    "records_without_text": total_records - records_with_text,
                    "database_path": str(self.db_path)
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            stats = self.get_database_stats()
            return {
                "status": "healthy",
                "service": "gui_ocr_search", 
                "database": "connected",
                "total_records": stats["total_records"],
                "database_path": str(self.db_path)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "gui_ocr_search",
                "error": str(e),
                "database_path": str(self.db_path) if self.db_path else "not_found"
            }