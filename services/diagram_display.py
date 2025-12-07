"""
Diagram Display Service for GuruAI

Handles diagram retrieval from local storage and provides
metadata for frontend display.

Features:
- Retrieve diagrams by ID or query
- Get diagram metadata (captions, page references)
- Handle labeled part explanations
- Serve diagram files from local storage

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
from PIL import Image

from services.diagram_processor_final import DiagramPage
from models.database import Base
from config import Config

logger = logging.getLogger(__name__)


class DiagramDisplayService:
    """
    Service for retrieving and displaying diagrams from local storage.
    
    Provides:
    - Diagram retrieval by ID or semantic search
    - Metadata extraction (captions, page references)
    - Labeled part explanations
    - File path resolution for frontend display
    """
    
    def __init__(self, db_path: str = None, diagrams_directory: str = None):
        """
        Initialize diagram display service.
        
        Args:
            db_path: Path to diagrams database
            diagrams_directory: Root directory for diagram images
        """
        # Initialize database connection
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'diagrams.db'
        
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Set diagrams directory
        if diagrams_directory is None:
            self.diagrams_directory = Path(__file__).parent.parent / 'diagrams'
        else:
            self.diagrams_directory = Path(diagrams_directory)
        
        logger.info(f"DiagramDisplayService initialized with {self.diagrams_directory}")
    
    def get_diagram_by_id(self, page_id: str) -> Optional[Dict]:
        """
        Retrieve diagram page by ID.
        
        Args:
            page_id: Unique page identifier
            
        Returns:
            Dictionary with diagram data and metadata, or None if not found
        """
        page = self.session.query(DiagramPage).filter_by(page_id=page_id).first()
        
        if not page:
            logger.warning(f"Diagram page not found: {page_id}")
            return None
        
        return self._format_diagram_response(page)
    
    def get_diagrams_by_chapter(self, subject: str, class_level: int, chapter: int) -> List[Dict]:
        """
        Retrieve all diagrams from a specific chapter.
        
        Args:
            subject: Subject name (Physics, Chemistry, etc.)
            class_level: Class level (11 or 12)
            chapter: Chapter number
            
        Returns:
            List of diagram dictionaries
        """
        pages = self.session.query(DiagramPage).filter_by(
            subject=subject,
            class_level=class_level,
            chapter=chapter
        ).order_by(DiagramPage.page_number).all()
        
        return [self._format_diagram_response(page) for page in pages]
    
    def get_diagrams_by_figure(self, subject: str, class_level: int, chapter: int, figure_number: str) -> Optional[Dict]:
        """
        Retrieve diagram page containing a specific figure.
        
        Args:
            subject: Subject name
            class_level: Class level (11 or 12)
            chapter: Chapter number
            figure_number: Figure number (e.g., "1.1", "2.5")
            
        Returns:
            Diagram dictionary or None if not found
        """
        # Query pages that contain this figure number
        pages = self.session.query(DiagramPage).filter_by(
            subject=subject,
            class_level=class_level,
            chapter=chapter
        ).all()
        
        for page in pages:
            figures = page.figures.split(',') if page.figures else []
            if figure_number in figures:
                return self._format_diagram_response(page)
        
        logger.warning(f"Figure {figure_number} not found in {subject} Class {class_level} Chapter {chapter}")
        return None
    
    def search_diagrams_by_caption(self, query: str, subject: Optional[str] = None, 
                                   class_level: Optional[int] = None) -> List[Dict]:
        """
        Search diagrams by caption text.
        
        Args:
            query: Search query
            subject: Optional subject filter
            class_level: Optional class level filter
            
        Returns:
            List of matching diagram dictionaries
        """
        query_lower = query.lower()
        
        # Build base query
        base_query = self.session.query(DiagramPage)
        
        if subject:
            base_query = base_query.filter_by(subject=subject)
        if class_level:
            base_query = base_query.filter_by(class_level=class_level)
        
        pages = base_query.all()
        
        # Filter by caption match
        matching_pages = []
        for page in pages:
            try:
                captions_dict = json.loads(page.captions) if page.captions else {}
                # Check if query appears in any caption
                for caption in captions_dict.values():
                    if query_lower in caption.lower():
                        matching_pages.append(page)
                        break
            except json.JSONDecodeError:
                continue
        
        return [self._format_diagram_response(page) for page in matching_pages]
    
    def get_diagram_file_path(self, page_id: str) -> Optional[Path]:
        """
        Get absolute file path for a diagram.
        
        Args:
            page_id: Unique page identifier
            
        Returns:
            Absolute path to diagram file, or None if not found
        """
        page = self.session.query(DiagramPage).filter_by(page_id=page_id).first()
        
        if not page:
            return None
        
        file_path = self.diagrams_directory / page.file_path
        
        if not file_path.exists():
            logger.error(f"Diagram file not found: {file_path}")
            return None
        
        return file_path
    
    def get_labeled_parts_explanation(self, page_id: str, context_text: str = None) -> Dict:
        """
        Generate explanation for labeled parts in a diagram.
        
        Args:
            page_id: Unique page identifier
            context_text: Optional NCERT context text for the diagram
            
        Returns:
            Dictionary with labeled parts and explanations
        """
        page = self.session.query(DiagramPage).filter_by(page_id=page_id).first()
        
        if not page:
            return {"error": "Diagram not found"}
        
        # Parse captions to extract labeled parts
        try:
            captions_dict = json.loads(page.captions) if page.captions else {}
        except json.JSONDecodeError:
            captions_dict = {}
        
        # Extract labeled parts from captions
        # Common patterns: "A - description", "1. description", "(a) description"
        labeled_parts = {}
        
        for fig_num, caption in captions_dict.items():
            parts = self._extract_labeled_parts_from_caption(caption)
            if parts:
                labeled_parts[fig_num] = parts
        
        return {
            "page_id": page_id,
            "figures": page.figures.split(',') if page.figures else [],
            "labeled_parts": labeled_parts,
            "context": context_text if context_text else "No additional context provided"
        }
    
    def _extract_labeled_parts_from_caption(self, caption: str) -> List[Dict]:
        """
        Extract labeled parts from caption text.
        
        Args:
            caption: Caption text
            
        Returns:
            List of dictionaries with label and description
        """
        import re
        
        labeled_parts = []
        
        # Pattern 1: "A - description" or "A: description"
        pattern1 = r'([A-Z])\s*[-:]\s*([^,;\.]+)'
        matches1 = re.finditer(pattern1, caption)
        for match in matches1:
            labeled_parts.append({
                "label": match.group(1),
                "description": match.group(2).strip()
            })
        
        # Pattern 2: "(a) description" or "(1) description"
        pattern2 = r'\(([a-z0-9])\)\s*([^,;\.]+)'
        matches2 = re.finditer(pattern2, caption, re.IGNORECASE)
        for match in matches2:
            labeled_parts.append({
                "label": match.group(1),
                "description": match.group(2).strip()
            })
        
        # Pattern 3: "1. description" or "i. description"
        pattern3 = r'([0-9]+|[ivxIVX]+)\.\s*([^,;\.]+)'
        matches3 = re.finditer(pattern3, caption)
        for match in matches3:
            labeled_parts.append({
                "label": match.group(1),
                "description": match.group(2).strip()
            })
        
        return labeled_parts
    
    def _format_diagram_response(self, page: DiagramPage) -> Dict:
        """
        Format diagram page data for API response.
        
        Args:
            page: DiagramPage database object
            
        Returns:
            Formatted dictionary with all diagram metadata
        """
        try:
            captions_dict = json.loads(page.captions) if page.captions else {}
        except json.JSONDecodeError:
            captions_dict = {}
        
        # Create a clear reference that shows the context
        figures_list = page.figures.split(',') if page.figures else []
        figure_text = ', '.join(figures_list)
        
        # Enhanced reference showing both figure numbers and chapter context
        reference = f"{page.subject} Class {page.class_level}, Chapter {page.chapter}"
        if figures_list:
            reference += f" (Figure {figure_text})"
        
        return {
            "page_id": page.page_id,
            "subject": page.subject,
            "class_level": page.class_level,
            "chapter": page.chapter,
            "page_number": page.page_number,
            "figures": figures_list,
            "captions": captions_dict,
            "file_path": page.file_path,
            "source_pdf": page.source_pdf,
            "width": page.width,
            "height": page.height,
            "reference": reference,
            "display_note": self._get_display_note(page, figures_list)
        }
    
    def _get_display_note(self, page: DiagramPage, figures_list: List[str]) -> Optional[str]:
        """
        Generate a display note if figure numbering might be confusing.
        
        Args:
            page: DiagramPage database object
            figures_list: List of figure numbers
            
        Returns:
            Display note string or None
        """
        # Check if any figure number doesn't match the chapter
        for fig in figures_list:
            if '.' in fig:
                try:
                    fig_chapter = int(fig.split('.')[0])
                    if fig_chapter != page.chapter:
                        # This is from a Part 2 book where figure numbering differs
                        return f"Note: This is from Chapter {page.chapter}. Figure numbering follows the original NCERT textbook."
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def get_diagrams_for_concept(self, concept: str, subject: Optional[str] = None) -> List[Dict]:
        """
        Get relevant diagrams for a concept based on caption matching.
        
        Args:
            concept: Concept or topic to search for
            subject: Optional subject filter
            
        Returns:
            List of relevant diagram dictionaries
        """
        return self.search_diagrams_by_caption(concept, subject=subject)
    
    def close(self):
        """Close database session."""
        self.session.close()
