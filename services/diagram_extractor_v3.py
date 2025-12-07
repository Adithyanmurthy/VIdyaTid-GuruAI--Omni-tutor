"""
Diagram Extractor - Captures Complete Pages with Diagrams

This extracts entire pages that contain diagrams, preserving:
- All text on the page
- Section headings
- Complete diagrams with labels
- Captions
- Full context

Requirements: 5.1, 13.2
"""

import fitz  # PyMuPDF
from PIL import Image
import io
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DiagramExtractor:
    """
    Extracts complete pages that contain diagrams.
    """
    
    def __init__(self, zoom: float = 2.0):
        """
        Initialize extractor.
        
        Args:
            zoom: Zoom factor for rendering (2.0 = 144 DPI, 3.0 = 216 DPI)
        """
        self.zoom = zoom
        self.figure_patterns = [
            r'Figure\s+(\d+\.?\d*)',
            r'Fig\.\s+(\d+\.?\d*)',
            r'FIGURE\s+(\d+\.?\d*)',
        ]
    
    def find_figures_on_page(self, page: fitz.Page, page_text: str) -> List[str]:
        """
        Find all figure numbers mentioned on a page.
        
        Args:
            page: PyMuPDF page object
            page_text: Extracted text from page
            
        Returns:
            List of figure numbers found on the page
        """
        figures = set()
        
        for pattern in self.figure_patterns:
            matches = re.finditer(pattern, page_text, re.IGNORECASE)
            for match in matches:
                fig_number = match.group(1)
                # Validate figure number format (should be like "1.1", "2.3", etc.)
                if self._is_valid_figure_number(fig_number):
                    figures.add(fig_number)
        
        return sorted(list(figures), key=lambda x: float(x) if '.' in x else float(x))
    
    def _is_valid_figure_number(self, fig_number: str) -> bool:
        """
        Validate that a figure number is in the correct format.
        
        Args:
            fig_number: Figure number string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Figure numbers should be like "1.1", "2.3", or just "1", "2"
            # They should not be page numbers or other random numbers
            if '.' in fig_number:
                parts = fig_number.split('.')
                if len(parts) == 2:
                    chapter = int(parts[0])
                    fig = int(parts[1])
                    # Chapter should be reasonable (1-20), figure should be reasonable (1-50)
                    return 1 <= chapter <= 20 and 1 <= fig <= 50
            else:
                # Single digit figure numbers (less common but valid)
                num = int(fig_number)
                return 1 <= num <= 20
        except (ValueError, IndexError):
            return False
        
        return False
    
    def render_complete_page(self, page: fitz.Page) -> Image.Image:
        """
        Render the complete page as a high-quality image.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            PIL Image of the complete page
        """
        # Create transformation matrix for zoom
        mat = fitz.Matrix(self.zoom, self.zoom)
        
        # Render the entire page
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        return image
    
    def extract_caption_for_figure(self, page_text: str, figure_number: str) -> str:
        """
        Extract caption for a specific figure from page text.
        
        Args:
            page_text: Full page text
            figure_number: Figure number to find caption for
            
        Returns:
            Caption text if found
        """
        # Try multiple patterns to find the caption
        patterns = [
            # Pattern 1: "Figure X.Y: caption text" or "Figure X.Y caption text"
            rf'(?:Figure|Fig\.?)\s+{re.escape(figure_number)}[:\s]+([^\n]+?)(?=\n|Figure|\Z)',
            # Pattern 2: "Figure X.Y" followed by text on next line
            rf'(?:Figure|Fig\.?)\s+{re.escape(figure_number)}\s*\n\s*([^\n]+)',
            # Pattern 3: More flexible - capture until next figure or end
            rf'(?:Figure|Fig\.?)\s+{re.escape(figure_number)}[:\s]*([^\.]+(?:\.[^\.]+){{0,2}})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
            if match:
                caption = match.group(1).strip()
                # Clean up caption
                caption = re.sub(r'\s+', ' ', caption)
                # Remove common artifacts
                caption = re.sub(r'Reprint \d{4}-\d{2}', '', caption)
                caption = caption.strip()
                
                if len(caption) > 10:  # Only return if caption is meaningful
                    return caption[:500]  # Limit length
        
        return ""
    
    def extract_diagrams_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract complete pages that contain diagrams.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page data dictionaries
        """
        pages_with_diagrams = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                # Find all figures on this page
                figures = self.find_figures_on_page(page, page_text)
                
                if figures:
                    # This page has diagrams - render the complete page
                    try:
                        image = self.render_complete_page(page)
                        
                        # Extract captions for all figures on this page
                        captions = {}
                        for fig_num in figures:
                            caption = self.extract_caption_for_figure(page_text, fig_num)
                            if caption:
                                captions[fig_num] = caption
                        
                        # Create page data
                        page_data = {
                            'image': image,
                            'page_number': page_num + 1,
                            'figures': figures,  # List of figure numbers on this page
                            'captions': captions,  # Dict of figure_number: caption
                            'source_pdf': pdf_path.name,
                            'page_text': page_text[:1000]  # First 1000 chars for context
                        }
                        
                        pages_with_diagrams.append(page_data)
                        # Log with more detail for debugging
                        caption_info = ', '.join([f"{fig}: {captions.get(fig, 'No caption')[:50]}" for fig in figures])
                        logger.info(f"Captured page {page_num + 1} with figures: {', '.join(figures)} | Captions: {caption_info}")
                        
                    except Exception as e:
                        logger.warning(f"Error rendering page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            logger.info(f"Extracted {len(pages_with_diagrams)} pages with diagrams from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return pages_with_diagrams



