"""
Image Processor - Handle image uploads and extract information

This module processes uploaded images to:
- Extract text using OCR (Tesseract and EasyOCR)
- Match diagrams against stored NCERT diagrams
- Classify content type (problem vs diagram)
- Preprocess images for better OCR results

Requirements: 2.1, 2.2, 4.5
"""

import io
import logging
import os
import platform
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# Optional OCR imports (may not be available on cloud)
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

# Optional ML imports
try:
    import torch
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Config
from services.diagram_processor_final import DiagramPage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Tesseract path for Windows
if platform.system() == 'Windows':
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\DELL\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Tesseract configured at: {path}")
            break
    else:
        logger.warning("Tesseract executable not found in common Windows locations")


class ImageProcessor:
    """
    Process uploaded images to extract text and match diagrams.
    
    Handles:
    - Image preprocessing (resize, denoise, contrast enhancement)
    - OCR text extraction using Tesseract and EasyOCR
    - Diagram matching against stored NCERT diagrams
    - Content type classification (problem vs diagram)
    """
    
    def __init__(self, db_path: str = None, chroma_path: str = None):
        """
        Initialize the image processor.
        
        Args:
            db_path: Path to SQLite database with diagram metadata
            chroma_path: Path to ChromaDB vector store
        """
        # Initialize database connection
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'diagrams.db'
        
        self.engine = create_engine(f'sqlite:///{db_path}')
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize EasyOCR reader (supports English and Hindi)
        logger.info("Initializing EasyOCR reader...")
        self.easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=torch.cuda.is_available())
        
        # Initialize embedding model for diagram matching
        logger.info("Loading visual embedding model for diagram matching...")
        self.embedding_model = models.resnet50(pretrained=True)
        self.embedding_model = torch.nn.Sequential(*list(self.embedding_model.children())[:-1])
        self.embedding_model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize ChromaDB for diagram matching
        if chroma_path is None:
            chroma_path = Config.CHROMA_DB_PATH / 'diagrams'
        
        logger.info("Connecting to ChromaDB for diagram matching...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.diagram_collection = self.chroma_client.get_collection(
                name="ncert_diagram_pages"
            )
            logger.info(f"Connected to diagram collection with {self.diagram_collection.count()} entries")
        except Exception as e:
            logger.warning(f"Could not load diagram collection: {e}")
            self.diagram_collection = None
    
    def _preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """
        Preprocess image for better OCR results.
        
        Steps:
        1. Resize if too large (max 3000px on longest side)
        2. Convert to grayscale for OCR
        3. Denoise using bilateral filter
        4. Enhance contrast
        5. Apply adaptive thresholding
        
        Args:
            image: Input PIL Image
            
        Returns:
            Tuple of (preprocessed PIL Image, OpenCV image array)
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        max_dimension = 3000
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {new_size}")
        
        # Convert to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Denoise using bilateral filter (preserves edges)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            contrast_enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Convert back to PIL Image
        processed_pil = Image.fromarray(binary)
        
        return processed_pil, img_cv
    
    def _extract_text_tesseract(self, image: Image.Image) -> str:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image: Preprocessed PIL Image
            
        Returns:
            Extracted text string
        """
        try:
            # Check if Tesseract is available
            if not hasattr(pytesseract, 'get_tesseract_version'):
                logger.warning("Tesseract not available, skipping")
                return ""
            
            # Configure Tesseract for better accuracy
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
        except Exception as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            return ""
    
    def _extract_text_easyocr(self, img_array: np.ndarray) -> str:
        """
        Extract text using EasyOCR.
        
        Args:
            img_array: OpenCV image array
            
        Returns:
            Extracted text string
        """
        try:
            results = self.easyocr_reader.readtext(img_array)
            # Combine all detected text
            text_parts = [result[1] for result in results]
            return ' '.join(text_parts).strip()
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return ""
    
    def extract_text(self, image_data: bytes) -> str:
        """
        Extract text from image using both Tesseract and EasyOCR.
        
        Uses both OCR engines and combines results for better accuracy.
        Falls back to EasyOCR if Tesseract is not available.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Extracted text string
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess
            processed_pil, img_cv = self._preprocess_image(image)
            
            # Extract text using both engines
            text_tesseract = self._extract_text_tesseract(processed_pil)
            text_easyocr = self._extract_text_easyocr(img_cv)
            
            # Use the longer result (usually more complete)
            # If Tesseract is not available, use EasyOCR
            if not text_tesseract:
                extracted_text = text_easyocr
                logger.debug("Using EasyOCR result (Tesseract not available)")
            elif len(text_easyocr) > len(text_tesseract):
                extracted_text = text_easyocr
                logger.debug("Using EasyOCR result")
            else:
                extracted_text = text_tesseract
                logger.debug("Using Tesseract result")
            
            logger.info(f"Extracted {len(extracted_text)} characters from image")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def _generate_visual_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate visual embedding for diagram matching.
        
        Args:
            image: PIL Image
            
        Returns:
            Embedding vector as numpy array
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.embedding_model(img_tensor)
        
        return embedding.squeeze().numpy()
    
    def find_matching_diagram(self, image_data: bytes, top_k: int = 3) -> Optional[Dict]:
        """
        Find matching NCERT diagram from stored collection.
        
        Uses visual similarity (embedding distance) to find the closest match.
        
        Args:
            image_data: Raw image bytes
            top_k: Number of top matches to retrieve
            
        Returns:
            Dictionary with match information or None if no good match found
        """
        if self.diagram_collection is None:
            logger.warning("Diagram collection not available")
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data))
            
            # Generate embedding
            embedding = self._generate_visual_embedding(image)
            
            # Query ChromaDB for similar diagrams
            results = self.diagram_collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k
            )
            
            if not results['ids'] or len(results['ids'][0]) == 0:
                logger.info("No matching diagrams found")
                return None
            
            # Get the best match
            best_match_id = results['ids'][0][0]
            best_match_distance = results['distances'][0][0]
            best_match_metadata = results['metadatas'][0][0]
            
            # Threshold for considering a match (lower is better)
            # Typical good matches have distance < 0.5
            if best_match_distance > 0.7:
                logger.info(f"Best match distance {best_match_distance:.3f} too high, no confident match")
                return None
            
            # Get full diagram information from database
            page_id = best_match_id
            diagram_page = self.session.query(DiagramPage).filter_by(page_id=page_id).first()
            
            if diagram_page:
                match_info = {
                    'page_id': page_id,
                    'subject': diagram_page.subject,
                    'class_level': diagram_page.class_level,
                    'chapter': diagram_page.chapter,
                    'page_number': diagram_page.page_number,
                    'figures': diagram_page.figures.split(',') if diagram_page.figures else [],
                    'captions': diagram_page.captions,
                    'file_path': diagram_page.file_path,
                    'similarity_score': 1.0 - best_match_distance,  # Convert distance to similarity
                    'confidence': 'high' if best_match_distance < 0.3 else 'medium'
                }
                
                logger.info(f"Found matching diagram: {page_id} (similarity: {match_info['similarity_score']:.3f})")
                return match_info
            else:
                logger.warning(f"Diagram page {page_id} not found in database")
                return None
                
        except Exception as e:
            logger.error(f"Error finding matching diagram: {e}")
            return None
    
    def _classify_content_type(self, extracted_text: str, matched_diagram: Optional[Dict]) -> str:
        """
        Classify the content type as 'problem' or 'diagram'.
        
        Classification logic:
        - If diagram match found with high confidence -> 'diagram'
        - If text contains question indicators -> 'problem'
        - If text is minimal and diagram match exists -> 'diagram'
        - Otherwise -> 'problem' (default)
        
        Args:
            extracted_text: Text extracted from image
            matched_diagram: Diagram match information (if any)
            
        Returns:
            Content type: 'problem' or 'diagram'
        """
        # Check for high-confidence diagram match
        if matched_diagram and matched_diagram.get('confidence') == 'high':
            return 'diagram'
        
        # Check for question/problem indicators
        problem_indicators = [
            'solve', 'find', 'calculate', 'prove', 'show that',
            'determine', 'compute', 'evaluate', 'derive',
            'question', 'problem', 'exercise', '?',
            'what', 'how', 'why', 'when', 'where'
        ]
        
        text_lower = extracted_text.lower()
        has_problem_indicators = any(indicator in text_lower for indicator in problem_indicators)
        
        if has_problem_indicators:
            return 'problem'
        
        # If text is minimal (< 50 chars) and we have a diagram match
        if len(extracted_text) < 50 and matched_diagram:
            return 'diagram'
        
        # Default to problem
        return 'problem'
    
    async def process_image(self, image_data: bytes) -> Dict:
        """
        Process uploaded image and extract all relevant information.
        
        Main entry point for image processing. Performs:
        1. Image preprocessing
        2. Text extraction using OCR
        3. Diagram matching against NCERT collection
        4. Content type classification
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary containing:
                - type: 'problem' or 'diagram'
                - text: Extracted text
                - matched_diagram: Diagram match info (if found)
                - confidence: Overall confidence in the processing
        """
        logger.info("Processing uploaded image...")
        
        try:
            # Extract text
            extracted_text = self.extract_text(image_data)
            
            # Find matching diagram
            matched_diagram = self.find_matching_diagram(image_data)
            
            # Classify content type
            content_type = self._classify_content_type(extracted_text, matched_diagram)
            
            # Determine overall confidence
            if matched_diagram and matched_diagram.get('confidence') == 'high':
                confidence = 'high'
            elif matched_diagram or len(extracted_text) > 20:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            result = {
                'type': content_type,
                'text': extracted_text,
                'matched_diagram': matched_diagram,
                'confidence': confidence
            }
            
            logger.info(f"Image processing complete: type={content_type}, confidence={confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                'type': 'unknown',
                'text': '',
                'matched_diagram': None,
                'confidence': 'low',
                'error': str(e)
            }
    
    def validate_image(self, image_data: bytes) -> Tuple[bool, str]:
        """
        Validate uploaded image before processing.
        
        Checks:
        - File can be opened as image
        - Image dimensions are reasonable
        - File size is within limits
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size (max 10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if len(image_data) > max_size:
                return False, f"Image file too large: {len(image_data) / 1024 / 1024:.1f}MB (max 10MB)"
            
            # Try to open image
            image = Image.open(io.BytesIO(image_data))
            
            # Check dimensions (max 10000x10000)
            max_dimension = 10000
            if max(image.size) > max_dimension:
                return False, f"Image dimensions too large: {image.size} (max {max_dimension}px)"
            
            # Check minimum dimensions (at least 50x50)
            min_dimension = 50
            if min(image.size) < min_dimension:
                return False, f"Image dimensions too small: {image.size} (min {min_dimension}px)"
            
            return True, "Image valid"
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
