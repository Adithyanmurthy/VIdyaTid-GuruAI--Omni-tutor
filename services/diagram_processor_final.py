"""
Final Diagram Processor - Complete Page Capture with Indexing

Combines:
- V3 Extractor: Captures complete pages with diagrams
- Indexing: Database storage and visual embeddings

Requirements: 5.1, 13.2
"""

import os
import re
import io
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import logging

from PIL import Image
import numpy as np

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

from sqlalchemy import Column, String, Integer, Text, JSON, create_engine
from sqlalchemy.orm import sessionmaker

from models.database import Base
from config import Config
from services.diagram_extractor_v3 import DiagramExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiagramPage(Base):
    """Database model for storing complete page metadata."""
    __tablename__ = 'diagram_pages'
    
    page_id = Column(String(255), primary_key=True)
    subject = Column(String(100), nullable=False)
    class_level = Column(Integer, nullable=False)
    chapter = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    figures = Column(Text)  # Comma-separated figure numbers
    captions = Column(Text)  # JSON string of figure_number: caption
    file_path = Column(String(500), nullable=False)
    source_pdf = Column(String(255))
    image_hash = Column(String(64))
    width = Column(Integer)
    height = Column(Integer)
    
    def to_dict(self):
        return {
            'page_id': self.page_id,
            'subject': self.subject,
            'class_level': self.class_level,
            'chapter': self.chapter,
            'page_number': self.page_number,
            'figures': self.figures.split(',') if self.figures else [],
            'captions': self.captions,
            'file_path': self.file_path,
            'source_pdf': self.source_pdf,
            'width': self.width,
            'height': self.height
        }


class DiagramProcessor:
    """
    Final processor using complete page capture.
    """
    
    def __init__(self, pdf_directory: str, diagrams_directory: str, db_path: str = None):
        self.pdf_directory = Path(pdf_directory)
        self.diagrams_directory = Path(diagrams_directory)
        self.diagrams_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractor
        self.extractor = DiagramExtractor(zoom=2.0)  # 144 DPI
        
        # Initialize database
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'diagrams.db'
        
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize embedding model
        logger.info("Loading visual embedding model...")
        self.embedding_model = models.resnet50(pretrained=True)
        self.embedding_model = torch.nn.Sequential(*list(self.embedding_model.children())[:-1])
        self.embedding_model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB for diagrams...")
        chroma_path = Config.CHROMA_DB_PATH / 'diagrams'
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.diagram_collection = self.chroma_client.get_or_create_collection(
            name="ncert_diagram_pages",
            metadata={"description": "Complete pages with diagrams from NCERT textbooks"}
        )
    
    def parse_pdf_metadata(self, filename: str) -> Dict[str, any]:
        """
        Parse metadata from PDF filename.
        
        Handles multiple formats:
        - Biology_11_Ch01.pdf
        - Chemistry_11_Part1_Ch01.pdf
        - Physics_12_Part2_Ch03.pdf
        - Mathematics_11_Ch05.pdf
        
        For multi-part books, adjusts chapter numbers:
        - Part1 chapters remain as-is
        - Part2 chapters are offset based on subject:
          - Chemistry: Part1 has 6 chapters, so Part2 Ch01 becomes Ch07
          - Physics: Part1 has 7-8 chapters, so Part2 Ch01 becomes Ch08
          - Mathematics: Part1 has 6 chapters, so Part2 Ch01 becomes Ch07
        """
        name = filename.replace('.pdf', '')
        parts = name.split('_')
        
        subject = parts[0] if len(parts) > 0 else 'Unknown'
        class_level = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        
        # Find part number
        part_number = 1  # Default to Part 1
        for part in parts:
            part_match = re.search(r'Part(\d+)', part, re.IGNORECASE)
            if part_match:
                part_number = int(part_match.group(1))
                break
        
        # Find chapter number - could be in different positions
        chapter = 0
        for part in parts:
            ch_match = re.search(r'Ch(\d+)', part, re.IGNORECASE)
            if ch_match:
                chapter = int(ch_match.group(1))
                break
        
        # Adjust chapter number for Part 2 books
        if part_number == 2 and chapter > 0:
            # Chapter offsets for Part 2 (based on number of chapters in Part 1)
            offsets = {
                'Chemistry': 6,   # Part1 has Ch01-Ch06
                'Physics': 8,     # Part1 has Ch01-Ch08 (class 11), Ch01-Ch08 (class 12)
                'Mathematics': 6  # Part1 has Ch01-Ch06
            }
            
            # Special case: Physics class 11 Part1 has 7 chapters, Part2 also has 7
            if subject == 'Physics' and class_level == 11:
                chapter += 7
            elif subject in offsets:
                chapter += offsets[subject]
        
        return {
            'subject': subject,
            'class_level': class_level,
            'chapter': chapter
        }
    
    def compute_image_hash(self, image: Image.Image) -> str:
        """Compute SHA256 hash of image."""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        return hashlib.sha256(img_bytes.getvalue()).hexdigest()
    
    def generate_visual_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate visual embedding using ResNet50."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.embedding_model(img_tensor)
        
        return embedding.squeeze().numpy()
    
    def save_page(self, page_data: Dict, file_metadata: Dict) -> str:
        """Save page image to organized directory structure."""
        subject_dir = self.diagrams_directory / file_metadata['subject']
        chapter_dir = subject_dir / f"class{file_metadata['class_level']}_ch{file_metadata['chapter']}"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename with page number and figures
        figs = '_'.join(page_data['figures'])
        filename = f"page_{page_data['page_number']}_figs_{figs}.png"
        file_path = chapter_dir / filename
        
        # Save image
        page_data['image'].save(file_path, 'PNG')
        
        relative_path = file_path.relative_to(self.diagrams_directory)
        return str(relative_path)
    
    def index_page(self, page_data: Dict, file_metadata: Dict, file_path: str) -> str:
        """Index page in database and vector store."""
        # Generate unique ID
        page_id = f"{file_metadata['subject']}_{file_metadata['class_level']}_" \
                  f"ch{file_metadata['chapter']}_p{page_data['page_number']}"
        
        # Check if already exists
        existing = self.session.query(DiagramPage).filter_by(page_id=page_id).first()
        if existing:
            logger.debug(f"Page already indexed: {page_id}")
            return existing.page_id
        
        # Compute hash
        img_hash = self.compute_image_hash(page_data['image'])
        
        # Create database entry
        import json
        page = DiagramPage(
            page_id=page_id,
            subject=file_metadata['subject'],
            class_level=file_metadata['class_level'],
            chapter=file_metadata['chapter'],
            page_number=page_data['page_number'],
            figures=','.join(page_data['figures']),
            captions=json.dumps(page_data['captions']),
            file_path=file_path,
            source_pdf=page_data['source_pdf'],
            image_hash=img_hash,
            width=page_data['image'].width,
            height=page_data['image'].height
        )
        
        try:
            self.session.add(page)
            self.session.commit()
        except Exception as e:
            logger.error(f"Database error for {page_id}: {e}")
            self.session.rollback()
            return page_id
        
        # Generate embedding
        embedding = self.generate_visual_embedding(page_data['image'])
        
        # Store in ChromaDB
        metadata = {
            'subject': file_metadata['subject'],
            'class_level': str(file_metadata['class_level']),
            'chapter': str(file_metadata['chapter']),
            'page_number': str(page_data['page_number']),
            'figures': ','.join(page_data['figures']),
            'source_pdf': page_data['source_pdf']
        }
        
        self.diagram_collection.add(
            ids=[page_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )
        
        logger.debug(f"Indexed page: {page_id}")
        return page_id
    
    def process_pdf(self, pdf_path: Path) -> int:
        """Process a single PDF."""
        logger.info(f"Processing pages from {pdf_path.name}...")
        
        # Parse file metadata
        file_metadata = self.parse_pdf_metadata(pdf_path.name)
        
        # Extract pages with diagrams
        pages = self.extractor.extract_diagrams_from_pdf(pdf_path)
        
        if not pages:
            logger.warning(f"No pages with diagrams found in {pdf_path.name}")
            return 0
        
        # Process each page
        indexed_count = 0
        for page_data in pages:
            try:
                # Save page
                file_path = self.save_page(page_data, file_metadata)
                
                # Index page
                self.index_page(page_data, file_metadata, file_path)
                indexed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing page: {e}")
                continue
        
        logger.info(f"Indexed {indexed_count} pages from {pdf_path.name}")
        return indexed_count
    
    def process_all_pdfs(self) -> Dict[str, int]:
        """Process all PDFs in directory."""
        if not self.pdf_directory.exists():
            logger.error(f"PDF directory does not exist: {self.pdf_directory}")
            return {}
        
        pdf_files = list(self.pdf_directory.glob('*.pdf'))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return {}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {}
        for pdf_path in pdf_files:
            try:
                page_count = self.process_pdf(pdf_path)
                results[pdf_path.name] = page_count
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results[pdf_path.name] = 0
        
        total_pages = sum(results.values())
        logger.info(f"Processing complete: {total_pages} total pages from {len(pdf_files)} files")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about indexed pages."""
        total_pages = self.session.query(DiagramPage).count()
        
        subjects = {}
        for subject in ['Physics', 'Chemistry', 'Mathematics', 'Biology']:
            count = self.session.query(DiagramPage).filter_by(subject=subject).count()
            if count > 0:
                subjects[subject] = count
        
        classes = {}
        for class_level in [11, 12]:
            count = self.session.query(DiagramPage).filter_by(class_level=class_level).count()
            if count > 0:
                classes[f'Class {class_level}'] = count
        
        return {
            'total_pages': total_pages,
            'by_subject': subjects,
            'by_class': classes,
            'vector_store_count': self.diagram_collection.count()
        }
