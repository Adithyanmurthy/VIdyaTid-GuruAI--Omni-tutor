"""
NCERT Content Processing Pipeline

This module handles:
1. Extracting text from NCERT PDFs
2. Chunking text into 500-word chunks with overlap
3. Generating embeddings using sentence-transformers
4. Storing embeddings in ChromaDB vector store

Requirements: 1.1, 13.1
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# PDF Processing
import PyPDF2
import pdfplumber

# Text Processing
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Embeddings and Vector Store
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configuration
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NCERTProcessor:
    """
    Processes NCERT PDF textbooks and creates a searchable vector store.
    """
    
    def __init__(self, pdf_directory: str, vector_store_path: str):
        """
        Initialize the NCERT processor.
        
        Args:
            pdf_directory: Path to directory containing NCERT PDFs
            vector_store_path: Path where ChromaDB vector store will be created
        """
        self.pdf_directory = Path(pdf_directory)
        self.vector_store_path = Path(vector_store_path)
        
        # Initialize embedding model (use Cloudflare if available)
        from services.cloudflare_ai import get_cloudflare_ai, is_cloudflare_ai_enabled
        
        self.use_cloudflare = is_cloudflare_ai_enabled()
        
        if self.use_cloudflare:
            logger.info("Using Cloudflare AI for embeddings (768d)")
            self.cloudflare_ai = get_cloudflare_ai()
            self.embedding_model = None
        else:
            logger.info("Loading local embedding model (384d)...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.cloudflare_ai = None
        
        # Initialize ChromaDB client
        logger.info("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="ncert_content",
            metadata={"description": "NCERT textbook content for JEE/NEET preparation"}
        )
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Extract text from a PDF file with page numbers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_data.append({
                            'text': text.strip(),
                            'page_number': page_num,
                            'file_name': pdf_path.name
                        })
                        
            logger.info(f"Extracted {len(pages_data)} pages from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            
        return pages_data
    
    def parse_pdf_metadata(self, filename: str) -> Dict[str, str]:
        """
        Parse metadata from PDF filename.
        Expected format: Subject_Class_ChapterNumber.pdf
        Example: Physics_12_01.pdf
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Dictionary containing subject, class, and chapter metadata
        """
        # Remove .pdf extension
        name = filename.replace('.pdf', '')
        
        # Try to parse structured filename
        parts = name.split('_')
        
        metadata = {
            'subject': parts[0] if len(parts) > 0 else 'Unknown',
            'class_level': parts[1] if len(parts) > 1 else 'Unknown',
            'chapter': parts[2] if len(parts) > 2 else 'Unknown'
        }
        
        return metadata
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into chunks of approximately chunk_size words with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Target number of words per chunk
            overlap: Number of words to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            word_count = len(words)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_word_count + word_count > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_words = 0
                for sent in reversed(current_chunk):
                    sent_words = len(word_tokenize(sent))
                    if overlap_words + sent_words <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_words += sent_words
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_word_count = overlap_words
            
            current_chunk.append(sentence)
            current_word_count += word_count
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_pdf(self, pdf_path: Path) -> int:
        """
        Process a single PDF file: extract text, chunk it, generate embeddings, and store.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks processed
        """
        logger.info(f"Processing {pdf_path.name}...")
        
        # Extract text from PDF
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        if not pages_data:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return 0
        
        # Parse metadata from filename
        file_metadata = self.parse_pdf_metadata(pdf_path.name)
        
        # Process each page
        total_chunks = 0
        documents = []
        metadatas = []
        ids = []
        
        for page_data in pages_data:
            # Chunk the page text
            chunks = self.chunk_text(page_data['text'])
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID
                chunk_id = f"{pdf_path.stem}_p{page_data['page_number']}_c{chunk_idx}"
                
                # Create metadata
                metadata = {
                    'subject': file_metadata['subject'],
                    'class_level': file_metadata['class_level'],
                    'chapter': file_metadata['chapter'],
                    'page_number': str(page_data['page_number']),
                    'file_name': pdf_path.name,
                    'chunk_index': str(chunk_idx)
                }
                
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(chunk_id)
                total_chunks += 1
        
        # Generate embeddings and store in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            # Generate embeddings (Cloudflare or local)
            if self.use_cloudflare:
                embeddings = []
                for doc in batch_docs:
                    try:
                        embedding = self.cloudflare_ai.generate_embeddings(doc)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Cloudflare embedding failed: {e}")
                        # Use zero vector as fallback
                        embeddings.append([0.0] * 768)
            else:
                embeddings = self.embedding_model.encode(batch_docs).tolist()
            
            # Add to collection
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
                embeddings=embeddings
            )
            
            logger.info(f"Stored batch {i//batch_size + 1} ({len(batch_docs)} chunks)")
        
        logger.info(f"Completed processing {pdf_path.name}: {total_chunks} chunks created")
        return total_chunks
    
    def process_all_pdfs(self) -> Dict[str, int]:
        """
        Process all PDF files in the configured directory.
        
        Returns:
            Dictionary mapping filenames to number of chunks processed
        """
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
                chunk_count = self.process_pdf(pdf_path)
                results[pdf_path.name] = chunk_count
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results[pdf_path.name] = 0
        
        # Summary
        total_chunks = sum(results.values())
        logger.info(f"Processing complete: {total_chunks} total chunks from {len(pdf_files)} files")
        
        return results
    
    def query_vector_store(self, query: str, top_k: int = 5) -> Dict:
        """
        Query the vector store for relevant content.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary containing results with documents, metadata, and distances
        """
        # Generate query embedding (Cloudflare or local)
        if self.use_cloudflare:
            query_embedding = [self.cloudflare_ai.generate_embeddings(query)]
        else:
            query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        count = self.collection.count()
        
        return {
            'total_chunks': count,
            'collection_name': self.collection.name,
            'metadata': self.collection.metadata
        }


def main():
    """
    Main function to run the NCERT processing pipeline.
    """
    # Configuration
    pdf_dir = Config.NCERT_PDF_DIR
    vector_store_dir = Config.VECTOR_STORE_DIR
    
    # Create processor
    processor = NCERTProcessor(pdf_dir, vector_store_dir)
    
    # Process all PDFs
    results = processor.process_all_pdfs()
    
    # Display results
    print("\n" + "="*50)
    print("NCERT Processing Complete")
    print("="*50)
    for filename, chunk_count in results.items():
        print(f"{filename}: {chunk_count} chunks")
    
    # Display collection stats
    stats = processor.get_collection_stats()
    print("\n" + "="*50)
    print("Vector Store Statistics")
    print("="*50)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Collection: {stats['collection_name']}")
    
    # Test query
    print("\n" + "="*50)
    print("Test Query")
    print("="*50)
    test_query = "What is Newton's first law of motion?"
    print(f"Query: {test_query}")
    results = processor.query_vector_store(test_query, top_k=3)
    
    if results['documents']:
        print(f"\nTop {len(results['documents'][0])} results:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            print(f"\n{i}. {metadata['subject']} - Class {metadata['class_level']} - Chapter {metadata['chapter']}")
            print(f"   Page: {metadata['page_number']}")
            print(f"   Content: {doc[:200]}...")


if __name__ == "__main__":
    main()