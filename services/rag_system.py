"""
RAG (Retrieval-Augmented Generation) System for GuruAI

This module implements the RAG system for retrieving relevant NCERT content
to ground AI responses in factual information.

Features:
- Semantic search using ChromaDB vector store
- Result reranking by relevance
- Multi-chapter detection and reference extraction
- Out-of-scope query handling

Requirements: 1.1, 1.2, 1.4, 1.5
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Embeddings and Vector Store (optional imports)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False

# Configuration
from config import Config

# Cloudflare AI
from services.cloudflare_ai import get_cloudflare_ai, is_cloudflare_ai_enabled

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation system for NCERT content.
    
    Handles semantic search, result reranking, and context retrieval
    to provide grounded responses based on NCERT textbooks.
    """
    
    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_store_path: Path to ChromaDB vector store (default: from config)
            embedding_model_name: Name of the embedding model (default: from config)
            collection_name: Name of the ChromaDB collection (default: from config)
        """
        self.vector_store_path = Path(vector_store_path or Config.CHROMA_DB_PATH)
        self.embedding_model_name = embedding_model_name or Config.EMBEDDING_MODEL_NAME
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        
        # Initialize embedding model (only if Cloudflare is not enabled)
        if is_cloudflare_ai_enabled():
            logger.info("Using Cloudflare AI for embeddings (768d) - skipping local model load")
            self.embedding_model = None
        else:
            logger.info(f"Loading local embedding model: {self.embedding_model_name}")
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Local embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        
        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at: {self.vector_store_path}")
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_store_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create the collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                count = self.collection.count()
                logger.info(f"Connected to existing collection '{self.collection_name}' with {count} documents")
            except Exception:
                logger.warning(f"Collection '{self.collection_name}' not found, creating new collection")
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "NCERT textbook content for JEE/NEET preparation"}
                )
                logger.info(f"Created new collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context from NCERT content.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            filters: Optional metadata filters (e.g., {'subject': 'Physics'})
        
        Returns:
            List of dictionaries containing retrieved context with metadata
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            # Generate query embedding using Cloudflare AI or local model
            logger.debug(f"Encoding query: {query[:100]}...")
            
            if is_cloudflare_ai_enabled():
                try:
                    logger.debug("Using Cloudflare AI (BGE) for embeddings")
                    cf_ai = get_cloudflare_ai()
                    query_embedding = [cf_ai.generate_embeddings(query)]
                except Exception as e:
                    logger.error(f"Cloudflare embeddings failed, using local model: {e}")
                    query_embedding = self.embedding_model.encode([query]).tolist()
            else:
                query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Prepare query parameters
            query_params = {
                'query_embeddings': query_embedding,
                'n_results': top_k
            }
            
            # Add filters if provided
            if filters:
                query_params['where'] = filters
            
            # Query the collection
            logger.debug(f"Querying collection with top_k={top_k}")
            results = self.collection.query(**query_params)
            
            # Format results
            formatted_results = self._format_results(results, query)
            
            # Rerank results by relevance
            reranked_results = self._rerank_results(formatted_results, query)
            
            logger.info(f"Retrieved {len(reranked_results)} results for query")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _format_results(self, raw_results: Dict, query: str) -> List[Dict]:
        """
        Format raw ChromaDB results into structured dictionaries.
        
        Args:
            raw_results: Raw results from ChromaDB query
            query: Original query string
        
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        if not raw_results or not raw_results.get('documents'):
            return formatted
        
        # ChromaDB returns results as lists of lists
        documents = raw_results['documents'][0] if raw_results['documents'] else []
        metadatas = raw_results['metadatas'][0] if raw_results['metadatas'] else []
        distances = raw_results['distances'][0] if raw_results['distances'] else []
        ids = raw_results['ids'][0] if raw_results['ids'] else []
        
        for i, (doc, metadata, distance, doc_id) in enumerate(
            zip(documents, metadatas, distances, ids)
        ):
            formatted.append({
                'content': doc,
                'metadata': metadata,
                'distance': distance,
                'id': doc_id,
                'rank': i + 1,
                'relevance_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
            })
        
        return formatted
    
    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Rerank results by relevance using additional scoring factors.
        
        Args:
            results: List of formatted results
            query: Original query string
        
        Returns:
            Reranked list of results
        """
        if not results:
            return results
        
        # Calculate additional relevance factors
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for result in results:
            content_lower = result['content'].lower()
            
            # Factor 1: Exact phrase match bonus
            if query_lower in content_lower:
                result['relevance_score'] *= 1.3
            
            # Factor 2: Term overlap bonus
            content_terms = set(content_lower.split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                overlap_ratio = overlap / len(query_terms)
                result['relevance_score'] *= (1.0 + overlap_ratio * 0.2)
            
            # Factor 3: Content length penalty (prefer concise, relevant content)
            content_length = len(result['content'].split())
            if content_length > 500:
                result['relevance_score'] *= 0.95
        
        # Sort by relevance score (descending)
        reranked = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked):
            result['rank'] = i + 1
        
        return reranked
    
    def detect_multi_chapter_references(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Detect and group results by chapter when multiple chapters are referenced.
        
        Args:
            results: List of retrieved results
        
        Returns:
            Dictionary mapping chapter identifiers to their results
        """
        chapter_groups = {}
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Create chapter identifier
            subject = metadata.get('subject', 'Unknown')
            class_level = metadata.get('class_level', 'Unknown')
            chapter = metadata.get('chapter', 'Unknown')
            
            chapter_id = f"{subject}_Class{class_level}_Ch{chapter}"
            
            if chapter_id not in chapter_groups:
                chapter_groups[chapter_id] = []
            
            chapter_groups[chapter_id].append(result)
        
        return chapter_groups
    
    def extract_references(self, results: List[Dict]) -> List[str]:
        """
        Extract formatted references from results.
        
        Args:
            results: List of retrieved results
        
        Returns:
            List of formatted reference strings
        """
        references = []
        seen_refs = set()
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Extract reference components
            subject = metadata.get('subject', 'Unknown')
            class_level = metadata.get('class_level', 'Unknown')
            chapter = metadata.get('chapter', 'Unknown')
            page = metadata.get('page_number', 'Unknown')
            
            # Format reference
            ref = f"{subject} Class {class_level}, Chapter {chapter}, Page {page}"
            
            # Avoid duplicates
            if ref not in seen_refs:
                references.append(ref)
                seen_refs.add(ref)
        
        return references
    
    def is_out_of_scope(self, query: str, results: List[Dict], threshold: float = 0.15) -> bool:
        """
        Determine if a query is out of scope (not covered in NCERT).
        
        Args:
            query: User query string
            results: Retrieved results
            threshold: Minimum relevance score threshold (lowered to 0.15 for better coverage)
        
        Returns:
            True if query is out of scope, False otherwise
        """
        # If vector store is empty, don't mark as out of scope - let AI handle it
        if not results:
            # Check if collection is empty
            try:
                count = self.collection.count()
                if count == 0:
                    logger.info("Vector store is empty - allowing query to proceed to AI")
                    return False  # Let AI handle it directly
            except Exception:
                pass
            return True
        
        # Check if best result meets threshold
        best_score = results[0].get('relevance_score', 0.0)
        
        if best_score < threshold:
            logger.info(f"Query appears out of scope (best score: {best_score:.3f})")
            return True
        
        return False
    
    def handle_out_of_scope_query(self, query: str) -> Dict:
        """
        Handle queries that are out of NCERT scope.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with out-of-scope message and suggestions
        """
        return {
            'out_of_scope': True,
            'message': (
                "This topic doesn't appear to be covered in NCERT textbooks. "
                "GuruAI focuses on NCERT content for JEE and NEET preparation. "
                "Please try asking about topics from NCERT Physics, Chemistry, "
                "Mathematics, or Biology (Classes 11-12)."
            ),
            'suggestions': self._get_topic_suggestions(query)
        }
    
    def _get_topic_suggestions(self, query: str) -> List[str]:
        """
        Get topic suggestions for out-of-scope queries.
        
        Args:
            query: User query string
        
        Returns:
            List of suggested topics
        """
        # Common NCERT topics by subject
        suggestions_map = {
            'physics': [
                "Newton's laws of motion",
                "Thermodynamics",
                "Electromagnetism",
                "Optics and wave theory"
            ],
            'chemistry': [
                "Chemical bonding",
                "Organic chemistry reactions",
                "Periodic table trends",
                "Electrochemistry"
            ],
            'math': [
                "Calculus and derivatives",
                "Trigonometry",
                "Probability and statistics",
                "Vectors and 3D geometry"
            ],
            'biology': [
                "Cell structure and function",
                "Genetics and evolution",
                "Human physiology",
                "Plant biology"
            ]
        }
        
        query_lower = query.lower()
        
        # Try to match subject
        for subject, topics in suggestions_map.items():
            if subject in query_lower:
                return topics[:3]
        
        # Return general suggestions
        return [
            "Try asking about specific NCERT chapters",
            "Search for fundamental concepts in your subject",
            "Ask about topics from your JEE/NEET syllabus"
        ]
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 5,
        include_references: bool = True
    ) -> Dict:
        """
        Get formatted context for LLM prompt generation.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve
            include_references: Whether to include reference information
        
        Returns:
            Dictionary with context, references, and metadata
        """
        # Retrieve results
        results = self.retrieve(query, top_k=top_k)
        
        # Check if out of scope
        if self.is_out_of_scope(query, results):
            return self.handle_out_of_scope_query(query)
        
        # Handle empty results (vector store empty) - return empty context for AI to handle
        if not results:
            logger.info("No results found (vector store may be empty) - returning empty context for AI")
            return {
                'out_of_scope': False,
                'context': '',
                'passages': [],
                'references': [],
                'multi_chapter': False,
                'chapter_groups': {},
                'num_results': 0,
                'top_relevance_score': 0.0,
                'no_rag_data': True  # Flag to indicate AI should answer without RAG
            }
        
        # Extract context passages
        context_passages = [result['content'] for result in results]
        
        # Detect multi-chapter references
        chapter_groups = self.detect_multi_chapter_references(results)
        
        # Extract references
        references = self.extract_references(results) if include_references else []
        
        # Format context
        formatted_context = {
            'out_of_scope': False,
            'context': '\n\n'.join(context_passages),
            'passages': context_passages,
            'references': references,
            'multi_chapter': len(chapter_groups) > 1,
            'chapter_groups': chapter_groups,
            'num_results': len(results),
            'top_relevance_score': results[0]['relevance_score'] if results else 0.0
        }
        
        return formatted_context
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        subject: Optional[str] = None,
        class_level: Optional[str] = None,
        chapter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search NCERT content with optional filters.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            subject: Optional subject filter
            class_level: Optional class level filter
            chapter: Optional chapter filter
        
        Returns:
            List of search results with content and metadata
        """
        # Build filters
        filters = {}
        if subject:
            filters['subject'] = subject
        if class_level:
            filters['class_level'] = str(class_level)
        if chapter:
            filters['chapter'] = str(chapter)
        
        # Retrieve results
        results = self.retrieve(query, top_k=top_k, filters=filters if filters else None)
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            count = self.collection.count()
            
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name,
                'vector_store_path': str(self.vector_store_path)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


def main():
    """
    Test the RAG system with sample queries.
    """
    print("="*60)
    print("RAG System Test")
    print("="*60)
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag = RAGSystem()
    
    # Display stats
    stats = rag.get_stats()
    print(f"\nVector Store Stats:")
    print(f"  Total documents: {stats.get('total_documents', 0)}")
    print(f"  Collection: {stats.get('collection_name', 'N/A')}")
    print(f"  Embedding model: {stats.get('embedding_model', 'N/A')}")
    
    # Test queries
    test_queries = [
        "What is Newton's first law of motion?",
        "Explain photosynthesis in plants",
        "What is the derivative of x squared?",
        "Describe the structure of benzene"
    ]
    
    print("\n" + "="*60)
    print("Test Queries")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        # Get context
        context = rag.get_context_for_llm(query, top_k=3)
        
        if context.get('out_of_scope'):
            print("Status: OUT OF SCOPE")
            print(f"Message: {context['message']}")
        else:
            print(f"Status: IN SCOPE")
            print(f"Results: {context['num_results']}")
            print(f"Multi-chapter: {context['multi_chapter']}")
            print(f"Top relevance: {context['top_relevance_score']:.3f}")
            
            if context['references']:
                print(f"\nReferences:")
                for ref in context['references'][:3]:
                    print(f"  - {ref}")
            
            if context['passages']:
                print(f"\nTop passage preview:")
                print(f"  {context['passages'][0][:200]}...")


if __name__ == "__main__":
    main()
