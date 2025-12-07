"""
Search Service for GuruAI - Semantic search across NCERT content.

This module implements comprehensive search functionality including:
- Semantic search across NCERT content
- Metadata extraction (subject, class, chapter, page)
- Result ranking by relevance
- Context display with surrounding paragraphs
- Diagram inclusion in search results when captions match
- Semantic suggestions for no-results scenarios

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import logging
import re
from typing import List, Dict, Optional, Any
from pathlib import Path

from services.rag_system import RAGSystem
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchService:
    """
    Comprehensive search service for NCERT content.
    
    Provides semantic search with metadata extraction, result ranking,
    context display, and diagram integration.
    """
    
    def __init__(
        self,
        rag_system: RAGSystem,
        diagram_db_path: Optional[str] = None
    ):
        """
        Initialize the Search Service.
        
        Args:
            rag_system: RAG system for semantic search
            diagram_db_path: Path to diagram database (optional)
        """
        self.rag = rag_system
        
        # Initialize diagram search
        self.diagram_db_path = diagram_db_path or (Path(__file__).parent.parent / 'diagrams.db')
        self._init_diagram_search()
        
        logger.info("SearchService initialized successfully")
    
    def _init_diagram_search(self):
        """Initialize diagram search system."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from services.diagram_processor_final import DiagramPage
            
            self.diagram_engine = create_engine(f'sqlite:///{self.diagram_db_path}')
            Session = sessionmaker(bind=self.diagram_engine)
            self.diagram_session = Session()
            self.DiagramPage = DiagramPage
            
            logger.info(f"Diagram search initialized with database: {self.diagram_db_path}")
        except Exception as e:
            logger.warning(f"Could not initialize diagram search: {e}")
            self.diagram_session = None
            self.DiagramPage = None
    
    def search(
        self,
        query: str,
        subject: Optional[str] = None,
        class_level: Optional[int] = None,
        chapter: Optional[int] = None,
        top_k: int = 10,
        include_context: bool = True,
        include_diagrams: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive search across NCERT content.
        
        Args:
            query: Search query string
            subject: Optional subject filter (Physics, Chemistry, Mathematics, Biology)
            class_level: Optional class level filter (11 or 12)
            chapter: Optional chapter filter
            top_k: Number of results to return
            include_context: Whether to include surrounding context
            include_diagrams: Whether to search and include diagrams
        
        Returns:
            Dictionary containing:
                - results: List of search results with metadata
                - total: Total number of results
                - diagrams: List of matching diagrams (if include_diagrams=True)
                - suggestions: Semantic suggestions (if no results found)
                - query_info: Information about the query
        """
        try:
            logger.info(f"Searching for: '{query}' (filters: subject={subject}, class={class_level}, chapter={chapter})")
            
            # Validate query
            is_valid, error_msg = self._validate_query(query)
            if not is_valid:
                return self._format_error_response(error_msg)
            
            # Build filters
            filters = self._build_filters(subject, class_level, chapter)
            
            # Perform semantic search using RAG system
            raw_results = self.rag.retrieve(
                query=query,
                top_k=top_k,
                filters=filters if filters else None
            )
            
            # Check if no results found
            if not raw_results:
                return self._handle_no_results(query, filters)
            
            # Format results with metadata and context
            formatted_results = self._format_search_results(
                raw_results,
                query,
                include_context=include_context
            )
            
            # Search for matching diagrams
            diagrams = []
            if include_diagrams:
                diagrams = self._search_diagrams(query, formatted_results, filters)
            
            # Build response
            response = {
                'success': True,
                'results': formatted_results,
                'total': len(formatted_results),
                'diagrams': diagrams,
                'query_info': {
                    'query': query,
                    'filters': filters,
                    'has_filters': bool(filters),
                    'top_k': top_k
                }
            }
            
            logger.info(f"Search completed: {len(formatted_results)} results, {len(diagrams)} diagrams")
            return response
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return self._format_error_response(f"Search failed: {str(e)}")
    
    def _validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Validate search query.
        
        Args:
            query: Search query string
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Search query cannot be empty"
        
        if len(query) > 500:
            return False, "Search query is too long (maximum 500 characters)"
        
        if len(query.strip()) < 2:
            return False, "Search query is too short (minimum 2 characters)"
        
        return True, None
    
    def _build_filters(
        self,
        subject: Optional[str],
        class_level: Optional[int],
        chapter: Optional[int]
    ) -> Optional[Dict[str, str]]:
        """
        Build metadata filters for search.
        
        Args:
            subject: Subject filter
            class_level: Class level filter
            chapter: Chapter filter
        
        Returns:
            Dictionary of filters or None
        """
        filters = {}
        
        if subject:
            # Normalize subject name
            subject_normalized = subject.strip().title()
            filters['subject'] = subject_normalized
        
        if class_level is not None:
            filters['class_level'] = str(class_level)
        
        if chapter is not None:
            filters['chapter'] = str(chapter)
        
        return filters if filters else None
    
    def _format_search_results(
        self,
        raw_results: List[Dict],
        query: str,
        include_context: bool = True
    ) -> List[Dict]:
        """
        Format search results with complete metadata and context.
        
        Args:
            raw_results: Raw results from RAG system
            query: Original search query
            include_context: Whether to include surrounding context
        
        Returns:
            List of formatted search result dictionaries
        """
        formatted_results = []
        
        for result in raw_results:
            metadata = result.get('metadata', {})
            content = result.get('content', '')
            
            # Extract metadata with fallback parsing
            subject = metadata.get('subject')
            class_level = metadata.get('class_level') or metadata.get('class')
            chapter = metadata.get('chapter')
            page_number = metadata.get('page_number') or metadata.get('page')
            section = metadata.get('section', '')
            
            # If metadata is missing, try to extract from content or ID
            if not subject or not class_level or not chapter:
                extracted = self._extract_metadata_from_content(content, result.get('id', ''))
                subject = subject or extracted.get('subject', 'NCERT')
                class_level = class_level or extracted.get('class_level', 11)
                chapter = chapter or extracted.get('chapter', 1)
                page_number = page_number or extracted.get('page', 1)
            
            # Format result
            formatted_result = {
                'id': result.get('id', ''),
                'content': content,
                'metadata': {
                    'subject': subject,
                    'class': class_level,
                    'chapter': chapter,
                    'page': page_number,
                    'section': section
                },
                'relevance_score': result.get('relevance_score', 0.0),
                'rank': result.get('rank', 0),
                'highlight': self._highlight_query_terms(content, query)
            }
            
            # Add surrounding context if requested
            if include_context:
                formatted_result['context'] = self._get_surrounding_context(
                    content,
                    result.get('id', ''),
                    metadata
                )
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _extract_metadata_from_content(self, content: str, doc_id: str) -> Dict[str, Any]:
        """
        Extract metadata from content or document ID when metadata is missing.
        
        Args:
            content: Content text
            doc_id: Document ID
        
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Try to extract from document ID (e.g., "Biology_11_Ch13_p151")
        if doc_id:
            parts = doc_id.split('_')
            if len(parts) >= 3:
                metadata['subject'] = parts[0]
                if parts[1].isdigit():
                    metadata['class_level'] = int(parts[1])
                # Look for chapter
                for part in parts:
                    if part.startswith('Ch') and part[2:].isdigit():
                        metadata['chapter'] = int(part[2:])
                    elif part.startswith('p') and part[1:].isdigit():
                        metadata['page'] = int(part[1:])
        
        # Try to extract from content
        if not metadata.get('subject'):
            # Look for subject names in content
            subjects = ['Physics', 'Chemistry', 'Mathematics', 'Biology']
            for subj in subjects:
                if subj.lower() in content.lower()[:200]:  # Check first 200 chars
                    metadata['subject'] = subj
                    break
        
        # Look for chapter references in content
        if not metadata.get('chapter'):
            chapter_match = re.search(r'Chapter\s+(\d+)', content, re.IGNORECASE)
            if chapter_match:
                metadata['chapter'] = int(chapter_match.group(1))
        
        # Look for page references
        if not metadata.get('page'):
            page_match = re.search(r'Page\s+(\d+)', content, re.IGNORECASE)
            if page_match:
                metadata['page'] = int(page_match.group(1))
        
        return metadata
    
    def _highlight_query_terms(self, content: str, query: str) -> str:
        """
        Highlight query terms in content for display.
        
        Args:
            content: Content text
            query: Search query
        
        Returns:
            Content with highlighted terms (marked with <mark> tags)
        """
        # Extract query terms
        query_terms = query.lower().split()
        
        # Create highlighted version
        highlighted = content
        for term in query_terms:
            if len(term) > 2:  # Only highlight terms longer than 2 characters
                # Case-insensitive replacement with <mark> tags
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", highlighted)
        
        return highlighted
    
    def _get_surrounding_context(
        self,
        content: str,
        doc_id: str,
        metadata: Dict
    ) -> Dict[str, str]:
        """
        Get surrounding context paragraphs for a search result.
        
        Args:
            content: Main content text
            doc_id: Document ID
            metadata: Document metadata
        
        Returns:
            Dictionary with before and after context
        """
        # For now, return the content itself as context
        # In a full implementation, this would query adjacent chunks
        # from the vector store based on doc_id and metadata
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # Return first and last sentences as context
        context = {
            'before': sentences[0].strip() if sentences else '',
            'after': sentences[-1].strip() if len(sentences) > 1 else '',
            'full': content
        }
        
        return context
    
    def _search_diagrams(
        self,
        query: str,
        search_results: List[Dict],
        filters: Optional[Dict[str, str]]
    ) -> List[Dict]:
        """
        Search for diagrams matching the query and search results.
        
        Args:
            query: Search query
            search_results: Text search results
            filters: Applied filters
        
        Returns:
            List of matching diagrams
        """
        if not self.diagram_session or not self.DiagramPage:
            logger.warning("Diagram search not available")
            return []
        
        try:
            diagrams = []
            
            # Strategy 1: Search diagrams from chapters in search results
            chapters_found = set()
            for result in search_results[:5]:  # Check top 5 results
                metadata = result.get('metadata', {})
                subject = metadata.get('subject')
                class_level = metadata.get('class')
                chapter = metadata.get('chapter')
                
                if subject and class_level and chapter:
                    try:
                        class_int = int(class_level)
                        chapter_int = int(chapter)
                        chapters_found.add((subject, class_int, chapter_int))
                    except (ValueError, TypeError):
                        continue
            
            # Query diagrams from these chapters
            for subject, class_level, chapter in chapters_found:
                chapter_diagrams = self.diagram_session.query(self.DiagramPage).filter_by(
                    subject=subject,
                    class_level=class_level,
                    chapter=chapter
                ).limit(3).all()
                
                for diag in chapter_diagrams:
                    diagrams.append(diag.to_dict())
            
            # Strategy 2: Search by caption keywords
            if len(diagrams) < 5:
                caption_diagrams = self._search_diagrams_by_caption(query, filters)
                diagrams.extend(caption_diagrams)
            
            # Remove duplicates and limit results
            seen_ids = set()
            unique_diagrams = []
            for diag in diagrams:
                diag_id = f"{diag.get('subject')}_{diag.get('class_level')}_{diag.get('chapter')}_{diag.get('page')}"
                if diag_id not in seen_ids:
                    seen_ids.add(diag_id)
                    unique_diagrams.append(diag)
                    if len(unique_diagrams) >= 5:
                        break
            
            logger.info(f"Found {len(unique_diagrams)} matching diagrams")
            return unique_diagrams
            
        except Exception as e:
            logger.error(f"Error searching diagrams: {e}")
            return []
    
    def _search_diagrams_by_caption(
        self,
        query: str,
        filters: Optional[Dict[str, str]],
        max_results: int = 5
    ) -> List[Dict]:
        """
        Search diagrams by matching keywords in captions.
        
        Args:
            query: Search query
            filters: Applied filters
            max_results: Maximum number of results
        
        Returns:
            List of matching diagrams
        """
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return []
            
            # Build base query
            base_query = self.diagram_session.query(self.DiagramPage)
            
            # Apply filters if provided
            if filters:
                if 'subject' in filters:
                    base_query = base_query.filter_by(subject=filters['subject'])
                if 'class_level' in filters:
                    base_query = base_query.filter_by(class_level=int(filters['class_level']))
                if 'chapter' in filters:
                    base_query = base_query.filter_by(chapter=int(filters['chapter']))
            
            # Search in captions
            diagrams = []
            all_diagrams = base_query.all()
            
            for diag in all_diagrams:
                captions_str = diag.captions.lower() if diag.captions else ""
                
                # Check if any keyword matches
                for keyword in keywords:
                    if keyword.lower() in captions_str:
                        diagrams.append(diag.to_dict())
                        break
                
                if len(diagrams) >= max_results:
                    break
            
            return diagrams
            
        except Exception as e:
            logger.error(f"Error in caption search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.
        
        Args:
            query: Search query
        
        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where',
            'explain', 'describe', 'tell', 'me', 'about', 'can', 'you',
            'please', 'in', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
            'by', 'as', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'shall'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Return top 10 keywords
    
    def _handle_no_results(
        self,
        query: str,
        filters: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Handle scenarios where no search results are found.
        
        Args:
            query: Search query
            filters: Applied filters
        
        Returns:
            Response with suggestions
        """
        logger.info(f"No results found for query: '{query}'")
        
        # Generate semantic suggestions
        suggestions = self._generate_suggestions(query, filters)
        
        return {
            'success': True,
            'results': [],
            'total': 0,
            'diagrams': [],
            'no_results': True,
            'message': self._get_no_results_message(filters),
            'suggestions': suggestions,
            'query_info': {
                'query': query,
                'filters': filters,
                'has_filters': bool(filters)
            }
        }
    
    def _get_no_results_message(self, filters: Optional[Dict[str, str]]) -> str:
        """
        Get appropriate message for no results scenario.
        
        Args:
            filters: Applied filters
        
        Returns:
            Message string
        """
        if filters:
            filter_desc = ", ".join([f"{k}={v}" for k, v in filters.items()])
            return (
                f"No results found with the applied filters ({filter_desc}). "
                "Try removing some filters or using different search terms."
            )
        else:
            return (
                "No results found for your search. "
                "Try using different keywords or check the suggestions below."
            )
    
    def _generate_suggestions(
        self,
        query: str,
        filters: Optional[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Generate semantic suggestions for queries with no results.
        
        Args:
            query: Search query
            filters: Applied filters
        
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        
        # Suggestion 1: Try without filters
        if filters:
            suggestions.append({
                'type': 'remove_filters',
                'text': 'Try searching without filters',
                'action': 'remove_all_filters'
            })
        
        # Suggestion 2: Related topics by subject
        subject_topics = {
            'Physics': [
                'mechanics', 'thermodynamics', 'electromagnetism', 'optics',
                'modern physics', 'waves', 'gravitation'
            ],
            'Chemistry': [
                'atomic structure', 'chemical bonding', 'thermodynamics',
                'equilibrium', 'organic chemistry', 'coordination compounds'
            ],
            'Mathematics': [
                'calculus', 'algebra', 'trigonometry', 'coordinate geometry',
                'vectors', 'probability', 'statistics'
            ],
            'Biology': [
                'cell biology', 'genetics', 'evolution', 'ecology',
                'human physiology', 'plant physiology', 'reproduction'
            ]
        }
        
        # If subject filter is applied, suggest topics from that subject
        if filters and 'subject' in filters:
            subject = filters['subject']
            if subject in subject_topics:
                for topic in subject_topics[subject][:3]:
                    suggestions.append({
                        'type': 'related_topic',
                        'text': f'Search for "{topic}"',
                        'query': topic
                    })
        else:
            # Suggest general topics
            suggestions.append({
                'type': 'general',
                'text': 'Try searching for specific NCERT chapter topics',
                'action': 'browse_chapters'
            })
        
        # Suggestion 3: Use simpler terms
        if len(query.split()) > 3:
            suggestions.append({
                'type': 'simplify',
                'text': 'Try using simpler or fewer keywords',
                'action': 'simplify_query'
            })
        
        # Suggestion 4: Check spelling
        suggestions.append({
            'type': 'spelling',
            'text': 'Check your spelling and try again',
            'action': 'check_spelling'
        })
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Format error response.
        
        Args:
            error_message: Error message
        
        Returns:
            Formatted error response
        """
        return {
            'success': False,
            'error': error_message,
            'results': [],
            'total': 0,
            'diagrams': [],
            'suggestions': []
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search service.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'rag_stats': self.rag.get_stats(),
            'diagram_db_path': str(self.diagram_db_path)
        }
        
        # Add diagram count if available
        if self.diagram_session and self.DiagramPage:
            try:
                diagram_count = self.diagram_session.query(self.DiagramPage).count()
                stats['total_diagrams'] = diagram_count
            except Exception:
                stats['total_diagrams'] = 0
        
        return stats


def main():
    """
    Test the Search Service with sample queries.
    """
    print("="*70)
    print("Search Service Test")
    print("="*70)
    
    # Initialize components
    print("\nInitializing search service...")
    
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Initialize Search Service
        search_service = SearchService(rag)
        
        # Display stats
        stats = search_service.get_stats()
        print(f"\nSearch Service Stats:")
        print(f"  Total documents: {stats['rag_stats'].get('total_documents', 0)}")
        print(f"  Total diagrams: {stats.get('total_diagrams', 0)}")
        
        # Test searches
        test_searches = [
            {
                'query': 'Newton laws',
                'subject': None,
                'class_level': None,
                'chapter': None
            },
            {
                'query': 'photosynthesis',
                'subject': 'Biology',
                'class_level': None,
                'chapter': None
            },
            {
                'query': 'derivatives',
                'subject': 'Mathematics',
                'class_level': 12,
                'chapter': None
            },
            {
                'query': 'nonexistent topic xyz',
                'subject': None,
                'class_level': None,
                'chapter': None
            }
        ]
        
        print("\n" + "="*70)
        print("Test Searches")
        print("="*70)
        
        for test in test_searches:
            print(f"\nQuery: '{test['query']}'")
            if test['subject'] or test['class_level'] or test['chapter']:
                filters = []
                if test['subject']:
                    filters.append(f"subject={test['subject']}")
                if test['class_level']:
                    filters.append(f"class={test['class_level']}")
                if test['chapter']:
                    filters.append(f"chapter={test['chapter']}")
                print(f"Filters: {', '.join(filters)}")
            print("-" * 70)
            
            # Perform search
            response = search_service.search(
                query=test['query'],
                subject=test['subject'],
                class_level=test['class_level'],
                chapter=test['chapter'],
                top_k=5,
                include_context=True,
                include_diagrams=True
            )
            
            if response.get('success'):
                print(f"Status: SUCCESS")
                print(f"Results: {response['total']}")
                print(f"Diagrams: {len(response.get('diagrams', []))}")
                
                if response.get('no_results'):
                    print(f"Message: {response.get('message')}")
                    print(f"Suggestions: {len(response.get('suggestions', []))}")
                    if response.get('suggestions'):
                        for sug in response['suggestions'][:3]:
                            print(f"  - {sug.get('text')}")
                else:
                    # Show top result
                    if response['results']:
                        top_result = response['results'][0]
                        print(f"\nTop Result:")
                        print(f"  Subject: {top_result['metadata']['subject']}")
                        print(f"  Class: {top_result['metadata']['class']}")
                        print(f"  Chapter: {top_result['metadata']['chapter']}")
                        print(f"  Page: {top_result['metadata']['page']}")
                        print(f"  Relevance: {top_result['relevance_score']:.3f}")
                        print(f"  Content preview: {top_result['content'][:150]}...")
            else:
                print(f"Status: ERROR")
                print(f"Error: {response.get('error')}")
        
        print("\n" + "="*70)
        print("Test Complete")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
