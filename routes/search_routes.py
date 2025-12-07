"""
Search Routes for GuruAI - API endpoints for search functionality.

This module provides REST API endpoints for searching NCERT content
with filtering, metadata extraction, and diagram integration.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import logging
from flask import Blueprint, request, jsonify
from typing import Optional

from services.search_service import SearchService
from services.rag_system import RAGSystem
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create blueprint
search_bp = Blueprint('search', __name__, url_prefix='/api/search')

# Global search service instance
_search_service: Optional[SearchService] = None


def init_search_service():
    """
    Initialize the search service.
    
    This should be called during application startup.
    """
    global _search_service
    
    try:
        logger.info("Initializing search service...")
        
        # Initialize RAG system
        rag_system = RAGSystem()
        
        # Initialize search service
        _search_service = SearchService(rag_system)
        
        logger.info("Search service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize search service: {e}", exc_info=True)
        raise


def get_search_service() -> SearchService:
    """
    Get the search service instance.
    
    Returns:
        SearchService instance
    
    Raises:
        RuntimeError: If search service is not initialized
    """
    if _search_service is None:
        raise RuntimeError("Search service not initialized. Call init_search_service() first.")
    return _search_service


@search_bp.route('', methods=['GET'])
def search():
    """
    Search NCERT content with optional filters.
    
    Query Parameters:
        q (str): Search query (required)
        subject (str): Subject filter (optional)
        class (int): Class level filter (optional)
        chapter (int): Chapter filter (optional)
        top_k (int): Number of results to return (default: 10)
        include_context (bool): Include surrounding context (default: true)
        include_diagrams (bool): Include matching diagrams (default: true)
    
    Returns:
        JSON response with search results, diagrams, and metadata
    
    Example:
        GET /api/search?q=Newton+laws&subject=Physics&class=11
    """
    try:
        # Get query parameters
        query = request.args.get('q', '').strip()
        subject = request.args.get('subject', None)
        class_level_str = request.args.get('class', None)
        chapter_str = request.args.get('chapter', None)
        top_k_str = request.args.get('top_k', '10')
        include_context_str = request.args.get('include_context', 'true')
        include_diagrams_str = request.args.get('include_diagrams', 'true')
        
        # Validate required parameters
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter "q" is required'
            }), 400
        
        # Parse optional parameters
        class_level = None
        if class_level_str:
            try:
                class_level = int(class_level_str)
                if class_level not in [11, 12]:
                    return jsonify({
                        'success': False,
                        'error': 'Class level must be 11 or 12'
                    }), 400
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid class level parameter'
                }), 400
        
        chapter = None
        if chapter_str:
            try:
                chapter = int(chapter_str)
                if chapter < 1:
                    return jsonify({
                        'success': False,
                        'error': 'Chapter must be a positive integer'
                    }), 400
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid chapter parameter'
                }), 400
        
        try:
            top_k = int(top_k_str)
            if top_k < 1 or top_k > 50:
                top_k = 10  # Default to 10 if out of range
        except ValueError:
            top_k = 10
        
        include_context = include_context_str.lower() in ['true', '1', 'yes']
        include_diagrams = include_diagrams_str.lower() in ['true', '1', 'yes']
        
        # Get search service
        search_service = get_search_service()
        
        # Perform search
        logger.info(f"Search request: query='{query}', subject={subject}, class={class_level}, chapter={chapter}")
        
        response = search_service.search(
            query=query,
            subject=subject,
            class_level=class_level,
            chapter=chapter,
            top_k=top_k,
            include_context=include_context,
            include_diagrams=include_diagrams
        )
        
        # Return response
        if response.get('success'):
            return jsonify(response), 200
        else:
            return jsonify(response), 400
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@search_bp.route('/autocomplete', methods=['GET'])
def autocomplete():
    """
    Get autocomplete suggestions for search queries.
    
    Query Parameters:
        q (str): Partial search query (required, min 2 characters)
        limit (int): Maximum number of suggestions (optional, default: 5)
    
    Returns:
        JSON response with suggestions:
        {
            "success": true,
            "suggestions": [array of suggestion strings],
            "query": "original query"
        }
    
    Status Codes:
        200: Success
        400: Bad request (missing or invalid query)
        500: Internal server error
    """
    try:
        # Get query parameter
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', 5, type=int)
        
        # Validate query
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter "q" is required'
            }), 400
        
        if len(query) < 2:
            return jsonify({
                'success': False,
                'error': 'Query must be at least 2 characters'
            }), 400
        
        if limit < 1 or limit > 20:
            limit = 5  # Default to 5
        
        # Get autocomplete suggestions
        suggestions = get_autocomplete_suggestions(query, limit)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'query': query
        }), 200
        
    except Exception as e:
        logger.error(f"Error in autocomplete endpoint: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500


def get_autocomplete_suggestions(query: str, limit: int = 5):
    """
    Generate autocomplete suggestions based on query.
    
    Args:
        query: Partial search query
        limit: Maximum number of suggestions
    
    Returns:
        List of suggestion strings
    """
    # Common NCERT topics and concepts
    common_topics = [
        # Physics
        "Newton's Laws of Motion",
        "Photosynthesis",
        "Photoelectric Effect",
        "Thermodynamics",
        "Electromagnetic Induction",
        "Gravitation",
        "Kinetic Theory of Gases",
        "Wave Optics",
        "Atomic Structure",
        "Nuclear Physics",
        
        # Chemistry
        "Chemical Bonding",
        "Periodic Table",
        "Electrochemistry",
        "Organic Chemistry",
        "Redox Reactions",
        "Chemical Equilibrium",
        "Thermochemistry",
        "Solutions",
        "Surface Chemistry",
        "Coordination Compounds",
        
        # Mathematics
        "Differentiation",
        "Integration",
        "Trigonometry",
        "Probability",
        "Matrices",
        "Determinants",
        "Vectors",
        "Complex Numbers",
        "Binomial Theorem",
        "Sequences and Series",
        
        # Biology
        "Cell Structure",
        "DNA Replication",
        "Protein Synthesis",
        "Respiration",
        "Photosynthesis in Plants",
        "Human Reproduction",
        "Genetics",
        "Evolution",
        "Ecosystem",
        "Biodiversity"
    ]
    
    # Filter topics that match the query (case-insensitive)
    query_lower = query.lower()
    suggestions = [
        topic for topic in common_topics
        if query_lower in topic.lower()
    ]
    
    # If no matches, suggest related terms
    if not suggestions:
        # Try to suggest based on first few characters
        suggestions = [
            topic for topic in common_topics
            if topic.lower().startswith(query_lower[:3])
        ]
    
    # Limit results
    return suggestions[:limit]


@search_bp.route('/stats', methods=['GET'])
def search_stats():
    """
    Get search service statistics.
    
    Returns:
        JSON response with statistics about the search service
    
    Example:
        GET /api/search/stats
    """
    try:
        search_service = get_search_service()
        stats = search_service.get_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting search stats: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to get statistics',
            'details': str(e)
        }), 500


@search_bp.route('/subjects', methods=['GET'])
def get_subjects():
    """
    Get list of available subjects.
    
    Returns:
        JSON response with list of subjects
    
    Example:
        GET /api/search/subjects
    """
    try:
        subjects = ['Physics', 'Chemistry', 'Mathematics', 'Biology']
        
        return jsonify({
            'success': True,
            'subjects': subjects
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting subjects: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to get subjects',
            'details': str(e)
        }), 500


@search_bp.route('/filters', methods=['GET'])
def get_available_filters():
    """
    Get available filter options.
    
    Returns:
        JSON response with available filter options
    
    Example:
        GET /api/search/filters
    """
    try:
        filters = {
            'subjects': ['Physics', 'Chemistry', 'Mathematics', 'Biology'],
            'classes': [11, 12],
            'chapters': {
                'Physics': list(range(1, 16)),
                'Chemistry': list(range(1, 17)),
                'Mathematics': list(range(1, 15)),
                'Biology': list(range(1, 23))
            }
        }
        
        return jsonify({
            'success': True,
            'filters': filters
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting filters: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to get filters',
            'details': str(e)
        }), 500


@search_bp.route('/health', methods=['GET'])
def search_health():
    """
    Health check endpoint for search service.
    
    Returns:
        JSON response with health status
    
    Example:
        GET /api/search/health
    """
    try:
        search_service = get_search_service()
        stats = search_service.get_stats()
        
        # Check if RAG system has documents
        total_docs = stats.get('rag_stats', {}).get('total_documents', 0)
        
        if total_docs > 0:
            status = 'healthy'
            message = f'Search service is operational with {total_docs} documents'
        else:
            status = 'degraded'
            message = 'Search service is running but no documents are indexed'
        
        return jsonify({
            'success': True,
            'status': status,
            'message': message,
            'stats': {
                'total_documents': total_docs,
                'total_diagrams': stats.get('total_diagrams', 0)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# Error handlers
@search_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@search_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


@search_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
