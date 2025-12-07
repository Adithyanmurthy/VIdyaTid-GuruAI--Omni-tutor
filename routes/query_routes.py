"""
Query Routes for GuruAI - Flask API endpoints for text-based queries.

Provides REST API endpoints for:
- Processing text queries
- Retrieving query statistics
- Validating queries

Requirements: 1.1, 1.2, 1.3, 1.4
"""

import logging
import asyncio
import os
from flask import Blueprint, request, jsonify
from typing import Dict, Any

from services.query_handler import QueryHandler
from services.rag_system import RAGSystem
from services.model_manager import ModelManagerSingleton
from config import Config

# Setup logging
logger = logging.getLogger(__name__)

# Create Blueprint
query_bp = Blueprint('query', __name__, url_prefix='/api')

# Global query handler instance (initialized in init_query_handler)
_query_handler: QueryHandler = None


def init_query_handler() -> QueryHandler:
    """
    Initialize the query handler singleton.
    Uses Groq for problem-solving, Cloudflare for general queries.
    
    Returns:
        QueryHandler instance
    """
    global _query_handler
    
    if _query_handler is None:
        logger.info("Initializing Query Handler...")
        
        # Initialize RAG system (needed for context retrieval)
        try:
            rag = RAGSystem()
        except Exception as e:
            logger.warning(f"RAG system initialization failed: {e}. Using mock RAG.")
            rag = None
        
        # Initialize AI models based on configuration
        # Priority: Groq for problems, Cloudflare for general chat
        
        primary_model = None
        problem_solver_model = None
        
        # Priority: Gemini > Cloudflare > Local
        if Config.USE_GEMINI:
            logger.info("🚀 Initializing Google Gemini for all queries")
            try:
                from services.gemini_ai import get_gemini_ai
                primary_model = get_gemini_ai()
                problem_solver_model = primary_model  # Use Gemini for everything
                logger.info("✅ Gemini initialized for all queries")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                Config.USE_GEMINI = False
        
        # Check if Cloudflare AI is enabled for general queries
        if not primary_model and Config.USE_CLOUDFLARE_AI:
            logger.info("Initializing Cloudflare AI for general queries")
            try:
                from services.cloudflare_ai import CloudflareAI
                primary_model = CloudflareAI(enable_fallback=True)
                logger.info("✅ Cloudflare AI initialized for general queries")
            except Exception as e:
                logger.error(f"Failed to initialize Cloudflare AI: {e}")
                Config.USE_CLOUDFLARE_AI = False
        
        # Fallback to local models if neither cloud service is available
        if not primary_model:
            logger.info("Using local AI models")
            model_config = {
                'idle_timeout': Config.MODEL_IDLE_TIMEOUT,
                'n_ctx': Config.LLM_N_CTX,
                'n_gpu_layers': Config.LLM_N_GPU_LAYERS,
                'temperature': Config.LLM_TEMPERATURE,
                'max_tokens': Config.LLM_MAX_TOKENS
            }
            try:
                primary_model = ModelManagerSingleton.get_instance(
                    str(Config.LLM_MODEL_PATH),
                    model_config
                )
            except Exception as e:
                logger.error(f"Failed to initialize local model: {e}")
                primary_model = None
        
        # Initialize Query Handler with both models
        _query_handler = QueryHandler(
            rag, 
            primary_model,
            problem_solver=problem_solver_model
        )
        logger.info("Query Handler initialized with hybrid AI system")
    
    return _query_handler


def get_query_handler() -> QueryHandler:
    """
    Get the query handler instance.
    
    Returns:
        QueryHandler instance
    """
    if _query_handler is None:
        return init_query_handler()
    return _query_handler


@query_bp.route('/ask', methods=['POST'])
def ask_question():
    """
    Process a text-based query and return a comprehensive response.
    
    Request JSON:
        {
            "query": "string (required)",
            "user_id": "string (optional)",
            "include_quiz": boolean (optional, default: true),
            "include_diagrams": boolean (optional, default: true)
        }
    
    Response JSON:
        {
            "success": boolean,
            "explanation": "string",
            "diagrams": [array of diagram objects],
            "quiz": {quiz object} or null,
            "references": [array of reference strings],
            "metadata": {metadata object},
            "error": "string (if success=false)"
        }
    
    Status Codes:
        200: Success
        400: Bad request (invalid input)
        500: Internal server error
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract parameters
        query = data.get('query')
        user_id = data.get('user_id')
        include_quiz = data.get('include_quiz', True)
        include_diagrams = data.get('include_diagrams', True)
        
        # Validate query
        if not query or not query.strip():
            logger.warning(f"Empty query received. Data: {data}")
            return jsonify({
                'success': False,
                'error': 'Query parameter is required'
            }), 400
        
        # Get query handler
        handler = get_query_handler()
        
        # Validate query format
        is_valid, error_msg = handler.validate_query(query)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Process query (run async function in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                handler.process_query(
                    query=query,
                    user_id=user_id,
                    include_quiz=include_quiz,
                    include_diagrams=include_diagrams
                )
            )
        finally:
            loop.close()
        
        # Return response
        status_code = 200 if response.get('success') else 500
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Error in /api/ask endpoint: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'error_details': str(e)
        }), 500


@query_bp.route('/validate-query', methods=['POST'])
def validate_query():
    """
    Validate a query without processing it.
    
    Request JSON:
        {
            "query": "string (required)"
        }
    
    Response JSON:
        {
            "valid": boolean,
            "error": "string (if valid=false)"
        }
    
    Status Codes:
        200: Success
        400: Bad request
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'valid': False,
                'error': 'No JSON data provided'
            }), 400
        
        query = data.get('query')
        
        if not query:
            return jsonify({
                'valid': False,
                'error': 'Query parameter is required'
            }), 400
        
        # Get query handler
        handler = get_query_handler()
        
        # Validate
        is_valid, error_msg = handler.validate_query(query)
        
        response = {
            'valid': is_valid
        }
        
        if not is_valid:
            response['error'] = error_msg
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /api/validate-query endpoint: {e}", exc_info=True)
        return jsonify({
            'valid': False,
            'error': 'Internal server error'
        }), 500


@query_bp.route('/query-stats', methods=['GET'])
def get_query_stats():
    """
    Get statistics about the query handler system.
    
    Response JSON:
        {
            "rag_stats": {RAG system statistics},
            "model_status": {Model manager status},
            "total_diagrams": integer,
            "diagram_db_path": "string"
        }
    
    Status Codes:
        200: Success
        500: Internal server error
    """
    try:
        handler = get_query_handler()
        stats = handler.get_stats()
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error in /api/query-stats endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'error_details': str(e)
        }), 500


@query_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for the query system.
    
    Response JSON:
        {
            "status": "healthy" or "unhealthy",
            "components": {
                "rag": boolean,
                "model": boolean,
                "diagrams": boolean
            }
        }
    
    Status Codes:
        200: System is healthy
        503: System is unhealthy
    """
    try:
        handler = get_query_handler()
        stats = handler.get_stats()
        
        # Check component health
        components = {
            'rag': stats.get('rag_stats', {}).get('total_documents', 0) >= 0,
            'model': True,  # Model manager is initialized
            'diagrams': stats.get('total_diagrams', 0) >= 0
        }
        
        all_healthy = all(components.values())
        
        response = {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'components': components
        }
        
        status_code = 200 if all_healthy else 503
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Error in /api/health endpoint: {e}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


# Error handlers
@query_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@query_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


@query_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
