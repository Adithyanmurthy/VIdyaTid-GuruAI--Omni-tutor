"""
Problem Solving Routes for GuruAI - Flask API endpoints for image-based problem solving.

Provides REST API endpoints for:
- Solving problems from uploaded images
- Validating images before processing
- Getting problem solving statistics

Requirements: 2.4, 2.5, 6.1, 6.2
"""

import logging
import asyncio
import os
from flask import Blueprint, request, jsonify
from typing import Dict, Any

from services.problem_solver import ProblemSolver
from services.image_processor import ImageProcessor
from services.model_manager import ModelManagerSingleton
from services.rag_system import RAGSystem
from config import Config

# Setup logging
logger = logging.getLogger(__name__)

# Create Blueprint
problem_bp = Blueprint('problem', __name__, url_prefix='/api')

# Global problem solver instance (initialized in init_problem_solver)
_problem_solver: ProblemSolver = None


def init_problem_solver() -> ProblemSolver:
    """
    Initialize the problem solver singleton.
    
    Returns:
        ProblemSolver instance
    """
    global _problem_solver
    
    if _problem_solver is None:
        logger.info("Initializing Problem Solver...")
        
        # Initialize Image Processor
        image_processor = ImageProcessor()
        
        # Initialize RAG system
        rag = RAGSystem()
        
        # Initialize AI model (Priority: Gemini > Cloudflare > Local)
        if Config.USE_GEMINI:
            logger.info("🚀 Using Google Gemini for problem solving")
            try:
                from services.gemini_ai import get_gemini_ai
                model_manager = get_gemini_ai()
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                logger.info("Falling back to Cloudflare")
                from services.cloudflare_ai import get_cloudflare_ai
                model_manager = get_cloudflare_ai()
        elif Config.USE_CLOUDFLARE_AI:
            logger.info("Using Cloudflare AI for problem solving")
            from services.cloudflare_ai import get_cloudflare_ai
            model_manager = get_cloudflare_ai()
        else:
            logger.info("Using local model for problem solving")
            # Initialize Model Manager
            model_config = {
                'idle_timeout': Config.MODEL_IDLE_TIMEOUT,
                'n_ctx': Config.LLM_N_CTX,
                'n_gpu_layers': Config.LLM_N_GPU_LAYERS,
                'temperature': Config.LLM_TEMPERATURE,
                'max_tokens': Config.LLM_MAX_TOKENS
            }
            model_manager = ModelManagerSingleton.get_instance(
                str(Config.LLM_MODEL_PATH),
                model_config
            )
        
        # Initialize Problem Solver
        _problem_solver = ProblemSolver(image_processor, model_manager, rag)
        logger.info("Problem Solver initialized successfully")
    
    return _problem_solver


def get_problem_solver() -> ProblemSolver:
    """
    Get the problem solver instance.
    
    Returns:
        ProblemSolver instance
    """
    if _problem_solver is None:
        return init_problem_solver()
    return _problem_solver


@problem_bp.route('/solve-image', methods=['POST'])
def solve_image():
    """
    Process an uploaded image containing a problem and generate a solution.
    
    Request:
        multipart/form-data with:
            - image: Image file (required)
            - user_id: User identifier (optional)
    
    Response JSON:
        {
            "success": boolean,
            "problem_text": "string",
            "solution": "string",
            "steps": [array of step objects],
            "formulas": [array of formula objects],
            "diagrams": [array of diagram objects],
            "references": [array of reference strings],
            "confidence": "string (low/medium/high)",
            "metadata": {metadata object},
            "error": "string (if success=false)",
            "suggestion": "string (if error occurred)"
        }
    
    Status Codes:
        200: Success
        400: Bad request (invalid input)
        500: Internal server error
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'suggestion': 'Please upload an image file containing the problem.'
            }), 400
        
        image_file = request.files['image']
        
        # Check if file is empty
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename',
                'suggestion': 'Please select a valid image file.'
            }), 400
        
        # Read image data
        image_data = image_file.read()
        
        # Get optional user_id
        user_id = request.form.get('user_id')
        
        # Get problem solver
        solver = get_problem_solver()
        
        # Validate image
        is_valid, error_msg = solver.validate_image_for_problem(image_data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Invalid image',
                'error_details': error_msg,
                'suggestion': 'Please upload a clearer image with better quality.'
            }), 400
        
        # Process image and solve problem (run async function in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                solver.solve_problem_from_image(
                    image_data=image_data,
                    user_id=user_id
                )
            )
        finally:
            loop.close()
        
        # Return response
        status_code = 200 if response.get('success') else 400
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Error in /api/solve-image endpoint: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'error_details': str(e),
            'suggestion': 'Please try again or contact support if the issue persists.'
        }), 500


@problem_bp.route('/validate-image', methods=['POST'])
def validate_image():
    """
    Validate an image before processing for problem solving.
    
    Request:
        multipart/form-data with:
            - image: Image file (required)
    
    Response JSON:
        {
            "valid": boolean,
            "error": "string (if valid=false)",
            "details": {
                "file_size": integer (bytes),
                "dimensions": [width, height] (if valid)
            }
        }
    
    Status Codes:
        200: Success
        400: Bad request
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'valid': False,
                'error': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        
        # Check if file is empty
        if image_file.filename == '':
            return jsonify({
                'valid': False,
                'error': 'Empty filename'
            }), 400
        
        # Read image data
        image_data = image_file.read()
        
        # Get problem solver
        solver = get_problem_solver()
        
        # Validate
        is_valid, error_msg = solver.validate_image_for_problem(image_data)
        
        response = {
            'valid': is_valid,
            'details': {
                'file_size': len(image_data)
            }
        }
        
        if not is_valid:
            response['error'] = error_msg
        else:
            # Try to get image dimensions
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_data))
                response['details']['dimensions'] = list(img.size)
            except Exception:
                pass
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /api/validate-image endpoint: {e}", exc_info=True)
        return jsonify({
            'valid': False,
            'error': 'Internal server error',
            'error_details': str(e)
        }), 500


@problem_bp.route('/problem-stats', methods=['GET'])
def get_problem_stats():
    """
    Get statistics about the problem solving system.
    
    Response JSON:
        {
            "image_processor_available": boolean,
            "model_status": {Model manager status},
            "rag_available": boolean
        }
    
    Status Codes:
        200: Success
        500: Internal server error
    """
    try:
        solver = get_problem_solver()
        
        # Get model status
        model_status = solver.llm.get_status()
        
        # Get RAG stats
        rag_stats = solver.rag.get_stats()
        
        stats = {
            'image_processor_available': solver.image_processor is not None,
            'model_status': model_status,
            'rag_available': rag_stats.get('total_documents', 0) > 0,
            'rag_stats': rag_stats
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error in /api/problem-stats endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'error_details': str(e)
        }), 500


# Error handlers
@problem_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@problem_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


@problem_bp.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 errors (file too large)."""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'suggestion': 'Please upload an image smaller than 10MB.'
    }), 413


@problem_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
