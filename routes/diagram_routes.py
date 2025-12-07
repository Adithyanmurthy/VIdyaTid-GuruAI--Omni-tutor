"""
Diagram Display Routes for GuruAI

Provides API endpoints for:
- Retrieving diagrams by ID or query
- Getting diagram metadata
- Serving diagram files
- Getting labeled part explanations

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import logging
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path

from services.diagram_display import DiagramDisplayService

logger = logging.getLogger(__name__)

# Create blueprint
diagram_bp = Blueprint('diagram', __name__, url_prefix='/api/diagrams')

# Global service instance
diagram_service = None


def init_diagram_service():
    """Initialize diagram display service."""
    global diagram_service
    try:
        diagram_service = DiagramDisplayService()
        logger.info("Diagram display service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize diagram service: {e}")
        raise


@diagram_bp.route('/by-id/<page_id>', methods=['GET'])
def get_diagram_by_id(page_id):
    """
    Get diagram by page ID.
    
    GET /api/diagrams/by-id/<page_id>
    
    Returns:
        JSON with diagram metadata and file path
    """
    try:
        if not diagram_service:
            return jsonify({"error": "Diagram service not initialized"}), 500
        
        diagram = diagram_service.get_diagram_by_id(page_id)
        
        if not diagram:
            return jsonify({"error": "Diagram not found"}), 404
        
        return jsonify(diagram), 200
        
    except Exception as e:
        logger.error(f"Error retrieving diagram {page_id}: {e}")
        return jsonify({"error": str(e)}), 500


@diagram_bp.route('/by-chapter', methods=['GET'])
def get_diagrams_by_chapter():
    """
    Get all diagrams from a specific chapter.
    
    GET /api/diagrams/by-chapter?subject=Physics&class=11&chapter=1
    
    Query Parameters:
        subject: Subject name (Physics, Chemistry, Mathematics, Biology)
        class: Class level (11 or 12)
        chapter: Chapter number
    
    Returns:
        JSON array of diagram metadata
    """
    try:
        if not diagram_service:
            return jsonify({"error": "Diagram service not initialized"}), 500
        
        subject = request.args.get('subject')
        class_level = request.args.get('class', type=int)
        chapter = request.args.get('chapter', type=int)
        
        if not all([subject, class_level, chapter]):
            return jsonify({"error": "Missing required parameters: subject, class, chapter"}), 400
        
        diagrams = diagram_service.get_diagrams_by_chapter(subject, class_level, chapter)
        
        return jsonify({
            "count": len(diagrams),
            "diagrams": diagrams
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving diagrams by chapter: {e}")
        return jsonify({"error": str(e)}), 500


@diagram_bp.route('/by-figure', methods=['GET'])
def get_diagram_by_figure():
    """
    Get diagram containing a specific figure number.
    
    GET /api/diagrams/by-figure?subject=Physics&class=11&chapter=1&figure=1.1
    
    Query Parameters:
        subject: Subject name
        class: Class level (11 or 12)
        chapter: Chapter number
        figure: Figure number (e.g., "1.1", "2.5")
    
    Returns:
        JSON with diagram metadata
    """
    try:
        if not diagram_service:
            return jsonify({"error": "Diagram service not initialized"}), 500
        
        subject = request.args.get('subject')
        class_level = request.args.get('class', type=int)
        chapter = request.args.get('chapter', type=int)
        figure_number = request.args.get('figure')
        
        if not all([subject, class_level, chapter, figure_number]):
            return jsonify({"error": "Missing required parameters: subject, class, chapter, figure"}), 400
        
        diagram = diagram_service.get_diagrams_by_figure(subject, class_level, chapter, figure_number)
        
        if not diagram:
            return jsonify({"error": "Figure not found"}), 404
        
        return jsonify(diagram), 200
        
    except Exception as e:
        logger.error(f"Error retrieving diagram by figure: {e}")
        return jsonify({"error": str(e)}), 500


@diagram_bp.route('/search', methods=['GET'])
def search_diagrams():
    """
    Search diagrams by caption text.
    
    GET /api/diagrams/search?q=photosynthesis&subject=Biology&class=11
    
    Query Parameters:
        q: Search query (required)
        subject: Subject filter (optional)
        class: Class level filter (optional)
    
    Returns:
        JSON array of matching diagrams
    """
    try:
        if not diagram_service:
            return jsonify({"error": "Diagram service not initialized"}), 500
        
        query = request.args.get('q')
        if not query:
            return jsonify({"error": "Missing required parameter: q"}), 400
        
        subject = request.args.get('subject')
        class_level = request.args.get('class', type=int)
        
        diagrams = diagram_service.search_diagrams_by_caption(query, subject, class_level)
        
        return jsonify({
            "query": query,
            "count": len(diagrams),
            "diagrams": diagrams
        }), 200
        
    except Exception as e:
        logger.error(f"Error searching diagrams: {e}")
        return jsonify({"error": str(e)}), 500


@diagram_bp.route('/file/<page_id>', methods=['GET'])
def serve_diagram_file(page_id):
    """
    Serve diagram image file.
    
    GET /api/diagrams/file/<page_id>
    
    Returns:
        Image file (PNG)
    """
    try:
        if not diagram_service:
            return jsonify({"error": "Diagram service not initialized"}), 500
        
        file_path = diagram_service.get_diagram_file_path(page_id)
        
        if not file_path or not file_path.exists():
            return jsonify({"error": "Diagram file not found"}), 404
        
        return send_file(file_path, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error serving diagram file {page_id}: {e}")
        return jsonify({"error": str(e)}), 500


@diagram_bp.route('/labeled-parts/<page_id>', methods=['GET'])
def get_labeled_parts(page_id):
    """
    Get labeled parts explanation for a diagram.
    
    GET /api/diagrams/labeled-parts/<page_id>?context=<optional_context>
    
    Query Parameters:
        context: Optional NCERT context text (optional)
    
    Returns:
        JSON with labeled parts and explanations
    """
    try:
        if not diagram_service:
            return jsonify({"error": "Diagram service not initialized"}), 500
        
        context_text = request.args.get('context')
        
        labeled_parts = diagram_service.get_labeled_parts_explanation(page_id, context_text)
        
        return jsonify(labeled_parts), 200
        
    except Exception as e:
        logger.error(f"Error getting labeled parts for {page_id}: {e}")
        return jsonify({"error": str(e)}), 500


@diagram_bp.route('/for-concept', methods=['GET'])
def get_diagrams_for_concept():
    """
    Get relevant diagrams for a concept.
    
    GET /api/diagrams/for-concept?concept=photosynthesis&subject=Biology
    
    Query Parameters:
        concept: Concept or topic (required)
        subject: Subject filter (optional)
    
    Returns:
        JSON array of relevant diagrams
    """
    try:
        if not diagram_service:
            return jsonify({"error": "Diagram service not initialized"}), 500
        
        concept = request.args.get('concept')
        if not concept:
            return jsonify({"error": "Missing required parameter: concept"}), 400
        
        subject = request.args.get('subject')
        
        diagrams = diagram_service.get_diagrams_for_concept(concept, subject)
        
        return jsonify({
            "concept": concept,
            "count": len(diagrams),
            "diagrams": diagrams
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting diagrams for concept: {e}")
        return jsonify({"error": str(e)}), 500
