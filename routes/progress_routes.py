"""
Progress tracking API routes for GuruAI.
Handles progress recording, statistics retrieval, and recommendations.
"""
from flask import Blueprint, request, jsonify
from functools import wraps
import logging

from models.database import SessionLocal
from services.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

# Create blueprint
progress_bp = Blueprint('progress', __name__, url_prefix='/api/progress')

# Global progress tracker instance
progress_tracker = None


def init_progress_tracker():
    """Initialize the progress tracker."""
    global progress_tracker
    db = SessionLocal()
    progress_tracker = ProgressTracker(db)
    logger.info("Progress tracker initialized")


def require_user_id(f):
    """Decorator to ensure user_id is provided in request."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = request.json.get('user_id') if request.is_json else request.args.get('user_id')
        if not user_id:
            return jsonify({
                'error': {
                    'code': 'MISSING_USER_ID',
                    'message': 'User ID is required'
                }
            }), 400
        return f(user_id=user_id, *args, **kwargs)
    return decorated_function


@progress_bp.route('/record', methods=['POST'])
def record_attempt():
    """
    Record question attempt(s) or test results.
    
    Supports two formats:
    
    1. Single Question Attempt:
        {
            "user_id": "string",
            "subject": "string",
            "chapter": "string",
            "topic": "string",
            "is_correct": boolean,
            "question_id": "string" (optional),
            "difficulty": "string" (optional)
        }
    
    2. Test Results (from question paper):
        {
            "paper_id": "string",
            "answers": {question_index: answer},
            "results": {
                "correct": int,
                "incorrect": int,
                "unattempted": int
            }
        }
    
    Returns:
        {
            "status": "success",
            "message": "string"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': {
                    'code': 'INVALID_REQUEST',
                    'message': 'Request body is required'
                }
            }), 400
        
        # Check if this is a test result submission
        if 'paper_id' in data and 'results' in data:
            # Handle test results
            return record_test_results(data)
        
        # Handle single question attempt
        required_fields = ['user_id', 'subject', 'chapter', 'topic', 'is_correct']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': f'Missing required fields: {", ".join(missing_fields)}'
                }
            }), 400
        
        # Record the attempt
        result = progress_tracker.record_attempt(
            user_id=data['user_id'],
            subject=data['subject'],
            chapter=data['chapter'],
            topic=data['topic'],
            is_correct=data['is_correct'],
            question_id=data.get('question_id'),
            difficulty=data.get('difficulty')
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error recording attempt: {e}")
        return jsonify({
            'error': {
                'code': 'RECORD_FAILED',
                'message': 'Failed to record attempt',
                'details': str(e)
            }
        }), 500


def record_test_results(data):
    """
    Record test results from question paper.
    
    Args:
        data: Dictionary with paper_id, answers, and results
    
    Returns:
        JSON response
    """
    try:
        paper_id = data.get('paper_id')
        answers = data.get('answers', {})
        results = data.get('results', {})
        
        # Get user_id from session or use default
        user_id = data.get('user_id', 'default_user')
        
        # Extract statistics
        correct = results.get('correct', 0)
        incorrect = results.get('incorrect', 0)
        unattempted = results.get('unattempted', 0)
        total = correct + incorrect + unattempted
        
        # Calculate accuracy
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Store test result (simplified - in production, save to database)
        test_result = {
            'paper_id': paper_id,
            'user_id': user_id,
            'total_questions': total,
            'correct': correct,
            'incorrect': incorrect,
            'unattempted': unattempted,
            'accuracy': accuracy,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        logger.info(f"Test results recorded: {test_result}")
        
        return jsonify({
            'status': 'success',
            'message': 'Test results recorded successfully',
            'summary': {
                'correct': correct,
                'incorrect': incorrect,
                'unattempted': unattempted,
                'accuracy': round(accuracy, 2)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error recording test results: {e}")
        return jsonify({
            'error': {
                'code': 'RECORD_FAILED',
                'message': 'Failed to record test results',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>', methods=['GET'])
def get_user_progress(user_id):
    """
    Get comprehensive progress data for a user.
    
    Args:
        user_id: User ID from URL path
    
    Returns:
        {
            "user_id": "string",
            "subjects": {...},
            "chapters": {...},
            "overall_stats": {...},
            "recent_activity": [...]
        }
    """
    try:
        progress = progress_tracker.get_user_progress(user_id)
        return jsonify(progress), 200
        
    except Exception as e:
        logger.error(f"Error getting user progress: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch progress',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>/subject/<subject>', methods=['GET'])
def get_subject_progress(user_id, subject):
    """
    Get detailed progress for a specific subject.
    
    Args:
        user_id: User ID from URL path
        subject: Subject name from URL path
    
    Returns:
        {
            "subject": "string",
            "chapters": {...},
            "total_attempted": int,
            "total_correct": int,
            "accuracy": float
        }
    """
    try:
        progress = progress_tracker.get_subject_progress(user_id, subject)
        return jsonify(progress), 200
        
    except Exception as e:
        logger.error(f"Error getting subject progress: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch subject progress',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>/chapter', methods=['GET'])
def get_chapter_progress(user_id):
    """
    Get detailed progress for a specific chapter.
    
    Query Parameters:
        subject: Subject name
        chapter: Chapter name
    
    Returns:
        {
            "subject": "string",
            "chapter": "string",
            "topics": [...],
            "total_attempted": int,
            "total_correct": int,
            "accuracy": float
        }
    """
    try:
        subject = request.args.get('subject')
        chapter = request.args.get('chapter')
        
        if not subject or not chapter:
            return jsonify({
                'error': {
                    'code': 'MISSING_PARAMETERS',
                    'message': 'Subject and chapter parameters are required'
                }
            }), 400
        
        progress = progress_tracker.get_chapter_progress(user_id, subject, chapter)
        return jsonify(progress), 200
        
    except Exception as e:
        logger.error(f"Error getting chapter progress: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch chapter progress',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>/weak-areas', methods=['GET'])
def get_weak_areas(user_id):
    """
    Get weak areas for a user.
    
    Query Parameters:
        threshold: Accuracy threshold (default: 60.0)
        min_attempts: Minimum attempts required (default: 3)
    
    Returns:
        [
            {
                "subject": "string",
                "chapter": "string",
                "topic": "string",
                "accuracy": float,
                "questions_attempted": int
            },
            ...
        ]
    """
    try:
        threshold = float(request.args.get('threshold', 60.0))
        min_attempts = int(request.args.get('min_attempts', 3))
        
        weak_areas = progress_tracker.get_weak_areas(
            user_id,
            threshold=threshold,
            min_attempts=min_attempts
        )
        
        return jsonify(weak_areas), 200
        
    except Exception as e:
        logger.error(f"Error getting weak areas: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch weak areas',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>/recommendations', methods=['GET'])
def get_recommendations(user_id):
    """
    Get study recommendations for a user.
    
    Query Parameters:
        max_recommendations: Maximum number of recommendations (default: 5)
    
    Returns:
        [
            {
                "priority": "string",
                "reason": "string",
                "subject": "string",
                "chapter": "string",
                "topic": "string",
                "current_accuracy": float,
                "questions_attempted": int,
                "recommendation": "string"
            },
            ...
        ]
    """
    try:
        max_recommendations = int(request.args.get('max_recommendations', 5))
        
        recommendations = progress_tracker.generate_recommendations(
            user_id,
            max_recommendations=max_recommendations
        )
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({
            'error': {
                'code': 'GENERATION_FAILED',
                'message': 'Failed to generate recommendations',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>/statistics', methods=['GET'])
def get_statistics(user_id):
    """
    Get comprehensive statistics summary for a user.
    
    Returns:
        {
            "total_topics_studied": int,
            "total_questions_attempted": int,
            "total_questions_correct": int,
            "overall_accuracy": float,
            "subjects_covered": int,
            "chapters_covered": int,
            "strong_subjects": [...],
            "weak_subjects": [...],
            "study_streak": int
        }
    """
    try:
        statistics = progress_tracker.get_statistics_summary(user_id)
        return jsonify(statistics), 200
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch statistics',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>/export', methods=['GET'])
def export_progress(user_id):
    """
    Export all progress data for a user.
    
    Returns:
        {
            "user_id": "string",
            "export_date": "string",
            "total_records": int,
            "progress_data": [...]
        }
    """
    try:
        export_data = progress_tracker.export_progress(user_id)
        return jsonify(export_data), 200
        
    except Exception as e:
        logger.error(f"Error exporting progress: {e}")
        return jsonify({
            'error': {
                'code': 'EXPORT_FAILED',
                'message': 'Failed to export progress',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>/reset', methods=['POST'])
def reset_topic_progress(user_id):
    """
    Reset progress for a specific topic.
    
    Request Body:
        {
            "subject": "string",
            "chapter": "string",
            "topic": "string"
        }
    
    Returns:
        {
            "status": "success",
            "message": "string",
            "progress": {...}
        }
    """
    try:
        data = request.get_json()
        
        required_fields = ['subject', 'chapter', 'topic']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': f'Missing required fields: {", ".join(missing_fields)}'
                }
            }), 400
        
        result = progress_tracker.reset_topic_progress(
            user_id=user_id,
            subject=data['subject'],
            chapter=data['chapter'],
            topic=data['topic']
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error resetting progress: {e}")
        return jsonify({
            'error': {
                'code': 'RESET_FAILED',
                'message': 'Failed to reset progress',
                'details': str(e)
            }
        }), 500


@progress_bp.route('/<user_id>/delete', methods=['DELETE'])
def delete_user_progress(user_id):
    """
    Delete all progress data for a user.
    
    Returns:
        {
            "status": "success",
            "message": "string",
            "deleted_count": int
        }
    """
    try:
        result = progress_tracker.delete_user_progress(user_id)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error deleting progress: {e}")
        return jsonify({
            'error': {
                'code': 'DELETE_FAILED',
                'message': 'Failed to delete progress',
                'details': str(e)
            }
        }), 500
