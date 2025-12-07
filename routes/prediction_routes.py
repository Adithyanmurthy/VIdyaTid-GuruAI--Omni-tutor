"""
Question Paper Prediction Routes
AI-powered prediction of future NEET questions based on patterns.
"""

import logging
from flask import Blueprint, request, jsonify
from functools import wraps
from services.question_predictor import QuestionPredictor
from services.feature_gate_service import get_feature_gate_service
from routes.auth_routes import require_auth

logger = logging.getLogger(__name__)

# Create Blueprint
prediction_bp = Blueprint('prediction', __name__, url_prefix='/api/prediction')


def get_predictor(user_id: str = None):
    """
    Get predictor instance for a specific user.
    
    Args:
        user_id: User ID for tier-based access checks
        
    Returns:
        QuestionPredictor instance
    """
    return QuestionPredictor(user_id=user_id)


@prediction_bp.route('/predict-paper', methods=['POST'])
@require_auth
def predict_paper(user_id: str, **kwargs):
    """
    Predict a future NEET question paper.
    
    Request JSON:
        {
            "subject": "Physics" | "Chemistry" | "Biology",
            "year": 2026,
            "use_ai": true
        }
    
    Response JSON:
        {
            "success": true,
            "paper": {predicted paper object},
            "insights": {prediction insights}
        }
    """
    try:
        data = request.get_json() or {}
        
        subject = data.get('subject')
        year = data.get('year', 2026)
        use_ai = data.get('use_ai', True)
        
        if not subject:
            return jsonify({
                'success': False,
                'error': 'Subject is required'
            }), 400
        
        if subject not in ['Physics', 'Chemistry', 'Biology']:
            return jsonify({
                'success': False,
                'error': 'Invalid subject. Must be Physics, Chemistry, or Biology'
            }), 400
        
        # Get predictor with user_id for tier-based access
        predictor = get_predictor(user_id=user_id)
        
        # Generate prediction (will check access and increment counter)
        paper = predictor.predict_question_paper(subject, year, use_ai)
        
        # Get insights (doesn't count against limit)
        insights = predictor.get_prediction_insights(subject)
        
        return jsonify({
            'success': True,
            'paper': paper,
            'insights': insights
        }), 200
        
    except PermissionError as e:
        # User doesn't have access or reached limit
        logger.warning(f"Permission denied for user {user_id}: {e}")
        
        # Get upgrade prompt
        feature_gate = get_feature_gate_service()
        upgrade_prompt = feature_gate.get_upgrade_prompt(user_id, 'prediction_insights')
        
        return jsonify({
            'success': False,
            'error': 'Access denied',
            'message': str(e),
            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
        }), 403
        
    except Exception as e:
        logger.error(f"Error in predict-paper: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to generate prediction',
            'details': str(e)
        }), 500


@prediction_bp.route('/insights/<subject>', methods=['GET'])
@require_auth
def get_insights(subject, user_id: str, **kwargs):
    """
    Get prediction insights for a subject.
    
    Response JSON:
        {
            "success": true,
            "insights": {
                "high_probability_chapters": [...],
                "recommended_focus": [...],
                "difficulty_trend": {...},
                "data_confidence": 0.85
            }
        }
    """
    try:
        if subject not in ['Physics', 'Chemistry', 'Biology']:
            return jsonify({
                'success': False,
                'error': 'Invalid subject'
            }), 400
        
        # Get predictor with user_id for tier-based access
        predictor = get_predictor(user_id=user_id)
        
        # Get insights (doesn't count against prediction limit)
        insights = predictor.get_prediction_insights(subject)
        
        return jsonify({
            'success': True,
            'insights': insights
        }), 200
        
    except PermissionError as e:
        # User doesn't have access
        logger.warning(f"Permission denied for user {user_id}: {e}")
        
        # Get upgrade prompt
        feature_gate = get_feature_gate_service()
        upgrade_prompt = feature_gate.get_upgrade_prompt(user_id, 'prediction_insights')
        
        return jsonify({
            'success': False,
            'error': 'Access denied',
            'message': str(e),
            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
        }), 403
        
    except Exception as e:
        logger.error(f"Error in get-insights: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to get insights',
            'details': str(e)
        }), 500


@prediction_bp.route('/smart-paper', methods=['POST'])
@require_auth
def generate_smart_paper(user_id: str, **kwargs):
    """
    Generate a smart practice paper based on weak areas.
    
    Request JSON:
        {
            "subject": "Physics",
            "focus_chapters": ["Mechanics", "Thermodynamics"],
            "difficulty_level": "mixed" | "easy" | "medium" | "hard"
        }
    
    Response JSON:
        {
            "success": true,
            "paper": {smart practice paper}
        }
    """
    try:
        data = request.get_json() or {}
        
        subject = data.get('subject')
        focus_chapters = data.get('focus_chapters', [])
        difficulty_level = data.get('difficulty_level', 'mixed')
        
        if not subject:
            return jsonify({
                'success': False,
                'error': 'Subject is required'
            }), 400
        
        # Get predictor with user_id for tier-based access
        predictor = get_predictor(user_id=user_id)
        
        # Generate smart paper (will check access and increment counter)
        paper = predictor.generate_smart_paper(
            subject=subject,
            focus_chapters=focus_chapters if focus_chapters else None,
            difficulty_level=difficulty_level
        )
        
        return jsonify({
            'success': True,
            'paper': paper
        }), 200
        
    except PermissionError as e:
        # User doesn't have access or reached limit
        logger.warning(f"Permission denied for user {user_id}: {e}")
        
        # Get upgrade prompt
        feature_gate = get_feature_gate_service()
        upgrade_prompt = feature_gate.get_upgrade_prompt(user_id, 'smart_paper_generation')
        
        return jsonify({
            'success': False,
            'error': 'Access denied',
            'message': str(e),
            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
        }), 403
        
    except Exception as e:
        logger.error(f"Error in smart-paper: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to generate smart paper',
            'details': str(e)
        }), 500


@prediction_bp.route('/chapter-analysis/<subject>', methods=['GET'])
@require_auth
def get_chapter_analysis(subject, user_id: str, **kwargs):
    """
    Get chapter-wise analysis and predictions.
    
    Response JSON:
        {
            "success": true,
            "analysis": {
                "chapters": [
                    {
                        "name": "Mechanics",
                        "frequency": 15,
                        "probability": 0.85,
                        "recommended": true
                    }
                ]
            }
        }
    """
    try:
        if subject not in ['Physics', 'Chemistry', 'Biology']:
            return jsonify({
                'success': False,
                'error': 'Invalid subject'
            }), 400
        
        # Get predictor with user_id for tier-based access
        predictor = get_predictor(user_id=user_id)
        
        # Analyze patterns (will check chapter_analysis access)
        patterns = predictor.analyze_previous_patterns(subject)
        
        # Format chapter analysis
        chapters = []
        for chapter, freq in patterns['top_chapters']:
            chapters.append({
                'name': chapter,
                'frequency': freq,
                'probability': min(freq / 50, 1.0),  # Normalize to 0-1
                'recommended': True
            })
        
        return jsonify({
            'success': True,
            'analysis': {
                'chapters': chapters,
                'total_analyzed': patterns['total_analyzed']
            }
        }), 200
        
    except PermissionError as e:
        # User doesn't have access
        logger.warning(f"Permission denied for user {user_id}: {e}")
        
        # Get upgrade prompt
        feature_gate = get_feature_gate_service()
        upgrade_prompt = feature_gate.get_upgrade_prompt(user_id, 'chapter_analysis')
        
        return jsonify({
            'success': False,
            'error': 'Access denied',
            'message': str(e),
            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
        }), 403
        
    except Exception as e:
        logger.error(f"Error in chapter-analysis: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to get analysis',
            'details': str(e)
        }), 500


@prediction_bp.route('/complete-neet/<int:year>', methods=['GET'])
@require_auth
def get_complete_neet_prediction(year, user_id: str, **kwargs):
    """
    Get complete NEET prediction with all 200 questions.
    
    Response JSON:
        {
            "success": true,
            "paper": {
                "paper_info": {
                    "total_questions": 200,
                    "to_attempt": 180,
                    "duration_minutes": 200,
                    "total_marks": 720,
                    "subjects": {...}
                },
                "questions": [200 questions]
            }
        }
    """
    try:
        # Get predictor with user_id for tier-based access
        predictor = get_predictor(user_id=user_id)
        
        # Generate complete NEET paper (will check access and increment counter)
        paper = predictor.predict_complete_neet_paper(year)
        
        return jsonify({
            'success': True,
            'paper': paper
        }), 200
        
    except PermissionError as e:
        # User doesn't have access or reached limit
        logger.warning(f"Permission denied for user {user_id}: {e}")
        
        # Get upgrade prompt
        feature_gate = get_feature_gate_service()
        upgrade_prompt = feature_gate.get_upgrade_prompt(user_id, 'complete_paper_prediction')
        
        return jsonify({
            'success': False,
            'error': 'Access denied',
            'message': str(e),
            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
        }), 403
        
    except Exception as e:
        logger.error(f"Error in complete-neet: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to generate complete NEET prediction',
            'details': str(e)
        }), 500


@prediction_bp.route('/full-prediction/<int:year>', methods=['GET'])
@require_auth
def get_full_prediction(year, user_id: str, **kwargs):
    """
    Get full NEET prediction for all subjects (separate papers).
    
    Response JSON:
        {
            "success": true,
            "predictions": {
                "Physics": {paper},
                "Chemistry": {paper},
                "Biology": {paper}
            },
            "overall_confidence": 0.85
        }
    """
    try:
        # Get predictor with user_id for tier-based access
        predictor = get_predictor(user_id=user_id)
        
        predictions = {}
        confidences = []
        
        # Generate predictions for each subject
        # Note: Each subject prediction will check access and increment counter
        for subject in ['Physics', 'Chemistry', 'Biology']:
            paper = predictor.predict_question_paper(subject, year)
            predictions[subject] = paper
            confidences.append(paper['paper_info']['prediction_confidence'])
        
        overall_confidence = sum(confidences) / len(confidences)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'overall_confidence': overall_confidence,
            'year': year
        }), 200
        
    except PermissionError as e:
        # User doesn't have access or reached limit
        logger.warning(f"Permission denied for user {user_id}: {e}")
        
        # Get upgrade prompt
        feature_gate = get_feature_gate_service()
        upgrade_prompt = feature_gate.get_upgrade_prompt(user_id, 'prediction_insights')
        
        return jsonify({
            'success': False,
            'error': 'Access denied',
            'message': str(e),
            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
        }), 403
        
    except Exception as e:
        logger.error(f"Error in full-prediction: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to generate full prediction',
            'details': str(e)
        }), 500
