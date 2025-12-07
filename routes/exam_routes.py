"""
Exam Routes
Handles NEET mock exam functionality
"""

from flask import Blueprint, render_template, jsonify, request, session
from services.question_predictor import QuestionPredictor
from services.prediction_file_loader import get_prediction_loader
import json

exam_bp = Blueprint('exam', __name__, url_prefix='/exam')


@exam_bp.route('/neet/<int:year>')
def neet_exam(year):
    """
    Start NEET mock exam.
    
    Args:
        year: Year of exam (e.g., 2026)
    """
    try:
        # Get user ID from session
        user_id = session.get('user_id')
        
        # Load prediction data
        loader = get_prediction_loader()
        exam_data = loader.load_complete_prediction(year)
        
        if not exam_data:
            # Fall back to predictor
            predictor = QuestionPredictor(user_id=user_id)
            exam_data = predictor.predict_complete_neet_paper(year=year, use_ai=False)
        
        # Render exam template
        return render_template('neet_exam.html', exam_data=exam_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@exam_bp.route('/smart-paper')
def smart_paper_exam():
    """
    Start smart paper practice exam.
    
    Query params:
        subject: Subject name
        difficulty: Difficulty level
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        subject = request.args.get('subject', 'Physics')
        difficulty = request.args.get('difficulty', 'medium')
        
        logger.info(f"Starting smart paper exam: subject={subject}, difficulty={difficulty}")
        
        # Get user ID from session
        user_id = session.get('user_id')
        
        # Generate smart paper
        predictor = QuestionPredictor(user_id=user_id)
        paper_data = predictor.generate_smart_paper(
            subject=subject,
            focus_chapters=None,  # Will use top chapters
            difficulty_level=difficulty
        )
        
        logger.info(f"Generated smart paper with {len(paper_data['questions'])} questions")
        
        # Format for exam template - match the structure expected by neet_exam.html
        exam_data = {
            'paper_info': {
                'exam_type': 'SMART_PRACTICE',
                'subject': subject,
                'year': 2026,
                'total_questions': paper_data['paper_info']['question_count'],
                'to_attempt': paper_data['paper_info']['question_count'],
                'duration_minutes': paper_data['paper_info']['duration_minutes'],
                'total_marks': paper_data['paper_info']['total_marks'],
                'pattern_type': 'all_compulsory',
                'difficulty_level': difficulty,
                'prediction_confidence': paper_data['paper_info'].get('prediction_confidence', 0.85)
            },
            'questions': paper_data['questions']
        }
        
        logger.info("Rendering exam template")
        
        # Render exam template
        return render_template('neet_exam.html', exam_data=exam_data, is_smart_paper=True)
    
    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        return render_template('error.html', error_message=str(e), error_code=403), 403
    
    except Exception as e:
        logger.error(f"Error in smart paper exam: {e}", exc_info=True)
        return render_template('error.html', error_message=f"Failed to load smart paper: {str(e)}", error_code=500), 500


@exam_bp.route('/submit', methods=['POST'])
def submit_exam():
    """
    Submit exam and calculate results.
    """
    try:
        data = request.json
        answers = data.get('answers', {})
        questions = data.get('questions', [])
        
        # Calculate results
        results = calculate_results(answers, questions)
        
        # TODO: Save results to database
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def calculate_results(answers, questions):
    """
    Calculate exam results.
    
    Args:
        answers: Dictionary of question_index: selected_option_index
        questions: List of question objects
    
    Returns:
        Dictionary with results
    """
    correct = 0
    incorrect = 0
    unattempted = 0
    
    subject_results = {
        'Physics': {'correct': 0, 'incorrect': 0, 'unattempted': 0},
        'Chemistry': {'correct': 0, 'incorrect': 0, 'unattempted': 0},
        'Biology': {'correct': 0, 'incorrect': 0, 'unattempted': 0}
    }
    
    for index, question in enumerate(questions):
        # Determine subject
        if index < 45:
            subject = 'Physics'
        elif index < 90:
            subject = 'Chemistry'
        else:
            subject = 'Biology'
        
        # Check answer
        if str(index) in answers:
            selected_option = question['options'][answers[str(index)]]
            if selected_option == question['correct_answer']:
                correct += 1
                subject_results[subject]['correct'] += 1
            else:
                incorrect += 1
                subject_results[subject]['incorrect'] += 1
        else:
            unattempted += 1
            subject_results[subject]['unattempted'] += 1
    
    # Calculate marks
    total_marks = (correct * 4) - (incorrect * 1)
    percentage = (correct / len(questions)) * 100 if questions else 0
    
    return {
        'correct': correct,
        'incorrect': incorrect,
        'unattempted': unattempted,
        'total_marks': total_marks,
        'percentage': round(percentage, 2),
        'subject_results': subject_results
    }
