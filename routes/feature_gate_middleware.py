"""
Feature Gate Middleware for VidyaTid.
Provides decorators for route protection based on subscription tier and usage limits.
"""
from flask import jsonify
from functools import wraps
import logging

from services.feature_gate_service import get_feature_gate_service
from services.usage_tracker import get_usage_tracker

logger = logging.getLogger(__name__)


def check_feature_access(feature: str):
    """
    Decorator to check if user has access to a specific feature.
    Returns upgrade prompt if access is denied.
    
    Args:
        feature: Feature name to check (e.g., 'diagrams', 'image_solving', 'mock_tests')
    
    Usage:
        @app.route('/api/diagrams')
        @require_auth
        @check_feature_access('diagrams')
        def get_diagrams(user_id, session):
            # Route implementation
            pass
    
    Returns:
        If access denied:
        {
            "error": {
                "code": "FEATURE_RESTRICTED",
                "message": "Feature requires upgrade",
                "feature": "string",
                "current_tier": "string",
                "required_tiers": ["string"],
                "upgrade_prompt": {
                    "feature_name": "string",
                    "feature_description": "string",
                    "current_tier": "string",
                    "available_tiers": [
                        {
                            "tier": "string",
                            "name": "string",
                            "price_monthly": int,
                            "price_yearly": int,
                            "queries_per_day": int
                        }
                    ],
                    "message": "string"
                }
            }
        }
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract user_id from kwargs (added by require_auth)
            user_id = kwargs.get('user_id')
            
            if not user_id:
                return jsonify({
                    'error': {
                        'code': 'MISSING_USER_ID',
                        'message': 'User ID not found in request'
                    }
                }), 401
            
            try:
                # Check feature access
                feature_gate_service = get_feature_gate_service()
                access_result = feature_gate_service.can_access_feature(user_id, feature)
                
                if not access_result.allowed:
                    logger.warning(f"User {user_id} denied access to feature: {feature}")
                    
                    return jsonify({
                        'error': {
                            'code': 'FEATURE_RESTRICTED',
                            'message': access_result.message,
                            'feature': feature,
                            'current_tier': access_result.current_tier,
                            'required_tiers': access_result.required_tiers,
                            'upgrade_prompt': access_result.upgrade_prompt
                        }
                    }), 403
                
                # Access granted, proceed with route
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Error checking feature access for user {user_id}, feature {feature}: {e}")
                # Fail closed for security
                return jsonify({
                    'error': {
                        'code': 'ACCESS_CHECK_ERROR',
                        'message': 'Error checking feature access',
                        'details': str(e)
                    }
                }), 500
        
        return decorated_function
    return decorator


def check_query_limit():
    """
    Decorator to check if user has queries remaining.
    Returns error if daily limit is reached.
    
    Usage:
        @app.route('/api/query')
        @require_auth
        @check_query_limit()
        def submit_query(user_id, session):
            # Route implementation
            pass
    
    Returns:
        If limit reached:
        {
            "error": {
                "code": "QUERY_LIMIT_REACHED",
                "message": "Daily query limit reached",
                "queries_remaining": 0,
                "upgrade_prompt": {
                    "feature_name": "Increased Query Limit",
                    "feature_description": "Get more queries per day",
                    "current_tier": "string",
                    "available_tiers": [...]
                }
            }
        }
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract user_id from kwargs (added by require_auth)
            user_id = kwargs.get('user_id')
            
            if not user_id:
                return jsonify({
                    'error': {
                        'code': 'MISSING_USER_ID',
                        'message': 'User ID not found in request'
                    }
                }), 401
            
            try:
                # Check query limit
                usage_tracker = get_usage_tracker()
                limit_result = usage_tracker.check_limit(user_id)
                
                if not limit_result.allowed:
                    logger.warning(f"User {user_id} reached daily query limit")
                    
                    # Get upgrade prompt
                    feature_gate_service = get_feature_gate_service()
                    upgrade_prompt = feature_gate_service.get_upgrade_prompt(user_id, 'basic_queries')
                    
                    return jsonify({
                        'error': {
                            'code': 'QUERY_LIMIT_REACHED',
                            'message': limit_result.message,
                            'queries_remaining': limit_result.queries_remaining,
                            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
                        }
                    }), 429  # Too Many Requests
                
                # Limit not reached, proceed with route
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Error checking query limit for user {user_id}: {e}")
                # Fail open for better user experience
                logger.warning(f"Allowing query due to error checking limit")
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def check_prediction_limit():
    """
    Decorator to check if user has predictions remaining for the month.
    Returns error if monthly limit is reached.
    
    Usage:
        @app.route('/api/prediction/generate')
        @require_auth
        @check_prediction_limit()
        def generate_prediction(user_id, session):
            # Route implementation
            pass
    
    Returns:
        If limit reached:
        {
            "error": {
                "code": "PREDICTION_LIMIT_REACHED",
                "message": "Monthly prediction limit reached",
                "predictions_remaining": 0,
                "upgrade_prompt": {
                    "feature_name": "Increased Prediction Limit",
                    "feature_description": "Get more predictions per month",
                    "current_tier": "string",
                    "available_tiers": [...]
                }
            }
        }
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract user_id from kwargs (added by require_auth)
            user_id = kwargs.get('user_id')
            
            if not user_id:
                return jsonify({
                    'error': {
                        'code': 'MISSING_USER_ID',
                        'message': 'User ID not found in request'
                    }
                }), 401
            
            try:
                # Check prediction limit
                feature_gate_service = get_feature_gate_service()
                allowed, predictions_remaining = feature_gate_service.check_prediction_limit(user_id)
                
                if not allowed:
                    logger.warning(f"User {user_id} reached monthly prediction limit")
                    
                    # Get upgrade prompt
                    upgrade_prompt = feature_gate_service.get_upgrade_prompt(user_id, 'complete_paper_prediction')
                    
                    return jsonify({
                        'error': {
                            'code': 'PREDICTION_LIMIT_REACHED',
                            'message': 'Monthly prediction limit reached. Upgrade to get more predictions.',
                            'predictions_remaining': predictions_remaining,
                            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
                        }
                    }), 429  # Too Many Requests
                
                # Limit not reached, proceed with route
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Error checking prediction limit for user {user_id}: {e}")
                # Fail closed for security (predictions are premium feature)
                return jsonify({
                    'error': {
                        'code': 'LIMIT_CHECK_ERROR',
                        'message': 'Error checking prediction limit',
                        'details': str(e)
                    }
                }), 500
        
        return decorated_function
    return decorator


def check_prediction_access(prediction_type: str):
    """
    Decorator to check if user has access to a specific prediction feature.
    Returns upgrade prompt if access is denied.
    
    Args:
        prediction_type: Type of prediction (chapter_analysis, prediction_insights, 
                        smart_paper_generation, complete_paper_prediction)
    
    Usage:
        @app.route('/api/prediction/smart-paper')
        @require_auth
        @check_prediction_access('smart_paper_generation')
        def generate_smart_paper(user_id, session):
            # Route implementation
            pass
    
    Returns:
        If access denied:
        {
            "error": {
                "code": "PREDICTION_RESTRICTED",
                "message": "Prediction feature requires upgrade",
                "prediction_type": "string",
                "current_tier": "string",
                "required_tiers": ["string"],
                "upgrade_prompt": {...}
            }
        }
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract user_id from kwargs (added by require_auth)
            user_id = kwargs.get('user_id')
            
            if not user_id:
                return jsonify({
                    'error': {
                        'code': 'MISSING_USER_ID',
                        'message': 'User ID not found in request'
                    }
                }), 401
            
            try:
                # Check prediction access
                feature_gate_service = get_feature_gate_service()
                access_result = feature_gate_service.check_prediction_access(user_id, prediction_type)
                
                if not access_result.allowed:
                    logger.warning(f"User {user_id} denied access to prediction: {prediction_type}")
                    
                    return jsonify({
                        'error': {
                            'code': 'PREDICTION_RESTRICTED',
                            'message': access_result.message,
                            'prediction_type': prediction_type,
                            'current_tier': access_result.current_tier,
                            'required_tiers': access_result.required_tiers,
                            'upgrade_prompt': access_result.upgrade_prompt
                        }
                    }), 403
                
                # Access granted, proceed with route
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Error checking prediction access for user {user_id}, type {prediction_type}: {e}")
                # Fail closed for security
                return jsonify({
                    'error': {
                        'code': 'ACCESS_CHECK_ERROR',
                        'message': 'Error checking prediction access',
                        'details': str(e)
                    }
                }), 500
        
        return decorated_function
    return decorator


def check_previous_papers_access(year: int):
    """
    Decorator to check if user has access to previous papers for a specific year.
    Returns upgrade prompt if access is denied.
    
    Args:
        year: Year of the paper to check access for
    
    Usage:
        @app.route('/api/papers/<int:year>')
        @require_auth
        @check_previous_papers_access(year)
        def get_paper(year, user_id, session):
            # Route implementation
            pass
    
    Note: The year parameter must be in the route path for this to work correctly.
    
    Returns:
        If access denied:
        {
            "error": {
                "code": "PAPER_RESTRICTED",
                "message": "Access to papers from {year} requires upgrade",
                "year": int,
                "current_tier": "string",
                "upgrade_prompt": {...}
            }
        }
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract user_id from kwargs (added by require_auth)
            user_id = kwargs.get('user_id')
            
            if not user_id:
                return jsonify({
                    'error': {
                        'code': 'MISSING_USER_ID',
                        'message': 'User ID not found in request'
                    }
                }), 401
            
            try:
                # Check previous papers access
                feature_gate_service = get_feature_gate_service()
                has_access = feature_gate_service.check_previous_papers_access(user_id, year)
                
                if not has_access:
                    logger.warning(f"User {user_id} denied access to papers from year {year}")
                    
                    # Get upgrade prompt
                    upgrade_prompt = feature_gate_service.get_upgrade_prompt(user_id, 'previous_papers')
                    
                    # Get current tier for message
                    from services.subscription_service import get_subscription_service
                    sub_service = get_subscription_service()
                    sub_info = sub_service.get_user_subscription(user_id)
                    current_tier = sub_info.tier if sub_info else 'free'
                    
                    return jsonify({
                        'error': {
                            'code': 'PAPER_RESTRICTED',
                            'message': f'Access to papers from {year} requires upgrade',
                            'year': year,
                            'current_tier': current_tier,
                            'upgrade_prompt': upgrade_prompt.to_dict() if upgrade_prompt else None
                        }
                    }), 403
                
                # Access granted, proceed with route
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Error checking previous papers access for user {user_id}, year {year}: {e}")
                # Fail closed for security
                return jsonify({
                    'error': {
                        'code': 'ACCESS_CHECK_ERROR',
                        'message': 'Error checking paper access',
                        'details': str(e)
                    }
                }), 500
        
        return decorated_function
    return decorator
