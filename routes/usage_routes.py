"""
Usage API routes for VidyaTid.
Handles usage tracking, history, and statistics.
"""
from flask import Blueprint, request, jsonify
import logging
from datetime import datetime, date

from routes.auth_routes import require_auth
from services.usage_tracker import get_usage_tracker

logger = logging.getLogger(__name__)

# Create blueprint
usage_bp = Blueprint('usage', __name__)

# Simple in-memory cache for frequently accessed data
_usage_cache = {}
_cache_ttl = 300  # 5 minutes in seconds


def _get_cache_key(user_id: str, endpoint: str) -> str:
    """Generate cache key for user and endpoint"""
    return f"{user_id}:{endpoint}"


def _is_cache_valid(cache_entry: dict) -> bool:
    """Check if cache entry is still valid"""
    if not cache_entry:
        return False
    
    timestamp = cache_entry.get('timestamp')
    if not timestamp:
        return False
    
    age = (datetime.utcnow() - timestamp).total_seconds()
    return age < _cache_ttl


def _get_from_cache(user_id: str, endpoint: str):
    """Get data from cache if valid"""
    cache_key = _get_cache_key(user_id, endpoint)
    cache_entry = _usage_cache.get(cache_key)
    
    if _is_cache_valid(cache_entry):
        logger.debug(f"Cache hit for {cache_key}")
        return cache_entry.get('data')
    
    return None


def _set_cache(user_id: str, endpoint: str, data):
    """Set data in cache with timestamp"""
    cache_key = _get_cache_key(user_id, endpoint)
    _usage_cache[cache_key] = {
        'data': data,
        'timestamp': datetime.utcnow()
    }
    logger.debug(f"Cache set for {cache_key}")


def _invalidate_cache(user_id: str):
    """Invalidate all cache entries for a user"""
    keys_to_remove = [key for key in _usage_cache.keys() if key.startswith(f"{user_id}:")]
    for key in keys_to_remove:
        del _usage_cache[key]
    logger.debug(f"Invalidated cache for user {user_id}")


@usage_bp.route('/api/usage/current', methods=['GET'])
@require_auth
def get_current_usage(user_id, session):
    """
    Get current day usage for the user.
    Includes caching for frequently accessed data.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "usage": {
                "usage_id": "string",
                "user_id": "string",
                "date": "string",
                "query_count": int,
                "queries_limit": int,
                "queries_remaining": int,
                "prediction_count": int,
                "predictions_limit": int,
                "predictions_remaining": int,
                "feature_usage": {},
                "created_at": "string",
                "updated_at": "string"
            },
            "warning": {
                "warning": boolean,
                "message": "string",
                "usage_percentage": float,
                "queries_remaining": int
            }
        }
    """
    try:
        # Check cache first
        cached_data = _get_from_cache(user_id, 'current')
        if cached_data:
            return jsonify(cached_data), 200
        
        # Get usage tracker
        usage_tracker = get_usage_tracker()
        
        # Get current usage
        usage_info = usage_tracker.get_usage(user_id)
        
        if not usage_info:
            return jsonify({
                'error': {
                    'code': 'USAGE_ERROR',
                    'message': 'Failed to get usage information'
                }
            }), 500
        
        # Check for usage warning
        warning_info = usage_tracker.check_usage_warning(user_id)
        
        response_data = {
            'usage': usage_info.to_dict(),
            'warning': warning_info
        }
        
        # Cache the response
        _set_cache(user_id, 'current', response_data)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error getting current usage for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'USAGE_ERROR',
                'message': 'Failed to get current usage',
                'details': str(e)
            }
        }), 500


@usage_bp.route('/api/usage/history', methods=['GET'])
@require_auth
def get_usage_history(user_id, session):
    """
    Get usage history for the user.
    Includes caching for frequently accessed data.
    
    Requires authentication token in header or cookie.
    
    Query Params:
        days: Number of days to fetch (default: 30, max: 90)
    
    Returns:
        {
            "history": [
                {
                    "usage_id": "string",
                    "date": "string",
                    "query_count": int,
                    "queries_limit": int,
                    "prediction_count": int,
                    "predictions_limit": int,
                    "feature_usage": {}
                }
            ],
            "days": int,
            "count": int
        }
    """
    try:
        # Get query parameters
        days = min(request.args.get('days', 30, type=int), 90)
        
        # Check cache first
        cache_key = f'history:{days}'
        cached_data = _get_from_cache(user_id, cache_key)
        if cached_data:
            return jsonify(cached_data), 200
        
        # Get usage tracker
        usage_tracker = get_usage_tracker()
        
        # Get usage history
        history = usage_tracker.get_usage_history(user_id, days=days)
        
        # Convert to dict
        history_data = [usage.to_dict() for usage in history]
        
        response_data = {
            'history': history_data,
            'days': days,
            'count': len(history_data)
        }
        
        # Cache the response
        _set_cache(user_id, cache_key, response_data)
        
        logger.info(f"Retrieved {len(history_data)} usage records for user {user_id}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error getting usage history for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'HISTORY_ERROR',
                'message': 'Failed to get usage history',
                'details': str(e)
            }
        }), 500


@usage_bp.route('/api/usage/stats', methods=['GET'])
@require_auth
def get_usage_stats(user_id, session):
    """
    Get usage statistics for dashboard display.
    Includes caching for frequently accessed data.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "stats": {
                "total_queries": int,
                "total_predictions": int,
                "avg_daily_queries": float,
                "days_tracked": int,
                "current_tier": "string",
                "queries_limit": int,
                "predictions_limit": int
            },
            "current_usage": {
                "query_count": int,
                "queries_remaining": int,
                "prediction_count": int,
                "predictions_remaining": int
            }
        }
    """
    try:
        # Check cache first
        cached_data = _get_from_cache(user_id, 'stats')
        if cached_data:
            return jsonify(cached_data), 200
        
        # Get usage tracker
        usage_tracker = get_usage_tracker()
        
        # Get usage statistics
        stats = usage_tracker.get_usage_stats(user_id)
        
        if not stats:
            return jsonify({
                'error': {
                    'code': 'STATS_ERROR',
                    'message': 'Failed to get usage statistics'
                }
            }), 500
        
        # Get current usage for today
        current_usage = usage_tracker.get_usage(user_id)
        
        response_data = {
            'stats': stats.to_dict(),
            'current_usage': {
                'query_count': current_usage.query_count if current_usage else 0,
                'queries_remaining': current_usage.queries_remaining if current_usage else 0,
                'prediction_count': current_usage.prediction_count if current_usage else 0,
                'predictions_remaining': current_usage.predictions_remaining if current_usage else 0
            } if current_usage else None
        }
        
        # Cache the response
        _set_cache(user_id, 'stats', response_data)
        
        logger.info(f"Retrieved usage stats for user {user_id}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error getting usage stats for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'STATS_ERROR',
                'message': 'Failed to get usage statistics',
                'details': str(e)
            }
        }), 500


@usage_bp.route('/api/usage/remaining', methods=['GET'])
@require_auth
def get_remaining_queries(user_id, session):
    """
    Get remaining queries for the user today.
    Lightweight endpoint for quick checks.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "queries_remaining": int,
            "predictions_remaining": int,
            "unlimited_queries": boolean,
            "unlimited_predictions": boolean
        }
    """
    try:
        # Get usage tracker
        usage_tracker = get_usage_tracker()
        
        # Get remaining queries
        queries_remaining = usage_tracker.get_remaining_queries(user_id)
        
        # Get current usage for predictions
        current_usage = usage_tracker.get_usage(user_id)
        predictions_remaining = current_usage.predictions_remaining if current_usage else 0
        
        return jsonify({
            'queries_remaining': queries_remaining,
            'predictions_remaining': predictions_remaining,
            'unlimited_queries': queries_remaining == -1,
            'unlimited_predictions': predictions_remaining == -1
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting remaining queries for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'REMAINING_ERROR',
                'message': 'Failed to get remaining queries',
                'details': str(e)
            }
        }), 500


@usage_bp.route('/api/usage/predictions', methods=['GET'])
@require_auth
def get_prediction_usage(user_id, session):
    """
    Get prediction usage information for the user.
    Shows monthly prediction usage and history.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "current_month": {
                "prediction_count": int,
                "predictions_limit": int,
                "predictions_remaining": int,
                "unlimited": boolean
            },
            "history": [
                {
                    "date": "string",
                    "prediction_count": int,
                    "predictions_limit": int
                }
            ]
        }
    """
    try:
        # Check cache first
        cached_data = _get_from_cache(user_id, 'predictions')
        if cached_data:
            return jsonify(cached_data), 200
        
        # Get usage tracker
        usage_tracker = get_usage_tracker()
        
        # Get current usage
        current_usage = usage_tracker.get_usage(user_id)
        
        if not current_usage:
            return jsonify({
                'error': {
                    'code': 'USAGE_ERROR',
                    'message': 'Failed to get prediction usage'
                }
            }), 500
        
        # Get usage history (last 30 days)
        history = usage_tracker.get_usage_history(user_id, days=30)
        
        # Filter to only include prediction data
        prediction_history = []
        for usage in history:
            if usage.prediction_count > 0 or usage.predictions_limit > 0:
                prediction_history.append({
                    'date': usage.date.isoformat() if usage.date else None,
                    'prediction_count': usage.prediction_count,
                    'predictions_limit': usage.predictions_limit
                })
        
        response_data = {
            'current_month': {
                'prediction_count': current_usage.prediction_count,
                'predictions_limit': current_usage.predictions_limit,
                'predictions_remaining': current_usage.predictions_remaining,
                'unlimited': current_usage.predictions_limit == -1
            },
            'history': prediction_history
        }
        
        # Cache the response
        _set_cache(user_id, 'predictions', response_data)
        
        logger.info(f"Retrieved prediction usage for user {user_id}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error getting prediction usage for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'PREDICTION_USAGE_ERROR',
                'message': 'Failed to get prediction usage',
                'details': str(e)
            }
        }), 500


@usage_bp.route('/api/usage/increment', methods=['POST'])
@require_auth
def increment_usage(user_id, session):
    """
    Increment usage counter for the user.
    Internal endpoint used by query processing.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "query_type": "query" | "prediction"
        }
    
    Returns:
        {
            "status": "success",
            "message": "Usage incremented",
            "queries_remaining": int
        }
    """
    try:
        data = request.get_json()
        
        query_type = data.get('query_type', 'query') if data else 'query'
        
        # Validate query type
        if query_type not in ['query', 'prediction']:
            return jsonify({
                'error': {
                    'code': 'INVALID_TYPE',
                    'message': 'query_type must be "query" or "prediction"'
                }
            }), 400
        
        # Get usage tracker
        usage_tracker = get_usage_tracker()
        
        # Increment usage
        success = usage_tracker.increment_usage(user_id, query_type=query_type)
        
        if not success:
            return jsonify({
                'error': {
                    'code': 'LIMIT_REACHED',
                    'message': f'Daily {query_type} limit reached'
                }
            }), 429  # Too Many Requests
        
        # Invalidate cache for this user
        _invalidate_cache(user_id)
        
        # Get remaining queries
        queries_remaining = usage_tracker.get_remaining_queries(user_id)
        
        logger.info(f"Incremented {query_type} usage for user {user_id}")
        
        return jsonify({
            'status': 'success',
            'message': f'{query_type.capitalize()} usage incremented',
            'queries_remaining': queries_remaining
        }), 200
        
    except Exception as e:
        logger.error(f"Error incrementing usage for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'INCREMENT_ERROR',
                'message': 'Failed to increment usage',
                'details': str(e)
            }
        }), 500
