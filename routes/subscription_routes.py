"""
Subscription API routes for VidyaTid.
Handles subscription management, upgrades, downgrades, and cancellations.
"""
from flask import Blueprint, request, jsonify
import logging

from routes.auth_routes import require_auth
from services.subscription_service import get_subscription_service
from services.tier_config import validate_tier

logger = logging.getLogger(__name__)

# Create blueprint
subscription_bp = Blueprint('subscription', __name__)


@subscription_bp.route('/api/subscription/current', methods=['GET'])
@require_auth
def get_current_subscription(user_id, session):
    """
    Get user's current subscription.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "subscription": {
                "subscription_id": "string",
                "user_id": "string",
                "tier": "string",
                "status": "string",
                "start_date": "string",
                "end_date": "string",
                "auto_renew": boolean,
                "is_active": boolean,
                "days_remaining": int,
                "cancelled_at": "string" or null,
                "scheduled_tier_change": "string" or null,
                "scheduled_change_date": "string" or null
            } or null
        }
    """
    try:
        subscription_service = get_subscription_service()
        subscription_info = subscription_service.get_user_subscription(user_id)
        
        if not subscription_info:
            return jsonify({
                'subscription': None,
                'message': 'No subscription found'
            }), 200
        
        return jsonify({
            'subscription': subscription_info.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting current subscription for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'SUBSCRIPTION_ERROR',
                'message': 'Failed to get subscription',
                'details': str(e)
            }
        }), 500


@subscription_bp.route('/api/subscription/upgrade', methods=['POST'])
@require_auth
def upgrade_subscription(user_id, session):
    """
    Upgrade user's subscription to a higher tier.
    Activates immediately with prorated pricing.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "new_tier": "starter" | "premium" | "ultimate"
        }
    
    Returns:
        {
            "status": "success",
            "upgrade": {
                "success": boolean,
                "message": "string",
                "prorated_amount": int,
                "new_end_date": "string"
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'new_tier' not in data:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'new_tier is required'
                }
            }), 400
        
        new_tier = data['new_tier']
        
        # Validate tier
        if not validate_tier(new_tier):
            return jsonify({
                'error': {
                    'code': 'INVALID_TIER',
                    'message': f'Invalid tier: {new_tier}. Must be one of: free, starter, premium, ultimate'
                }
            }), 400
        
        # Perform upgrade
        subscription_service = get_subscription_service()
        upgrade_result = subscription_service.upgrade_subscription(user_id, new_tier)
        
        if not upgrade_result.success:
            return jsonify({
                'error': {
                    'code': 'UPGRADE_FAILED',
                    'message': upgrade_result.message
                }
            }), 400
        
        logger.info(f"User {user_id} upgraded to {new_tier}")
        
        return jsonify({
            'status': 'success',
            'upgrade': upgrade_result.to_dict(),
            'message': upgrade_result.message
        }), 200
        
    except Exception as e:
        logger.error(f"Error upgrading subscription for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'UPGRADE_ERROR',
                'message': 'Failed to upgrade subscription',
                'details': str(e)
            }
        }), 500


@subscription_bp.route('/api/subscription/downgrade', methods=['POST'])
@require_auth
def downgrade_subscription(user_id, session):
    """
    Schedule a downgrade to a lower tier.
    Downgrade takes effect at the end of the current billing cycle.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "new_tier": "free" | "starter" | "premium"
        }
    
    Returns:
        {
            "status": "success",
            "downgrade": {
                "success": boolean,
                "message": "string",
                "scheduled_date": "string"
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'new_tier' not in data:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'new_tier is required'
                }
            }), 400
        
        new_tier = data['new_tier']
        
        # Validate tier
        if not validate_tier(new_tier):
            return jsonify({
                'error': {
                    'code': 'INVALID_TIER',
                    'message': f'Invalid tier: {new_tier}. Must be one of: free, starter, premium, ultimate'
                }
            }), 400
        
        # Schedule downgrade
        subscription_service = get_subscription_service()
        downgrade_result = subscription_service.downgrade_subscription(user_id, new_tier)
        
        if not downgrade_result.success:
            return jsonify({
                'error': {
                    'code': 'DOWNGRADE_FAILED',
                    'message': downgrade_result.message
                }
            }), 400
        
        logger.info(f"User {user_id} scheduled downgrade to {new_tier}")
        
        return jsonify({
            'status': 'success',
            'downgrade': downgrade_result.to_dict(),
            'message': downgrade_result.message
        }), 200
        
    except Exception as e:
        logger.error(f"Error scheduling downgrade for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'DOWNGRADE_ERROR',
                'message': 'Failed to schedule downgrade',
                'details': str(e)
            }
        }), 500


@subscription_bp.route('/api/subscription/cancel', methods=['POST'])
@require_auth
def cancel_subscription(user_id, session):
    """
    Cancel user's subscription.
    Sets auto_renew to false and records cancellation timestamp.
    Subscription remains active until end_date.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "status": "success",
            "message": "Subscription cancelled. Access until {end_date}"
        }
    """
    try:
        subscription_service = get_subscription_service()
        
        # Get current subscription to show end date
        subscription_info = subscription_service.get_user_subscription(user_id)
        
        if not subscription_info:
            return jsonify({
                'error': {
                    'code': 'NO_SUBSCRIPTION',
                    'message': 'No active subscription found'
                }
            }), 404
        
        # Cancel subscription
        success = subscription_service.cancel_subscription(user_id)
        
        if not success:
            return jsonify({
                'error': {
                    'code': 'CANCEL_FAILED',
                    'message': 'Failed to cancel subscription'
                }
            }), 500
        
        logger.info(f"User {user_id} cancelled subscription")
        
        # Format end date for message
        end_date_str = subscription_info.end_date.strftime('%Y-%m-%d') if subscription_info.end_date else 'unknown'
        
        return jsonify({
            'status': 'success',
            'message': f'Subscription cancelled. You will have access until {end_date_str}',
            'end_date': subscription_info.end_date.isoformat() if subscription_info.end_date else None
        }), 200
        
    except Exception as e:
        logger.error(f"Error cancelling subscription for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'CANCEL_ERROR',
                'message': 'Failed to cancel subscription',
                'details': str(e)
            }
        }), 500
