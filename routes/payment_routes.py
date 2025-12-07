"""
Payment API routes for VidyaTid.
Handles subscription creation, payment verification, and webhooks.
"""
from flask import Blueprint, request, jsonify, render_template
import logging
import json

from services.payment_service import get_payment_service
from routes.auth_routes import require_auth

logger = logging.getLogger(__name__)

# Create blueprint
payment_bp = Blueprint('payment', __name__)


@payment_bp.route('/pricing')
def pricing_page():
    """Render the pricing page"""
    return render_template('pricing.html')


@payment_bp.route('/api/payment/pricing', methods=['GET'])
def get_pricing():
    """
    Get pricing information for all plans.
    
    Returns:
        {
            "free": {...},
            "basic": {...},
            "premium": {...}
        }
    """
    try:
        payment_service = get_payment_service()
        pricing = payment_service.get_pricing_info()
        
        return jsonify(pricing), 200
        
    except Exception as e:
        logger.error(f"Error getting pricing: {e}")
        return jsonify({
            'error': {
                'code': 'PRICING_ERROR',
                'message': 'Failed to get pricing information',
                'details': str(e)
            }
        }), 500


@payment_bp.route('/api/payment/subscription/create', methods=['POST'])
@require_auth
def create_subscription(user_id, session):
    """
    Create a new subscription.
    
    Request Body:
        {
            "tier": "basic" | "premium",
            "duration": "monthly" | "yearly"
        }
    
    Returns:
        {
            "subscription_id": "sub_xxx",
            "status": "created",
            "short_url": "https://rzp.io/xxx"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'tier' not in data or 'duration' not in data:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'Tier and duration are required'
                }
            }), 400
        
        tier = data['tier']
        duration = data['duration']
        
        # Validate tier and duration
        valid_tiers = ['basic', 'premium']
        valid_durations = ['monthly', 'yearly']
        
        if tier not in valid_tiers:
            return jsonify({
                'error': {
                    'code': 'INVALID_TIER',
                    'message': f'Tier must be one of: {", ".join(valid_tiers)}'
                }
            }), 400
        
        if duration not in valid_durations:
            return jsonify({
                'error': {
                    'code': 'INVALID_DURATION',
                    'message': f'Duration must be one of: {", ".join(valid_durations)}'
                }
            }), 400
        
        # Get user info
        from models.database import SessionLocal
        from models.user import User
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                return jsonify({
                    'error': {
                        'code': 'USER_NOT_FOUND',
                        'message': 'User not found'
                    }
                }), 404
            
            # Create subscription
            payment_service = get_payment_service()
            subscription = payment_service.create_subscription(
                user_id=user_id,
                tier=tier,
                duration=duration,
                customer_email=user.username,  # Assuming username is email
                customer_name=user.username
            )
            
            logger.info(f"Created subscription for user {user_id}: {tier} {duration}")
            
            return jsonify({
                'status': 'success',
                'subscription': subscription,
                'message': 'Subscription created successfully'
            }), 201
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error creating subscription: {e}")
        return jsonify({
            'error': {
                'code': 'SUBSCRIPTION_ERROR',
                'message': 'Failed to create subscription',
                'details': str(e)
            }
        }), 500


@payment_bp.route('/api/payment/order/create', methods=['POST'])
@require_auth
def create_order(user_id, session):
    """
    Create a payment order for one-time payment or tier-based subscription.
    Enhanced to support tier-based orders.
    
    Request Body:
        {
            "amount": 49900,  // in paise (₹499) - optional if tier is provided
            "currency": "INR",
            "notes": {},
            "tier": "starter" | "premium" | "ultimate",  // optional
            "duration": "monthly" | "yearly"  // optional, defaults to monthly
        }
    
    Returns:
        {
            "order_id": "order_xxx",
            "amount": 49900,
            "currency": "INR",
            "key_id": "rzp_test_xxx",
            "tier": "starter" (if tier-based),
            "duration": "monthly" (if tier-based)
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': {
                    'code': 'MISSING_DATA',
                    'message': 'Request body is required'
                }
            }), 400
        
        payment_service = get_payment_service()
        
        # Check if this is a tier-based order
        tier = data.get('tier')
        
        if tier:
            # Tier-based order
            duration = data.get('duration', 'monthly')
            
            try:
                order = payment_service.create_order_for_tier(
                    user_id=user_id,
                    tier=tier,
                    duration=duration
                )
                
                # Add key_id for frontend
                order['key_id'] = payment_service.key_id
                
                logger.info(f"Created tier-based order for user {user_id}: tier={tier}, duration={duration}")
                
                return jsonify({
                    'status': 'success',
                    'order': order
                }), 201
                
            except ValueError as ve:
                return jsonify({
                    'error': {
                        'code': 'INVALID_TIER',
                        'message': str(ve)
                    }
                }), 400
        else:
            # Regular amount-based order
            if 'amount' not in data:
                return jsonify({
                    'error': {
                        'code': 'MISSING_AMOUNT',
                        'message': 'Amount or tier is required'
                    }
                }), 400
            
            amount = data['amount']
            currency = data.get('currency', 'INR')
            notes = data.get('notes', {})
            notes['user_id'] = user_id
            
            # Create order
            order = payment_service.create_order(
                amount=amount,
                currency=currency,
                notes=notes
            )
            
            # Add key_id for frontend
            order['key_id'] = payment_service.key_id
            
            logger.info(f"Created order for user {user_id}: {order['order_id']}")
            
            return jsonify({
                'status': 'success',
                'order': order
            }), 201
        
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        return jsonify({
            'error': {
                'code': 'ORDER_ERROR',
                'message': 'Failed to create order',
                'details': str(e)
            }
        }), 500


@payment_bp.route('/api/payment/verify', methods=['POST'])
@require_auth
def verify_payment(user_id, session):
    """
    Verify payment signature after successful payment.
    
    Request Body:
        {
            "razorpay_order_id": "order_xxx",
            "razorpay_payment_id": "pay_xxx",
            "razorpay_signature": "xxx"
        }
    
    Returns:
        {
            "status": "success",
            "verified": true
        }
    """
    try:
        data = request.get_json()
        
        required_fields = ['razorpay_order_id', 'razorpay_payment_id', 'razorpay_signature']
        if not data or not all(field in data for field in required_fields):
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'Order ID, Payment ID, and Signature are required'
                }
            }), 400
        
        # Verify signature
        payment_service = get_payment_service()
        is_valid = payment_service.verify_payment_signature(
            razorpay_order_id=data['razorpay_order_id'],
            razorpay_payment_id=data['razorpay_payment_id'],
            razorpay_signature=data['razorpay_signature']
        )
        
        if not is_valid:
            return jsonify({
                'error': {
                    'code': 'INVALID_SIGNATURE',
                    'message': 'Payment signature verification failed'
                }
            }), 400
        
        # Record payment
        payment_service.record_payment(
            user_id=user_id,
            razorpay_payment_id=data['razorpay_payment_id'],
            razorpay_order_id=data['razorpay_order_id'],
            amount=data.get('amount', 0),
            currency=data.get('currency', 'INR'),
            status='captured',
            method=data.get('method')
        )
        
        logger.info(f"Payment verified for user {user_id}: {data['razorpay_payment_id']}")
        
        return jsonify({
            'status': 'success',
            'verified': True,
            'message': 'Payment verified successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error verifying payment: {e}")
        return jsonify({
            'error': {
                'code': 'VERIFICATION_ERROR',
                'message': 'Failed to verify payment',
                'details': str(e)
            }
        }), 500


@payment_bp.route('/api/payment/webhook', methods=['POST'])
def payment_webhook():
    """
    Handle Razorpay webhooks for subscription and payment events.
    Enhanced with signature verification and comprehensive event handling.
    
    Headers:
        X-Razorpay-Signature: Webhook signature
    
    Body:
        Razorpay webhook payload
    
    Returns:
        {
            "status": "success" | "error",
            "message": "string"
        }
    """
    try:
        # Get signature from header
        signature = request.headers.get('X-Razorpay-Signature')
        
        if not signature:
            logger.warning("Webhook received without signature")
            return jsonify({
                'status': 'error',
                'message': 'Missing signature'
            }), 400
        
        # Get raw payload
        payload = request.get_data(as_text=True)
        
        # Process webhook with signature verification
        payment_service = get_payment_service()
        result = payment_service.process_webhook(payload, signature)
        
        if not result['success']:
            logger.error(f"Webhook processing failed: {result.get('message')}")
            return jsonify({
                'status': 'error',
                'message': result.get('message', 'Webhook processing failed')
            }), 400 if result.get('error') == 'signature_verification_failed' else 500
        
        logger.info(f"Successfully processed webhook: {result.get('event', 'unknown')}")
        
        return jsonify({
            'status': 'success',
            'message': result.get('message', 'Webhook processed successfully')
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Webhook processing failed'
        }), 500



@payment_bp.route('/api/payment/subscription', methods=['GET'])
@require_auth
def get_subscription(user_id, session):
    """
    Get current user's subscription.
    
    Returns:
        {
            "subscription": {...} or null
        }
    """
    try:
        payment_service = get_payment_service()
        subscription = payment_service.get_user_subscription(user_id)
        
        return jsonify({
            'subscription': subscription
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting subscription: {e}")
        return jsonify({
            'error': {
                'code': 'SUBSCRIPTION_ERROR',
                'message': 'Failed to get subscription',
                'details': str(e)
            }
        }), 500


@payment_bp.route('/api/payment/subscription/cancel', methods=['POST'])
@require_auth
def cancel_subscription_route(user_id, session):
    """
    Cancel current user's subscription.
    
    Returns:
        {
            "status": "success",
            "message": "Subscription cancelled"
        }
    """
    try:
        payment_service = get_payment_service()
        success = payment_service.cancel_subscription(user_id)
        
        if not success:
            return jsonify({
                'error': {
                    'code': 'CANCEL_FAILED',
                    'message': 'Failed to cancel subscription'
                }
            }), 500
        
        return jsonify({
            'status': 'success',
            'message': 'Subscription cancelled successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error cancelling subscription: {e}")
        return jsonify({
            'error': {
                'code': 'CANCEL_ERROR',
                'message': 'Failed to cancel subscription',
                'details': str(e)
            }
        }), 500


@payment_bp.route('/api/payment/history', methods=['GET'])
@require_auth
def get_payment_history(user_id, session):
    """
    Get user's payment history with enhanced filtering.
    
    Query Params:
        limit: Number of payments to return (default: 10, max: 100)
        payment_type: Filter by payment type (subscription, upgrade, one_time)
        status: Filter by status (pending, captured, failed, refunded)
    
    Returns:
        {
            "payments": [
                {
                    "payment_id": "string",
                    "razorpay_payment_id": "string",
                    "amount": int,
                    "currency": "string",
                    "status": "string",
                    "payment_type": "string",
                    "payment_method": "string",
                    "created_at": "string",
                    "metadata": {}
                }
            ],
            "count": int
        }
    """
    try:
        # Get query parameters
        limit = min(request.args.get('limit', 10, type=int), 100)
        payment_type = request.args.get('payment_type')
        status = request.args.get('status')
        
        # Get payment service
        payment_service = get_payment_service()
        
        # Get all payments for user
        payments = payment_service.get_user_payments(user_id, limit=limit)
        
        # Apply filters if provided
        if payment_type:
            payments = [p for p in payments if p.get('payment_type') == payment_type]
        
        if status:
            payments = [p for p in payments if p.get('status') == status]
        
        logger.info(f"Retrieved {len(payments)} payment records for user {user_id}")
        
        return jsonify({
            'payments': payments,
            'count': len(payments)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting payment history: {e}")
        return jsonify({
            'error': {
                'code': 'HISTORY_ERROR',
                'message': 'Failed to get payment history',
                'details': str(e)
            }
        }), 500
