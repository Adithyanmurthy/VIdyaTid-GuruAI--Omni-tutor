"""
Payment Service for VidyaTid using Razorpay.
Handles subscriptions, one-time payments, and webhook processing.
"""
import os
import razorpay
import hmac
import hashlib
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from models.database import SessionLocal
from models.user import User
from models.subscription import Subscription as UserSubscription
from models.payment import Payment
from services.email_service import get_email_service
from services.tier_config import get_tier_price, validate_tier
import json

logger = logging.getLogger(__name__)


class PaymentService:
    """Service for handling Razorpay payments and subscriptions"""
    
    # Subscription tiers
    TIER_FREE = 'free'
    TIER_BASIC = 'basic'
    TIER_PREMIUM = 'premium'
    
    # Plan durations
    DURATION_MONTHLY = 'monthly'
    DURATION_YEARLY = 'yearly'
    
    def __init__(self):
        """Initialize Razorpay client"""
        self.key_id = os.getenv('RAZORPAY_KEY_ID')
        self.key_secret = os.getenv('RAZORPAY_KEY_SECRET')
        self.webhook_secret = os.getenv('RAZORPAY_WEBHOOK_SECRET')
        
        if not self.key_id or not self.key_secret:
            logger.warning("Razorpay credentials not configured")
            self.client = None
        else:
            self.client = razorpay.Client(auth=(self.key_id, self.key_secret))
            logger.info("Razorpay client initialized")
        
        # Plan IDs from environment
        self.plans = {
            f'{self.TIER_BASIC}_{self.DURATION_MONTHLY}': os.getenv('RAZORPAY_PLAN_BASIC_MONTHLY'),
            f'{self.TIER_BASIC}_{self.DURATION_YEARLY}': os.getenv('RAZORPAY_PLAN_BASIC_YEARLY'),
            f'{self.TIER_PREMIUM}_{self.DURATION_MONTHLY}': os.getenv('RAZORPAY_PLAN_PREMIUM_MONTHLY'),
            f'{self.TIER_PREMIUM}_{self.DURATION_YEARLY}': os.getenv('RAZORPAY_PLAN_PREMIUM_YEARLY'),
        }
    
    def create_subscription(
        self,
        user_id: str,
        tier: str,
        duration: str,
        customer_email: str,
        customer_name: str
    ) -> Dict:
        """
        Create a Razorpay subscription.
        
        Args:
            user_id: User ID
            tier: Subscription tier (basic/premium)
            duration: Duration (monthly/yearly)
            customer_email: Customer email
            customer_name: Customer name
            
        Returns:
            Dict with subscription details
        """
        if not self.client:
            raise Exception("Razorpay not configured")
        
        plan_key = f'{tier}_{duration}'
        plan_id = self.plans.get(plan_key)
        
        if not plan_id:
            raise Exception(f"Plan not found: {plan_key}")
        
        try:
            # Create subscription in Razorpay
            subscription_data = {
                'plan_id': plan_id,
                'customer_notify': 1,
                'total_count': 12 if duration == self.DURATION_YEARLY else 1,
                'notes': {
                    'user_id': user_id,
                    'tier': tier,
                    'duration': duration
                }
            }
            
            razorpay_subscription = self.client.subscription.create(subscription_data)
            
            logger.info(f"Created Razorpay subscription: {razorpay_subscription['id']}")
            
            return {
                'subscription_id': razorpay_subscription['id'],
                'status': razorpay_subscription['status'],
                'short_url': razorpay_subscription.get('short_url'),
                'plan_id': plan_id,
                'tier': tier,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            raise
    
    def create_order(
        self,
        amount: int,
        currency: str = 'INR',
        receipt: Optional[str] = None,
        notes: Optional[Dict] = None
    ) -> Dict:
        """
        Create a Razorpay order for one-time payment.
        
        Args:
            amount: Amount in paise (₹1 = 100 paise)
            currency: Currency code (default: INR)
            receipt: Receipt ID
            notes: Additional notes
            
        Returns:
            Dict with order details
        """
        if not self.client:
            raise Exception("Razorpay not configured")
        
        try:
            order_data = {
                'amount': amount,
                'currency': currency,
                'receipt': receipt or f'order_{datetime.now().timestamp()}',
                'notes': notes or {}
            }
            
            order = self.client.order.create(data=order_data)
            
            logger.info(f"Created Razorpay order: {order['id']}")
            
            return {
                'order_id': order['id'],
                'amount': order['amount'],
                'currency': order['currency'],
                'status': order['status']
            }
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise
    
    def create_order_for_tier(
        self,
        user_id: str,
        tier: str,
        duration: str = 'monthly'
    ) -> Dict:
        """
        Create a Razorpay order for a specific subscription tier.
        Wrapper method that maps tier names to prices from TIER_CONFIG.
        
        Args:
            user_id: User ID
            tier: Subscription tier (free, starter, premium, ultimate)
            duration: Billing duration ('monthly' or 'yearly')
            
        Returns:
            Dict with order details including tier and user information
            
        Raises:
            ValueError: If tier is invalid or price cannot be determined
            Exception: If Razorpay order creation fails
        """
        # Validate tier
        if not validate_tier(tier):
            raise ValueError(f"Invalid tier: {tier}")
        
        # Get price for tier and duration
        amount = get_tier_price(tier, duration)
        
        if amount is None:
            raise ValueError(f"Could not determine price for tier {tier} with duration {duration}")
        
        # Free tier doesn't require payment
        if amount == 0:
            raise ValueError("Cannot create order for free tier")
        
        # Create receipt ID
        receipt = f"tier_{tier}_{duration}_{user_id}_{int(datetime.now().timestamp())}"
        
        # Add metadata with tier and user information
        notes = {
            'user_id': user_id,
            'tier': tier,
            'duration': duration,
            'order_type': 'subscription'
        }
        
        # Create the order
        order = self.create_order(
            amount=amount,
            currency='INR',
            receipt=receipt,
            notes=notes
        )
        
        # Add tier information to response
        order['tier'] = tier
        order['duration'] = duration
        order['user_id'] = user_id
        
        logger.info(f"Created order for user {user_id}: tier={tier}, duration={duration}, amount=₹{amount/100}")
        
        return order
    
    def verify_payment_signature(
        self,
        razorpay_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str
    ) -> bool:
        """
        Verify Razorpay payment signature.
        
        Args:
            razorpay_order_id: Order ID
            razorpay_payment_id: Payment ID
            razorpay_signature: Signature to verify
            
        Returns:
            True if signature is valid
        """
        if not self.client:
            return False
        
        try:
            params_dict = {
                'razorpay_order_id': razorpay_order_id,
                'razorpay_payment_id': razorpay_payment_id,
                'razorpay_signature': razorpay_signature
            }
            
            self.client.utility.verify_payment_signature(params_dict)
            return True
            
        except razorpay.errors.SignatureVerificationError:
            logger.error("Payment signature verification failed")
            return False
    
    def verify_webhook_signature(self, payload: str, signature: str) -> bool:
        """
        Verify webhook signature from Razorpay.
        
        Args:
            payload: Webhook payload (raw body)
            signature: X-Razorpay-Signature header
            
        Returns:
            True if signature is valid
        """
        if not self.webhook_secret:
            logger.warning("Webhook secret not configured")
            return False
        
        try:
            expected_signature = hmac.new(
                self.webhook_secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, signature)
            
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {e}")
            return False
    
    def process_webhook(self, payload: str, signature: str) -> Dict[str, Any]:
        """
        Process Razorpay webhook with signature verification.
        Handles different webhook event types and activates subscriptions.
        
        Args:
            payload: Webhook payload (raw body as string)
            signature: X-Razorpay-Signature header value
            
        Returns:
            Dict with processing result:
            {
                'success': bool,
                'message': str,
                'event': str (optional),
                'payment_id': str (optional)
            }
        """
        # Verify webhook signature
        if not self.verify_webhook_signature(payload, signature):
            logger.error("Webhook signature verification failed - possible security breach")
            return {
                'success': False,
                'message': 'Invalid webhook signature',
                'error': 'signature_verification_failed'
            }
        
        try:
            # Parse webhook payload
            webhook_data = json.loads(payload)
            event = webhook_data.get('event')
            
            logger.info(f"Processing webhook event: {event}")
            
            # Handle different event types
            if event == 'payment.captured':
                return self._handle_payment_captured(webhook_data)
            
            elif event == 'payment.failed':
                return self._handle_payment_failed(webhook_data)
            
            elif event == 'subscription.activated':
                return self._handle_subscription_activated(webhook_data)
            
            elif event == 'subscription.charged':
                return self._handle_subscription_charged(webhook_data)
            
            elif event == 'subscription.cancelled':
                return self._handle_subscription_cancelled(webhook_data)
            
            elif event == 'order.paid':
                return self._handle_order_paid(webhook_data)
            
            else:
                logger.warning(f"Unhandled webhook event: {event}")
                return {
                    'success': True,
                    'message': f'Event {event} acknowledged but not processed',
                    'event': event
                }
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse webhook payload: {e}")
            return {
                'success': False,
                'message': 'Invalid JSON payload',
                'error': 'json_parse_error'
            }
        
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return {
                'success': False,
                'message': f'Error processing webhook: {str(e)}',
                'error': 'processing_error'
            }
    
    def _handle_payment_captured(self, webhook_data: Dict) -> Dict[str, Any]:
        """Handle payment.captured webhook event"""
        try:
            payment_entity = webhook_data.get('payload', {}).get('payment', {}).get('entity', {})
            
            payment_id = payment_entity.get('id')
            order_id = payment_entity.get('order_id')
            amount = payment_entity.get('amount')
            status = payment_entity.get('status')
            method = payment_entity.get('method')
            
            # Extract user_id and tier from notes
            notes = payment_entity.get('notes', {})
            user_id = notes.get('user_id')
            tier = notes.get('tier')
            duration = notes.get('duration', 'monthly')
            
            if not user_id or not tier:
                logger.error(f"Missing user_id or tier in payment notes: {notes}")
                return {
                    'success': False,
                    'message': 'Missing required payment metadata',
                    'event': 'payment.captured'
                }
            
            # Import here to avoid circular dependency
            from services.subscription_service import get_subscription_service
            
            # Calculate duration in days
            duration_days = 365 if duration == 'yearly' else 30
            
            # Create/activate subscription
            sub_service = get_subscription_service()
            subscription = sub_service.create_subscription(
                user_id=user_id,
                tier=tier,
                duration_days=duration_days,
                razorpay_sub_id=order_id
            )
            
            if not subscription:
                logger.error(f"Failed to create subscription for user {user_id}")
                return {
                    'success': False,
                    'message': 'Failed to activate subscription',
                    'event': 'payment.captured'
                }
            
            # Record payment
            self.record_payment(
                user_id=user_id,
                razorpay_payment_id=payment_id,
                razorpay_order_id=order_id,
                amount=amount,
                currency='INR',
                status='captured',
                method=method,
                payment_type='subscription',
                subscription_id=subscription.subscription_id,
                metadata={
                    'tier': tier,
                    'duration': duration,
                    'webhook_event': 'payment.captured'
                }
            )
            
            # Send confirmation email
            try:
                db = SessionLocal()
                user = db.query(User).filter_by(user_id=user_id).first()
                if user:
                    email_service = get_email_service()
                    email_service.send_subscription_activated(
                        user_email=user.username,
                        username=user.username,
                        subscription_details={
                            'tier': tier.capitalize(),
                            'end_date': subscription.end_date.strftime('%B %d, %Y')
                        }
                    )
                db.close()
            except Exception as e:
                logger.error(f"Failed to send confirmation email: {e}")
            
            logger.info(f"Successfully processed payment.captured for user {user_id}")
            
            return {
                'success': True,
                'message': 'Payment captured and subscription activated',
                'event': 'payment.captured',
                'payment_id': payment_id,
                'subscription_id': subscription.subscription_id
            }
        
        except Exception as e:
            logger.error(f"Error handling payment.captured: {e}")
            return {
                'success': False,
                'message': f'Error processing payment: {str(e)}',
                'event': 'payment.captured'
            }
    
    def _handle_payment_failed(self, webhook_data: Dict) -> Dict[str, Any]:
        """Handle payment.failed webhook event"""
        try:
            payment_entity = webhook_data.get('payload', {}).get('payment', {}).get('entity', {})
            
            payment_id = payment_entity.get('id')
            order_id = payment_entity.get('order_id')
            amount = payment_entity.get('amount')
            error_code = payment_entity.get('error_code')
            error_description = payment_entity.get('error_description')
            
            # Extract user_id from notes
            notes = payment_entity.get('notes', {})
            user_id = notes.get('user_id')
            
            if user_id:
                # Record failed payment
                self.record_payment(
                    user_id=user_id,
                    razorpay_payment_id=payment_id,
                    razorpay_order_id=order_id,
                    amount=amount,
                    currency='INR',
                    status='failed',
                    payment_type='subscription',
                    metadata={
                        'error_code': error_code,
                        'error_description': error_description,
                        'webhook_event': 'payment.failed'
                    }
                )
                
                # Send failure notification email
                try:
                    db = SessionLocal()
                    user = db.query(User).filter_by(user_id=user_id).first()
                    if user:
                        email_service = get_email_service()
                        # Note: This method needs to be implemented in EmailService
                        if hasattr(email_service, 'send_payment_failed'):
                            email_service.send_payment_failed(
                                user_email=user.username,
                                amount=amount,
                                reason=error_description or 'Payment processing failed'
                            )
                    db.close()
                except Exception as e:
                    logger.error(f"Failed to send failure notification email: {e}")
            
            logger.info(f"Processed payment.failed for payment {payment_id}")
            
            return {
                'success': True,
                'message': 'Payment failure recorded',
                'event': 'payment.failed',
                'payment_id': payment_id
            }
        
        except Exception as e:
            logger.error(f"Error handling payment.failed: {e}")
            return {
                'success': False,
                'message': f'Error processing payment failure: {str(e)}',
                'event': 'payment.failed'
            }
    
    def _handle_subscription_activated(self, webhook_data: Dict) -> Dict[str, Any]:
        """Handle subscription.activated webhook event"""
        logger.info("Subscription activated event received")
        return {
            'success': True,
            'message': 'Subscription activation acknowledged',
            'event': 'subscription.activated'
        }
    
    def _handle_subscription_charged(self, webhook_data: Dict) -> Dict[str, Any]:
        """Handle subscription.charged webhook event"""
        logger.info("Subscription charged event received")
        return {
            'success': True,
            'message': 'Subscription charge acknowledged',
            'event': 'subscription.charged'
        }
    
    def _handle_subscription_cancelled(self, webhook_data: Dict) -> Dict[str, Any]:
        """Handle subscription.cancelled webhook event"""
        logger.info("Subscription cancelled event received")
        return {
            'success': True,
            'message': 'Subscription cancellation acknowledged',
            'event': 'subscription.cancelled'
        }
    
    def _handle_order_paid(self, webhook_data: Dict) -> Dict[str, Any]:
        """Handle order.paid webhook event"""
        logger.info("Order paid event received")
        return {
            'success': True,
            'message': 'Order payment acknowledged',
            'event': 'order.paid'
        }
    
    def process_upgrade_payment(
        self,
        user_id: str,
        new_tier: str
    ) -> Dict[str, Any]:
        """
        Process upgrade payment with proration.
        Calculates prorated amount, creates Razorpay order, and prepares for payment capture.
        
        Args:
            user_id: User ID
            new_tier: New tier to upgrade to
            
        Returns:
            Dict with upgrade payment details:
            {
                'success': bool,
                'message': str,
                'order_id': str (if successful),
                'amount': int (prorated amount in paise),
                'tier': str,
                'error': str (if failed)
            }
        """
        try:
            # Import here to avoid circular dependency
            from services.subscription_service import get_subscription_service
            
            # Get subscription service
            sub_service = get_subscription_service()
            
            # Get current subscription
            subscription_info = sub_service.get_user_subscription(user_id)
            
            if not subscription_info:
                return {
                    'success': False,
                    'message': 'No active subscription found',
                    'error': 'no_subscription'
                }
            
            # Validate it's an upgrade
            from services.tier_config import is_upgrade
            
            if not is_upgrade(subscription_info.tier, new_tier):
                return {
                    'success': False,
                    'message': f'Cannot upgrade from {subscription_info.tier} to {new_tier}',
                    'error': 'invalid_upgrade'
                }
            
            # Calculate prorated amount
            days_remaining = subscription_info.days_remaining
            prorated_amount = sub_service.calculate_prorated_credit(
                subscription_info.tier,
                new_tier,
                days_remaining
            )
            
            if prorated_amount <= 0:
                return {
                    'success': False,
                    'message': 'Invalid prorated amount calculated',
                    'error': 'invalid_amount'
                }
            
            # Create Razorpay order for the prorated difference
            receipt = f"upgrade_{subscription_info.tier}_to_{new_tier}_{user_id}_{int(datetime.now().timestamp())}"
            
            notes = {
                'user_id': user_id,
                'tier': new_tier,
                'old_tier': subscription_info.tier,
                'order_type': 'upgrade',
                'days_remaining': days_remaining,
                'prorated': True
            }
            
            order = self.create_order(
                amount=prorated_amount,
                currency='INR',
                receipt=receipt,
                notes=notes
            )
            
            logger.info(f"Created upgrade order for user {user_id}: {subscription_info.tier} -> {new_tier}, amount=₹{prorated_amount/100}")
            
            return {
                'success': True,
                'message': 'Upgrade order created',
                'order_id': order['order_id'],
                'amount': prorated_amount,
                'amount_inr': prorated_amount / 100,
                'tier': new_tier,
                'old_tier': subscription_info.tier,
                'days_remaining': days_remaining
            }
        
        except Exception as e:
            logger.error(f"Error processing upgrade payment for user {user_id}: {e}")
            return {
                'success': False,
                'message': f'Error processing upgrade: {str(e)}',
                'error': 'processing_error'
            }
    
    def capture_upgrade_payment(
        self,
        user_id: str,
        new_tier: str,
        razorpay_payment_id: str,
        razorpay_order_id: str,
        razorpay_signature: str
    ) -> Dict[str, Any]:
        """
        Capture upgrade payment and update subscription.
        Verifies payment signature and activates the new tier.
        
        Args:
            user_id: User ID
            new_tier: New tier to upgrade to
            razorpay_payment_id: Razorpay payment ID
            razorpay_order_id: Razorpay order ID
            razorpay_signature: Payment signature for verification
            
        Returns:
            Dict with capture result
        """
        try:
            # Verify payment signature
            if not self.verify_payment_signature(
                razorpay_order_id,
                razorpay_payment_id,
                razorpay_signature
            ):
                logger.error(f"Payment signature verification failed for upgrade: user={user_id}")
                return {
                    'success': False,
                    'message': 'Payment signature verification failed',
                    'error': 'signature_verification_failed'
                }
            
            # Import here to avoid circular dependency
            from services.subscription_service import get_subscription_service
            
            # Upgrade subscription
            sub_service = get_subscription_service()
            upgrade_result = sub_service.upgrade_subscription(user_id, new_tier)
            
            if not upgrade_result.success:
                logger.error(f"Failed to upgrade subscription: {upgrade_result.message}")
                return {
                    'success': False,
                    'message': upgrade_result.message,
                    'error': 'subscription_upgrade_failed'
                }
            
            # Get subscription to record payment
            subscription_info = sub_service.get_user_subscription(user_id)
            
            # Record payment
            db = SessionLocal()
            try:
                # Fetch payment details from Razorpay
                payment_details = self.client.payment.fetch(razorpay_payment_id)
                
                self.record_payment(
                    user_id=user_id,
                    razorpay_payment_id=razorpay_payment_id,
                    razorpay_order_id=razorpay_order_id,
                    amount=payment_details.get('amount', 0),
                    currency='INR',
                    status='captured',
                    method=payment_details.get('method'),
                    payment_type='upgrade',
                    subscription_id=subscription_info.subscription_id if subscription_info else None,
                    metadata={
                        'new_tier': new_tier,
                        'prorated_amount': upgrade_result.prorated_amount,
                        'upgrade': True
                    }
                )
                
                # Send upgrade confirmation email
                user = db.query(User).filter_by(user_id=user_id).first()
                if user:
                    email_service = get_email_service()
                    if hasattr(email_service, 'send_upgrade_confirmation'):
                        email_service.send_upgrade_confirmation(
                            user_email=user.username,
                            old_tier=payment_details.get('notes', {}).get('old_tier', 'previous'),
                            new_tier=new_tier
                        )
                
                db.close()
            except Exception as e:
                logger.error(f"Error recording upgrade payment: {e}")
                if db:
                    db.close()
            
            logger.info(f"Successfully captured upgrade payment for user {user_id}: tier={new_tier}")
            
            return {
                'success': True,
                'message': 'Upgrade completed successfully',
                'tier': new_tier,
                'subscription_id': subscription_info.subscription_id if subscription_info else None
            }
        
        except Exception as e:
            logger.error(f"Error capturing upgrade payment for user {user_id}: {e}")
            return {
                'success': False,
                'message': f'Error capturing payment: {str(e)}',
                'error': 'capture_error'
            }
    
    def activate_subscription(
        self,
        user_id: str,
        razorpay_subscription_id: str,
        tier: str,
        duration: str,
        amount: int
    ) -> bool:
        """
        Activate user subscription in database.
        
        Args:
            user_id: User ID
            razorpay_subscription_id: Razorpay subscription ID
            tier: Subscription tier
            duration: Duration (monthly/yearly)
            amount: Amount paid in paise
            
        Returns:
            True if successful
        """
        db = SessionLocal()
        try:
            # Calculate end date
            if duration == self.DURATION_MONTHLY:
                end_date = datetime.utcnow() + timedelta(days=30)
            else:  # yearly
                end_date = datetime.utcnow() + timedelta(days=365)
            
            # Check if subscription exists
            subscription = db.query(UserSubscription).filter_by(user_id=user_id).first()
            
            if subscription:
                # Update existing subscription
                subscription.tier = tier
                subscription.status = 'active'
                subscription.razorpay_subscription_id = razorpay_subscription_id
                subscription.start_date = datetime.utcnow()
                subscription.end_date = end_date
                subscription.auto_renew = True
            else:
                # Create new subscription
                subscription = UserSubscription(
                    user_id=user_id,
                    tier=tier,
                    status='active',
                    razorpay_subscription_id=razorpay_subscription_id,
                    start_date=datetime.utcnow(),
                    end_date=end_date,
                    auto_renew=True
                )
                db.add(subscription)
            
            db.commit()
            logger.info(f"Activated subscription for user {user_id}: {tier}")
            
            # Send email notification
            try:
                user = db.query(User).filter_by(user_id=user_id).first()
                if user:
                    email_service = get_email_service()
                    email_service.send_subscription_activated(
                        user_email=user.username,  # Assuming username is email
                        username=user.username,
                        subscription_details={
                            'tier': tier.capitalize(),
                            'end_date': end_date.strftime('%B %d, %Y')
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to send subscription email: {e}")
            
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to activate subscription: {e}")
            return False
        finally:
            db.close()
    
    def cancel_subscription(self, user_id: str) -> bool:
        """
        Cancel user subscription.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful
        """
        db = SessionLocal()
        try:
            subscription = db.query(UserSubscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                logger.warning(f"No subscription found for user {user_id}")
                return False
            
            # Cancel in Razorpay if exists
            if subscription.razorpay_subscription_id and self.client:
                try:
                    self.client.subscription.cancel(subscription.razorpay_subscription_id)
                except Exception as e:
                    logger.error(f"Failed to cancel Razorpay subscription: {e}")
            
            # Update database
            subscription.status = 'cancelled'
            subscription.auto_renew = False
            subscription.cancelled_at = datetime.utcnow()
            
            db.commit()
            logger.info(f"Cancelled subscription for user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to cancel subscription: {e}")
            return False
        finally:
            db.close()
    
    def record_payment(
        self,
        user_id: str,
        razorpay_payment_id: str,
        razorpay_order_id: Optional[str],
        amount: int,
        currency: str,
        status: str,
        method: Optional[str] = None,
        payment_type: str = 'subscription',
        subscription_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Record payment in database with enhanced metadata support.
        
        Args:
            user_id: User ID
            razorpay_payment_id: Razorpay payment ID
            razorpay_order_id: Razorpay order ID
            amount: Amount in paise
            currency: Currency code
            status: Payment status (pending, captured, failed, refunded)
            method: Payment method (card, upi, netbanking, wallet)
            payment_type: Type of payment (subscription, upgrade, one_time)
            subscription_id: Associated subscription ID
            metadata: Additional payment information as dictionary
            
        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            payment = Payment(
                user_id=user_id,
                razorpay_payment_id=razorpay_payment_id,
                razorpay_order_id=razorpay_order_id,
                amount=amount,
                currency=currency,
                status=status,
                payment_method=method,
                payment_type=payment_type,
                subscription_id=subscription_id,
                payment_metadata=metadata or {}
            )
            
            db.add(payment)
            db.commit()
            
            logger.info(f"Recorded {payment_type} payment {razorpay_payment_id} for user {user_id}: status={status}, amount=₹{amount/100}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to record payment: {e}")
            return False
        finally:
            db.close()
    
    def update_payment_status(
        self,
        razorpay_payment_id: str,
        new_status: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update payment status in database.
        
        Args:
            razorpay_payment_id: Razorpay payment ID
            new_status: New payment status (captured, failed, refunded)
            metadata: Additional metadata to merge with existing
            
        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            payment = db.query(Payment).filter_by(
                razorpay_payment_id=razorpay_payment_id
            ).first()
            
            if not payment:
                logger.warning(f"Payment not found: {razorpay_payment_id}")
                return False
            
            # Validate status transition
            valid_transitions = {
                'pending': ['captured', 'failed'],
                'captured': ['refunded'],
                'failed': [],
                'refunded': []
            }
            
            if new_status not in valid_transitions.get(payment.status, []):
                logger.error(f"Invalid status transition: {payment.status} -> {new_status}")
                return False
            
            # Update status
            payment.status = new_status
            
            # Merge metadata if provided
            if metadata:
                current_metadata = payment.payment_metadata or {}
                current_metadata.update(metadata)
                payment.payment_metadata = current_metadata
            
            db.commit()
            
            logger.info(f"Updated payment {razorpay_payment_id} status: {payment.status} -> {new_status}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update payment status: {e}")
            return False
        finally:
            db.close()
    
    def record_refund(
        self,
        razorpay_payment_id: str,
        refund_amount: int,
        reason: str,
        razorpay_refund_id: Optional[str] = None
    ) -> bool:
        """
        Record a refund for a payment.
        Updates payment status to 'refunded' and adds refund metadata.
        
        Args:
            razorpay_payment_id: Original payment ID
            refund_amount: Amount refunded in paise
            reason: Reason for refund
            razorpay_refund_id: Razorpay refund ID
            
        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            payment = db.query(Payment).filter_by(
                razorpay_payment_id=razorpay_payment_id
            ).first()
            
            if not payment:
                logger.warning(f"Payment not found for refund: {razorpay_payment_id}")
                return False
            
            if payment.status != 'captured':
                logger.error(f"Cannot refund payment with status: {payment.status}")
                return False
            
            # Update payment status and add refund metadata
            payment.status = 'refunded'
            
            refund_metadata = {
                'refund_amount': refund_amount,
                'refund_reason': reason,
                'refund_id': razorpay_refund_id,
                'refunded_at': datetime.utcnow().isoformat()
            }
            
            current_metadata = payment.payment_metadata or {}
            current_metadata.update(refund_metadata)
            payment.payment_metadata = current_metadata
            
            db.commit()
            
            logger.info(f"Recorded refund for payment {razorpay_payment_id}: amount=₹{refund_amount/100}, reason={reason}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to record refund: {e}")
            return False
        finally:
            db.close()
    
    def issue_refund(
        self,
        razorpay_payment_id: str,
        amount: Optional[int] = None,
        reason: str = 'Customer request'
    ) -> Dict[str, Any]:
        """
        Issue a refund through Razorpay and record it in database.
        
        Args:
            razorpay_payment_id: Payment ID to refund
            amount: Amount to refund in paise (None for full refund)
            reason: Reason for refund
            
        Returns:
            Dict with refund result:
            {
                'success': bool,
                'message': str,
                'refund_id': str (if successful),
                'amount': int (if successful)
            }
        """
        if not self.client:
            return {
                'success': False,
                'message': 'Razorpay not configured',
                'error': 'razorpay_not_configured'
            }
        
        try:
            # Create refund in Razorpay
            refund_data = {
                'notes': {
                    'reason': reason
                }
            }
            
            if amount is not None:
                refund_data['amount'] = amount
            
            refund = self.client.payment.refund(razorpay_payment_id, refund_data)
            
            refund_id = refund.get('id')
            refund_amount = refund.get('amount')
            
            # Record refund in database
            self.record_refund(
                razorpay_payment_id=razorpay_payment_id,
                refund_amount=refund_amount,
                reason=reason,
                razorpay_refund_id=refund_id
            )
            
            logger.info(f"Issued refund {refund_id} for payment {razorpay_payment_id}: ₹{refund_amount/100}")
            
            return {
                'success': True,
                'message': 'Refund issued successfully',
                'refund_id': refund_id,
                'amount': refund_amount,
                'amount_inr': refund_amount / 100
            }
            
        except Exception as e:
            logger.error(f"Failed to issue refund: {e}")
            return {
                'success': False,
                'message': f'Failed to issue refund: {str(e)}',
                'error': 'refund_failed'
            }
    
    def get_payment_by_id(self, razorpay_payment_id: str) -> Optional[Dict]:
        """
        Get payment details by Razorpay payment ID.
        
        Args:
            razorpay_payment_id: Razorpay payment ID
            
        Returns:
            Payment dict or None if not found
        """
        db = SessionLocal()
        try:
            payment = db.query(Payment).filter_by(
                razorpay_payment_id=razorpay_payment_id
            ).first()
            
            if not payment:
                return None
            
            return payment.to_dict()
            
        finally:
            db.close()
    
    def get_user_subscription(self, user_id: str) -> Optional[Dict]:
        """
        Get user's current subscription.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict with subscription details or None
        """
        db = SessionLocal()
        try:
            subscription = db.query(UserSubscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                return None
            
            return subscription.to_dict()
            
        finally:
            db.close()
    
    def get_user_payments(self, user_id: str, limit: int = 10) -> list:
        """
        Get user's payment history.
        
        Args:
            user_id: User ID
            limit: Maximum number of payments to return
            
        Returns:
            List of payment dicts
        """
        db = SessionLocal()
        try:
            payments = db.query(Payment).filter_by(
                user_id=user_id
            ).order_by(
                Payment.created_at.desc()
            ).limit(limit).all()
            
            return [p.to_dict() for p in payments]
            
        finally:
            db.close()
    
    def get_pricing_info(self) -> Dict:
        """
        Get pricing information for all plans.
        
        Returns:
            Dict with pricing details
        """
        return {
            'free': {
                'name': 'Free',
                'monthly': 0,
                'yearly': 0,
                'features': [
                    '10 questions per day',
                    'Basic explanations',
                    'NCERT content access',
                    'Community support'
                ],
                'limits': {
                    'questions_per_day': 10,
                    'detailed_explanations': False,
                    'previous_papers': False,
                    'mock_tests': False,
                    'progress_tracking': False
                }
            },
            'basic': {
                'name': 'Basic',
                'monthly': 199,
                'yearly': 1999,
                'yearly_savings': 17,
                'features': [
                    '100 questions per day',
                    'Detailed explanations',
                    'Previous year papers',
                    'Chapter-wise tests',
                    'Progress tracking',
                    'Email support'
                ],
                'limits': {
                    'questions_per_day': 100,
                    'detailed_explanations': True,
                    'previous_papers': True,
                    'mock_tests': False,
                    'progress_tracking': True
                },
                'popular': False
            },
            'premium': {
                'name': 'Premium',
                'monthly': 499,
                'yearly': 4999,
                'yearly_savings': 17,
                'features': [
                    'Unlimited questions',
                    'AI tutor 24/7',
                    'All previous papers',
                    'Full-length mock tests',
                    'Advanced analytics',
                    'Doubt solving',
                    'Priority support',
                    'Offline access'
                ],
                'limits': {
                    'questions_per_day': -1,  # unlimited
                    'detailed_explanations': True,
                    'previous_papers': True,
                    'mock_tests': True,
                    'progress_tracking': True,
                    'ai_tutor': True,
                    'doubt_solving': True
                },
                'popular': True
            }
        }


# Global instance
_payment_service = None


def get_payment_service() -> PaymentService:
    """Get or create PaymentService instance"""
    global _payment_service
    if _payment_service is None:
        _payment_service = PaymentService()
    return _payment_service
