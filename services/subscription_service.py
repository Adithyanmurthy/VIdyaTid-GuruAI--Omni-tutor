"""
Subscription Service for VidyaTid.

Manages subscription lifecycle, tier changes, and status checks.
Handles subscription creation, upgrades, downgrades, cancellations, and renewals.
"""
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from models.database import SessionLocal
from models.user import User
from models.subscription import Subscription
from models.usage import Usage
from services.tier_config import (
    get_tier_config,
    get_tier_features,
    get_tier_limits,
    get_tier_price,
    get_queries_per_day,
    get_predictions_per_month,
    validate_tier,
    is_upgrade,
    is_downgrade,
    compare_tiers
)

logger = logging.getLogger(__name__)


class SubscriptionInfo:
    """Data class for subscription information"""
    
    def __init__(self, subscription: Subscription):
        self.subscription_id = subscription.subscription_id
        self.user_id = subscription.user_id
        self.tier = subscription.tier
        self.status = subscription.status
        self.start_date = subscription.start_date
        self.end_date = subscription.end_date
        self.auto_renew = subscription.auto_renew
        self.is_active = subscription.is_active()
        self.days_remaining = subscription.days_remaining()
        self.cancelled_at = subscription.cancelled_at
        self.scheduled_tier_change = subscription.scheduled_tier_change
        self.scheduled_change_date = subscription.scheduled_change_date
        self.razorpay_subscription_id = subscription.razorpay_subscription_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'subscription_id': self.subscription_id,
            'user_id': self.user_id,
            'tier': self.tier,
            'status': self.status,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'auto_renew': self.auto_renew,
            'is_active': self.is_active,
            'days_remaining': self.days_remaining,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'scheduled_tier_change': self.scheduled_tier_change,
            'scheduled_change_date': self.scheduled_change_date.isoformat() if self.scheduled_change_date else None,
            'razorpay_subscription_id': self.razorpay_subscription_id
        }


class UpgradeResult:
    """Result of an upgrade operation"""
    
    def __init__(self, success: bool, message: str, prorated_amount: int = 0, 
                 new_end_date: Optional[datetime] = None):
        self.success = success
        self.message = message
        self.prorated_amount = prorated_amount
        self.new_end_date = new_end_date
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'message': self.message,
            'prorated_amount': self.prorated_amount,
            'new_end_date': self.new_end_date.isoformat() if self.new_end_date else None
        }


class DowngradeResult:
    """Result of a downgrade operation"""
    
    def __init__(self, success: bool, message: str, scheduled_date: Optional[datetime] = None):
        self.success = success
        self.message = message
        self.scheduled_date = scheduled_date
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'message': self.message,
            'scheduled_date': self.scheduled_date.isoformat() if self.scheduled_date else None
        }


class SubscriptionService:
    """Service for managing user subscriptions"""
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize SubscriptionService.
        
        Args:
            db: Optional database session. If not provided, creates a new session.
        """
        self.db = db
        self._owns_session = db is None
        if self._owns_session:
            self.db = SessionLocal()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._owns_session and self.db:
            if exc_type is not None:
                self.db.rollback()
            self.db.close()
    
    def close(self):
        """Close database session if owned"""
        if self._owns_session and self.db:
            self.db.close()
    
    def get_user_subscription(self, user_id: str) -> Optional[SubscriptionInfo]:
        """
        Get user's active subscription.
        
        Args:
            user_id: User ID
            
        Returns:
            SubscriptionInfo object or None if no subscription exists
        """
        try:
            subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                logger.info(f"No subscription found for user {user_id}")
                return None
            
            return SubscriptionInfo(subscription)
            
        except Exception as e:
            logger.error(f"Error fetching subscription for user {user_id}: {e}")
            return None
    
    def create_subscription(
        self,
        user_id: str,
        tier: str,
        duration_days: int,
        razorpay_sub_id: Optional[str] = None
    ) -> Optional[Subscription]:
        """
        Create a new subscription for a user.
        
        Args:
            user_id: User ID
            tier: Subscription tier (free, starter, premium, ultimate)
            duration_days: Duration in days
            razorpay_sub_id: Optional Razorpay subscription ID
            
        Returns:
            Created Subscription object or None on failure
        """
        try:
            # Validate tier
            if not validate_tier(tier):
                logger.error(f"Invalid tier: {tier}")
                return None
            
            # Check if user exists
            user = self.db.query(User).filter_by(user_id=user_id).first()
            if not user:
                logger.error(f"User not found: {user_id}")
                return None
            
            # Calculate dates
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=duration_days)
            
            # Check if subscription already exists
            existing_sub = self.db.query(Subscription).filter_by(user_id=user_id).first()
            
            if existing_sub:
                # Update existing subscription
                existing_sub.tier = tier
                existing_sub.status = 'active'
                existing_sub.razorpay_subscription_id = razorpay_sub_id
                existing_sub.start_date = start_date
                existing_sub.end_date = end_date
                existing_sub.auto_renew = True
                existing_sub.cancelled_at = None
                existing_sub.scheduled_tier_change = None
                existing_sub.scheduled_change_date = None
                existing_sub.updated_at = datetime.utcnow()
                
                subscription = existing_sub
                logger.info(f"Updated subscription for user {user_id} to tier {tier}")
            else:
                # Create new subscription
                subscription = Subscription(
                    user_id=user_id,
                    tier=tier,
                    status='active',
                    razorpay_subscription_id=razorpay_sub_id,
                    start_date=start_date,
                    end_date=end_date,
                    auto_renew=True
                )
                self.db.add(subscription)
                logger.info(f"Created new subscription for user {user_id} with tier {tier}")
            
            # Create or update usage record for today
            today = datetime.utcnow().date()
            usage = self.db.query(Usage).filter_by(
                user_id=user_id,
                date=today
            ).first()
            
            queries_limit = get_queries_per_day(tier)
            predictions_limit = get_predictions_per_month(tier)
            
            if usage:
                usage.queries_limit = queries_limit
                usage.predictions_limit = predictions_limit
                usage.subscription_id = subscription.subscription_id
                usage.updated_at = datetime.utcnow()
            else:
                usage = Usage(
                    user_id=user_id,
                    queries_limit=queries_limit,
                    predictions_limit=predictions_limit,
                    subscription_id=subscription.subscription_id,
                    date=today
                )
                self.db.add(usage)
            
            self.db.commit()
            self.db.refresh(subscription)
            
            return subscription
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating subscription for user {user_id}: {e}")
            return None
    
    def get_tier_features(self, tier: str) -> Optional[Dict[str, Any]]:
        """
        Get features available for a specific tier.
        
        Args:
            tier: Subscription tier
            
        Returns:
            Dictionary of features or None if tier is invalid
        """
        return get_tier_features(tier)
    
    def validate_subscription_status(self, user_id: str) -> bool:
        """
        Validate and update subscription status if needed.
        
        Args:
            user_id: User ID
            
        Returns:
            True if subscription is active, False otherwise
        """
        try:
            subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                return False
            
            # Check if subscription has expired
            if subscription.is_expired() and subscription.status == 'active':
                subscription.status = 'expired'
                subscription.updated_at = datetime.utcnow()
                self.db.commit()
                logger.info(f"Marked subscription as expired for user {user_id}")
                return False
            
            return subscription.is_active()
            
        except Exception as e:
            logger.error(f"Error validating subscription status for user {user_id}: {e}")
            return False
    
    def check_and_expire_subscriptions(self) -> int:
        """
        Find and expire old subscriptions.
        Also executes scheduled downgrades.
        
        Returns:
            Number of subscriptions expired/updated
        """
        try:
            count = 0
            now = datetime.utcnow()
            
            # Find all active subscriptions that have passed their end_date
            expired_subs = self.db.query(Subscription).filter(
                Subscription.status == 'active',
                Subscription.end_date <= now
            ).all()
            
            for subscription in expired_subs:
                # Check if there's a scheduled downgrade
                if subscription.scheduled_tier_change and subscription.scheduled_change_date:
                    if subscription.scheduled_change_date <= now:
                        # Execute scheduled downgrade
                        old_tier = subscription.tier
                        subscription.tier = subscription.scheduled_tier_change
                        subscription.scheduled_tier_change = None
                        subscription.scheduled_change_date = None
                        subscription.status = 'active'
                        # Extend subscription for another period (30 days for monthly)
                        subscription.start_date = now
                        subscription.end_date = now + timedelta(days=30)
                        subscription.updated_at = now
                        
                        logger.info(f"Executed scheduled downgrade for user {subscription.user_id}: {old_tier} -> {subscription.tier}")
                        count += 1
                    else:
                        # Scheduled downgrade not yet due, keep active
                        continue
                else:
                    # No scheduled downgrade, mark as expired
                    subscription.status = 'expired'
                    subscription.updated_at = now
                    
                    # If cancelled, downgrade to free tier
                    if subscription.cancelled_at is not None:
                        subscription.tier = 'free'
                        logger.info(f"Downgraded cancelled subscription to free tier for user {subscription.user_id}")
                    
                    logger.info(f"Expired subscription for user {subscription.user_id}")
                    count += 1
            
            if count > 0:
                self.db.commit()
                logger.info(f"Processed {count} expired/scheduled subscriptions")
            
            return count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error checking and expiring subscriptions: {e}")
            return 0
    
    def renew_subscription(self, user_id: str, duration_days: int = 30) -> bool:
        """
        Manually renew a subscription.
        
        Args:
            user_id: User ID
            duration_days: Duration to extend (default 30 days)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                logger.error(f"No subscription found for user {user_id}")
                return False
            
            # Calculate new end date
            # If subscription is still active, extend from current end_date
            # If expired, extend from now
            if subscription.is_active():
                new_end_date = subscription.end_date + timedelta(days=duration_days)
            else:
                subscription.start_date = datetime.utcnow()
                new_end_date = subscription.start_date + timedelta(days=duration_days)
            
            subscription.end_date = new_end_date
            subscription.status = 'active'
            subscription.auto_renew = True
            subscription.updated_at = datetime.utcnow()
            
            self.db.commit()
            logger.info(f"Renewed subscription for user {user_id} until {new_end_date}")
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error renewing subscription for user {user_id}: {e}")
            return False
    
    def cancel_subscription(self, user_id: str) -> bool:
        """
        Cancel a user's subscription.
        Sets auto_renew to false and records cancellation timestamp.
        Subscription remains active until end_date.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                logger.error(f"No subscription found for user {user_id}")
                return False
            
            if subscription.status == 'cancelled':
                logger.warning(f"Subscription already cancelled for user {user_id}")
                return True
            
            # Set cancellation fields
            subscription.auto_renew = False
            subscription.cancelled_at = datetime.utcnow()
            subscription.status = 'cancelled'
            subscription.updated_at = datetime.utcnow()
            
            self.db.commit()
            logger.info(f"Cancelled subscription for user {user_id}. Access until {subscription.end_date}")
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error cancelling subscription for user {user_id}: {e}")
            return False
    
    def calculate_prorated_credit(
        self,
        current_tier: str,
        new_tier: str,
        days_remaining: int
    ) -> int:
        """
        Calculate prorated credit for an upgrade.
        
        Args:
            current_tier: Current subscription tier
            new_tier: New subscription tier
            days_remaining: Days remaining in current subscription
            
        Returns:
            Prorated amount in paise (can be negative for downgrades)
        """
        try:
            # Get monthly prices
            current_price = get_tier_price(current_tier, 'monthly')
            new_price = get_tier_price(new_tier, 'monthly')
            
            if current_price is None or new_price is None:
                logger.error(f"Invalid tier for proration: {current_tier} or {new_tier}")
                return 0
            
            # Calculate daily rates
            current_daily_rate = current_price / 30
            new_daily_rate = new_price / 30
            
            # Calculate credit from remaining days of current subscription
            credit = current_daily_rate * days_remaining
            
            # Calculate cost for remaining days at new tier
            new_cost = new_daily_rate * days_remaining
            
            # Prorated amount is the difference
            prorated_amount = int(new_cost - credit)
            
            logger.info(f"Prorated amount for {current_tier} -> {new_tier} with {days_remaining} days: ₹{prorated_amount/100:.2f}")
            
            return prorated_amount
            
        except Exception as e:
            logger.error(f"Error calculating prorated credit: {e}")
            return 0
    
    def upgrade_subscription(
        self,
        user_id: str,
        new_tier: str
    ) -> UpgradeResult:
        """
        Upgrade a user's subscription to a higher tier.
        Activates immediately with prorated pricing.
        
        Args:
            user_id: User ID
            new_tier: New tier to upgrade to
            
        Returns:
            UpgradeResult object
        """
        try:
            # Validate new tier
            if not validate_tier(new_tier):
                return UpgradeResult(
                    success=False,
                    message=f"Invalid tier: {new_tier}"
                )
            
            # Get current subscription
            subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                return UpgradeResult(
                    success=False,
                    message="No subscription found"
                )
            
            # Check if it's actually an upgrade
            if not is_upgrade(subscription.tier, new_tier):
                return UpgradeResult(
                    success=False,
                    message=f"Cannot upgrade from {subscription.tier} to {new_tier}"
                )
            
            # Calculate prorated amount
            days_remaining = subscription.days_remaining()
            prorated_amount = self.calculate_prorated_credit(
                subscription.tier,
                new_tier,
                days_remaining
            )
            
            # Update subscription
            old_tier = subscription.tier
            subscription.tier = new_tier
            subscription.status = 'active'
            subscription.updated_at = datetime.utcnow()
            
            # Clear any scheduled changes
            subscription.scheduled_tier_change = None
            subscription.scheduled_change_date = None
            
            # Update usage limits for today
            today = datetime.utcnow().date()
            usage = self.db.query(Usage).filter_by(
                user_id=user_id,
                date=today
            ).first()
            
            if usage:
                usage.queries_limit = get_queries_per_day(new_tier)
                usage.predictions_limit = get_predictions_per_month(new_tier)
                usage.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            logger.info(f"Upgraded subscription for user {user_id}: {old_tier} -> {new_tier}")
            
            return UpgradeResult(
                success=True,
                message=f"Successfully upgraded to {new_tier}",
                prorated_amount=prorated_amount,
                new_end_date=subscription.end_date
            )
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error upgrading subscription for user {user_id}: {e}")
            return UpgradeResult(
                success=False,
                message=f"Error upgrading subscription: {str(e)}"
            )
    
    def downgrade_subscription(
        self,
        user_id: str,
        new_tier: str
    ) -> DowngradeResult:
        """
        Schedule a downgrade to a lower tier.
        Downgrade takes effect at the end of the current billing cycle.
        
        Args:
            user_id: User ID
            new_tier: New tier to downgrade to
            
        Returns:
            DowngradeResult object
        """
        try:
            # Validate new tier
            if not validate_tier(new_tier):
                return DowngradeResult(
                    success=False,
                    message=f"Invalid tier: {new_tier}"
                )
            
            # Get current subscription
            subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                return DowngradeResult(
                    success=False,
                    message="No subscription found"
                )
            
            # Check if it's actually a downgrade
            if not is_downgrade(subscription.tier, new_tier):
                return DowngradeResult(
                    success=False,
                    message=f"Cannot downgrade from {subscription.tier} to {new_tier}"
                )
            
            # Schedule the downgrade
            subscription.scheduled_tier_change = new_tier
            subscription.scheduled_change_date = subscription.end_date
            subscription.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            logger.info(f"Scheduled downgrade for user {user_id}: {subscription.tier} -> {new_tier} on {subscription.end_date}")
            
            return DowngradeResult(
                success=True,
                message=f"Downgrade to {new_tier} scheduled for {subscription.end_date.strftime('%Y-%m-%d')}",
                scheduled_date=subscription.end_date
            )
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error scheduling downgrade for user {user_id}: {e}")
            return DowngradeResult(
                success=False,
                message=f"Error scheduling downgrade: {str(e)}"
            )


# Global instance
_subscription_service = None


def get_subscription_service(db: Optional[Session] = None) -> SubscriptionService:
    """
    Get or create SubscriptionService instance.
    
    Args:
        db: Optional database session
        
    Returns:
        SubscriptionService instance
    """
    if db is not None:
        return SubscriptionService(db)
    
    global _subscription_service
    if _subscription_service is None:
        _subscription_service = SubscriptionService()
    return _subscription_service
