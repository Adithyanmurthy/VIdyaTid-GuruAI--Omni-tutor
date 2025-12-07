"""
UsageTracker Service for VidyaTid.

Tracks daily query usage, enforces limits, and handles resets.
Manages usage counters, daily resets, and usage statistics.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from models.database import SessionLocal
from models.user import User
from models.subscription import Subscription
from models.usage import Usage
from services.tier_config import (
    get_queries_per_day,
    get_predictions_per_month,
    validate_tier
)

logger = logging.getLogger(__name__)


class UsageInfo:
    """Data class for usage information"""
    
    def __init__(self, usage: Usage):
        self.usage_id = usage.usage_id
        self.user_id = usage.user_id
        self.subscription_id = usage.subscription_id
        self.date = usage.date
        self.query_count = usage.query_count
        self.queries_limit = usage.queries_limit
        self.queries_remaining = usage.get_remaining_queries()
        self.prediction_count = usage.prediction_count
        self.predictions_limit = usage.predictions_limit
        self.predictions_remaining = usage.get_remaining_predictions()
        self.feature_usage = usage.feature_usage
        self.created_at = usage.created_at
        self.updated_at = usage.updated_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'usage_id': self.usage_id,
            'user_id': self.user_id,
            'subscription_id': self.subscription_id,
            'date': self.date.isoformat() if self.date else None,
            'query_count': self.query_count,
            'queries_limit': self.queries_limit,
            'queries_remaining': self.queries_remaining,
            'prediction_count': self.prediction_count,
            'predictions_limit': self.predictions_limit,
            'predictions_remaining': self.predictions_remaining,
            'feature_usage': self.feature_usage,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class LimitCheckResult:
    """Result of a limit check operation"""
    
    def __init__(self, allowed: bool, message: str, queries_remaining: int = 0):
        self.allowed = allowed
        self.message = message
        self.queries_remaining = queries_remaining
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'allowed': self.allowed,
            'message': self.message,
            'queries_remaining': self.queries_remaining
        }


class UsageStats:
    """Statistics about user usage"""
    
    def __init__(
        self,
        total_queries: int,
        total_predictions: int,
        avg_daily_queries: float,
        days_tracked: int,
        current_tier: str,
        queries_limit: int,
        predictions_limit: int
    ):
        self.total_queries = total_queries
        self.total_predictions = total_predictions
        self.avg_daily_queries = avg_daily_queries
        self.days_tracked = days_tracked
        self.current_tier = current_tier
        self.queries_limit = queries_limit
        self.predictions_limit = predictions_limit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_queries': self.total_queries,
            'total_predictions': self.total_predictions,
            'avg_daily_queries': self.avg_daily_queries,
            'days_tracked': self.days_tracked,
            'current_tier': self.current_tier,
            'queries_limit': self.queries_limit,
            'predictions_limit': self.predictions_limit
        }


class UsageTracker:
    """Service for tracking and managing user usage"""
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize UsageTracker.
        
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
    
    def get_usage(self, user_id: str, usage_date: Optional[date] = None) -> Optional[UsageInfo]:
        """
        Get usage record for a user on a specific date.
        Creates a new record if one doesn't exist.
        
        Args:
            user_id: User ID
            usage_date: Date to get usage for (defaults to today)
            
        Returns:
            UsageInfo object or None on error
        """
        try:
            if usage_date is None:
                usage_date = datetime.utcnow().date()
            
            # Try to fetch existing usage record
            usage = self.db.query(Usage).filter_by(
                user_id=user_id,
                date=usage_date
            ).first()
            
            if usage:
                return UsageInfo(usage)
            
            # No usage record exists, create one
            # Get user's subscription to determine limits
            subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
            
            if not subscription:
                # No subscription, create free tier usage
                tier = 'free'
                subscription_id = None
            else:
                tier = subscription.tier
                subscription_id = subscription.subscription_id
            
            # Get limits for the tier
            queries_limit = get_queries_per_day(tier)
            predictions_limit = get_predictions_per_month(tier)
            
            # Create new usage record
            usage = Usage(
                user_id=user_id,
                queries_limit=queries_limit,
                predictions_limit=predictions_limit,
                subscription_id=subscription_id,
                date=usage_date
            )
            
            self.db.add(usage)
            self.db.commit()
            self.db.refresh(usage)
            
            logger.info(f"Created usage record for user {user_id} on {usage_date}")
            
            return UsageInfo(usage)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error getting usage for user {user_id} on {usage_date}: {e}")
            return None
    
    def increment_usage(self, user_id: str, query_type: str = 'query') -> bool:
        """
        Increment usage counter for a user.
        Uses atomic database operation for thread safety.
        
        Args:
            user_id: User ID
            query_type: Type of query ('query' or 'prediction')
            
        Returns:
            True if increment was successful, False if limit reached or error
        """
        try:
            today = datetime.utcnow().date()
            
            # Get or create usage record
            usage = self.db.query(Usage).filter_by(
                user_id=user_id,
                date=today
            ).first()
            
            if not usage:
                # Create usage record first
                usage_info = self.get_usage(user_id, today)
                if not usage_info:
                    logger.error(f"Failed to create usage record for user {user_id}")
                    return False
                
                # Fetch the created record
                usage = self.db.query(Usage).filter_by(
                    user_id=user_id,
                    date=today
                ).first()
            
            # Check if increment is allowed
            if query_type == 'prediction':
                if not usage.has_predictions_remaining():
                    logger.warning(f"User {user_id} has reached prediction limit")
                    return False
                
                # Increment prediction counter
                success = usage.increment_prediction_count()
            else:
                if not usage.has_queries_remaining():
                    logger.warning(f"User {user_id} has reached query limit")
                    return False
                
                # Increment query counter
                success = usage.increment_query_count()
            
            if success:
                self.db.commit()
                logger.info(f"Incremented {query_type} count for user {user_id}")
                return True
            else:
                logger.warning(f"Failed to increment {query_type} count for user {user_id}")
                return False
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error incrementing usage for user {user_id}: {e}")
            return False
    
    def check_limit(self, user_id: str) -> LimitCheckResult:
        """
        Check if user can submit a query.
        
        Args:
            user_id: User ID
            
        Returns:
            LimitCheckResult object
        """
        try:
            today = datetime.utcnow().date()
            
            # Get usage record
            usage = self.db.query(Usage).filter_by(
                user_id=user_id,
                date=today
            ).first()
            
            if not usage:
                # No usage record, create one
                usage_info = self.get_usage(user_id, today)
                if not usage_info:
                    return LimitCheckResult(
                        allowed=False,
                        message="Error checking usage limit",
                        queries_remaining=0
                    )
                
                # Fetch the created record
                usage = self.db.query(Usage).filter_by(
                    user_id=user_id,
                    date=today
                ).first()
            
            # Check if queries are remaining
            if usage.has_queries_remaining():
                queries_remaining = usage.get_remaining_queries()
                return LimitCheckResult(
                    allowed=True,
                    message="Query allowed",
                    queries_remaining=queries_remaining
                )
            else:
                return LimitCheckResult(
                    allowed=False,
                    message="Daily query limit reached",
                    queries_remaining=0
                )
            
        except Exception as e:
            logger.error(f"Error checking limit for user {user_id}: {e}")
            # Fail open for better user experience
            return LimitCheckResult(
                allowed=True,
                message="Error checking limit, allowing query",
                queries_remaining=-1
            )
    
    def get_remaining_queries(self, user_id: str) -> int:
        """
        Get number of queries remaining for a user today.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of queries remaining (-1 for unlimited, 0 if limit reached)
        """
        try:
            today = datetime.utcnow().date()
            
            # Get usage record
            usage = self.db.query(Usage).filter_by(
                user_id=user_id,
                date=today
            ).first()
            
            if not usage:
                # No usage record, get user's tier limit
                subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
                tier = subscription.tier if subscription else 'free'
                return get_queries_per_day(tier)
            
            return usage.get_remaining_queries()
            
        except Exception as e:
            logger.error(f"Error getting remaining queries for user {user_id}: {e}")
            return 0

    
    def reset_daily_usage(self, reset_date: Optional[date] = None) -> int:
        """
        Reset daily usage counters for all users.
        Query all users and reset based on their tier limits.
        
        Args:
            reset_date: Date to reset for (defaults to today)
            
        Returns:
            Count of successfully reset users
        """
        try:
            if reset_date is None:
                reset_date = datetime.utcnow().date()
            
            count = 0
            failed_users = []
            
            # Get all users
            users = self.db.query(User).all()
            
            logger.info(f"Starting daily reset for {len(users)} users on {reset_date}")
            
            for user in users:
                try:
                    # Get user's subscription
                    subscription = self.db.query(Subscription).filter_by(
                        user_id=user.user_id
                    ).first()
                    
                    # Determine tier
                    tier = subscription.tier if subscription else 'free'
                    
                    # Get limits for the tier
                    queries_limit = get_queries_per_day(tier)
                    predictions_limit = get_predictions_per_month(tier)
                    
                    # Get or create usage record for the date
                    usage = self.db.query(Usage).filter_by(
                        user_id=user.user_id,
                        date=reset_date
                    ).first()
                    
                    if usage:
                        # Reset existing record
                        usage.reset_counters(
                            new_queries_limit=queries_limit,
                            new_predictions_limit=predictions_limit
                        )
                    else:
                        # Create new usage record
                        usage = Usage(
                            user_id=user.user_id,
                            queries_limit=queries_limit,
                            predictions_limit=predictions_limit,
                            subscription_id=subscription.subscription_id if subscription else None,
                            date=reset_date
                        )
                        self.db.add(usage)
                    
                    count += 1
                    
                except Exception as user_error:
                    logger.error(f"Error resetting usage for user {user.user_id}: {user_error}")
                    failed_users.append(user.user_id)
                    continue
            
            # Commit all changes
            self.db.commit()
            
            logger.info(f"Successfully reset {count} users. Failed: {len(failed_users)}")
            
            if failed_users:
                logger.warning(f"Failed to reset users: {failed_users}")
            
            return count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during daily reset: {e}")
            return 0
    
    def get_usage_history(self, user_id: str, days: int = 30) -> List[UsageInfo]:
        """
        Get usage history for a user.
        
        Args:
            user_id: User ID
            days: Number of days to fetch (default 30)
            
        Returns:
            List of UsageInfo objects
        """
        try:
            # Calculate date range
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days - 1)
            
            # Query usage records
            usage_records = self.db.query(Usage).filter(
                Usage.user_id == user_id,
                Usage.date >= start_date,
                Usage.date <= end_date
            ).order_by(Usage.date.desc()).all()
            
            # Convert to UsageInfo objects
            history = [UsageInfo(usage) for usage in usage_records]
            
            logger.info(f"Retrieved {len(history)} usage records for user {user_id}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting usage history for user {user_id}: {e}")
            return []
    
    def get_usage_stats(self, user_id: str) -> Optional[UsageStats]:
        """
        Get usage statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            UsageStats object or None on error
        """
        try:
            # Get user's subscription
            subscription = self.db.query(Subscription).filter_by(user_id=user_id).first()
            tier = subscription.tier if subscription else 'free'
            
            # Get limits
            queries_limit = get_queries_per_day(tier)
            predictions_limit = get_predictions_per_month(tier)
            
            # Calculate date range (last 30 days)
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=29)
            
            # Query usage records
            usage_records = self.db.query(Usage).filter(
                Usage.user_id == user_id,
                Usage.date >= start_date,
                Usage.date <= end_date
            ).all()
            
            # Calculate statistics
            total_queries = sum(usage.query_count for usage in usage_records)
            total_predictions = sum(usage.prediction_count for usage in usage_records)
            days_tracked = len(usage_records)
            avg_daily_queries = total_queries / days_tracked if days_tracked > 0 else 0
            
            stats = UsageStats(
                total_queries=total_queries,
                total_predictions=total_predictions,
                avg_daily_queries=round(avg_daily_queries, 2),
                days_tracked=days_tracked,
                current_tier=tier,
                queries_limit=queries_limit,
                predictions_limit=predictions_limit
            )
            
            logger.info(f"Generated usage stats for user {user_id}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting usage stats for user {user_id}: {e}")
            return None
    
    def track_prediction_usage(self, user_id: str) -> bool:
        """
        Track prediction usage for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.increment_usage(user_id, query_type='prediction')
    
    def reset_monthly_predictions(self, reset_date: Optional[date] = None) -> int:
        """
        Reset monthly prediction counters for all users.
        
        Args:
            reset_date: Date to reset for (defaults to today)
            
        Returns:
            Count of successfully reset users
        """
        try:
            if reset_date is None:
                reset_date = datetime.utcnow().date()
            
            count = 0
            
            # Get all users
            users = self.db.query(User).all()
            
            logger.info(f"Starting monthly prediction reset for {len(users)} users on {reset_date}")
            
            for user in users:
                try:
                    # Get user's subscription
                    subscription = self.db.query(Subscription).filter_by(
                        user_id=user.user_id
                    ).first()
                    
                    # Determine tier
                    tier = subscription.tier if subscription else 'free'
                    
                    # Get prediction limit for the tier
                    predictions_limit = get_predictions_per_month(tier)
                    
                    # Get or create usage record for the date
                    usage = self.db.query(Usage).filter_by(
                        user_id=user.user_id,
                        date=reset_date
                    ).first()
                    
                    if usage:
                        # Reset prediction counter only
                        usage.prediction_count = 0
                        usage.predictions_limit = predictions_limit
                        usage.updated_at = datetime.utcnow()
                    else:
                        # Create new usage record
                        queries_limit = get_queries_per_day(tier)
                        usage = Usage(
                            user_id=user.user_id,
                            queries_limit=queries_limit,
                            predictions_limit=predictions_limit,
                            subscription_id=subscription.subscription_id if subscription else None,
                            date=reset_date
                        )
                        self.db.add(usage)
                    
                    count += 1
                    
                except Exception as user_error:
                    logger.error(f"Error resetting predictions for user {user.user_id}: {user_error}")
                    continue
            
            # Commit all changes
            self.db.commit()
            
            logger.info(f"Successfully reset predictions for {count} users")
            
            return count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during monthly prediction reset: {e}")
            return 0
    
    def check_usage_warning(self, user_id: str) -> Dict[str, Any]:
        """
        Check if user has reached 80% usage threshold.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with warning status and details
        """
        try:
            today = datetime.utcnow().date()
            
            # Get usage record
            usage = self.db.query(Usage).filter_by(
                user_id=user_id,
                date=today
            ).first()
            
            if not usage:
                return {
                    'warning': False,
                    'message': 'No usage data available',
                    'usage_percentage': 0
                }
            
            # Calculate usage percentage
            if usage.queries_limit == -1:
                # Unlimited queries
                return {
                    'warning': False,
                    'message': 'Unlimited queries',
                    'usage_percentage': 0
                }
            
            usage_percentage = (usage.query_count / usage.queries_limit) * 100 if usage.queries_limit > 0 else 0
            
            # Check if warning threshold reached (80%)
            if usage_percentage >= 80:
                queries_remaining = usage.get_remaining_queries()
                return {
                    'warning': True,
                    'message': f'You have used {usage_percentage:.0f}% of your daily queries',
                    'usage_percentage': round(usage_percentage, 2),
                    'queries_remaining': queries_remaining,
                    'queries_used': usage.query_count,
                    'queries_limit': usage.queries_limit
                }
            else:
                return {
                    'warning': False,
                    'message': 'Usage within normal range',
                    'usage_percentage': round(usage_percentage, 2),
                    'queries_remaining': usage.get_remaining_queries(),
                    'queries_used': usage.query_count,
                    'queries_limit': usage.queries_limit
                }
            
        except Exception as e:
            logger.error(f"Error checking usage warning for user {user_id}: {e}")
            return {
                'warning': False,
                'message': 'Error checking usage',
                'usage_percentage': 0
            }
    
    def should_send_warning(self, user_id: str) -> bool:
        """
        Determine if a warning notification should be sent.
        Checks if warning threshold is reached and warning hasn't been sent today.
        
        Args:
            user_id: User ID
            
        Returns:
            True if warning should be sent, False otherwise
        """
        try:
            warning_info = self.check_usage_warning(user_id)
            
            if not warning_info['warning']:
                return False
            
            # Check if warning was already sent today
            # We track this in feature_usage
            today = datetime.utcnow().date()
            usage = self.db.query(Usage).filter_by(
                user_id=user_id,
                date=today
            ).first()
            
            if not usage:
                return False
            
            # Check if warning was already sent
            if isinstance(usage.feature_usage, dict):
                warning_sent = usage.feature_usage.get('warning_sent', False)
                if warning_sent:
                    return False
            
            # Mark warning as sent
            if not isinstance(usage.feature_usage, dict):
                usage.feature_usage = {}
            
            # Create a new dict to trigger SQLAlchemy change detection
            new_feature_usage = dict(usage.feature_usage)
            new_feature_usage['warning_sent'] = True
            usage.feature_usage = new_feature_usage
            usage.updated_at = datetime.utcnow()
            
            # Mark the attribute as modified for SQLAlchemy
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(usage, 'feature_usage')
            
            self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if warning should be sent for user {user_id}: {e}")
            return False


# Global instance
_usage_tracker = None


def get_usage_tracker(db: Optional[Session] = None) -> UsageTracker:
    """
    Get or create UsageTracker instance.
    
    Args:
        db: Optional database session
        
    Returns:
        UsageTracker instance
    """
    if db is not None:
        return UsageTracker(db)
    
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker
