"""
FeatureGateService for VidyaTid.

Controls access to features based on subscription tier.
Enforces tier-based restrictions and provides upgrade prompts.
"""
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from models.database import SessionLocal
from models.user import User
from models.subscription import Subscription
from services.subscription_service import SubscriptionService, get_subscription_service
from services.usage_tracker import UsageTracker, get_usage_tracker
from services.tier_config import (
    get_tier_features,
    get_tier_config,
    get_prediction_features,
    get_tier_price,
    get_tier_display_name,
    has_feature,
    has_prediction_feature,
    validate_tier,
    get_all_tiers,
    compare_tiers
)

logger = logging.getLogger(__name__)


class AccessResult:
    """Result of a feature access check"""
    
    def __init__(
        self,
        allowed: bool,
        message: str,
        current_tier: Optional[str] = None,
        required_tiers: Optional[List[str]] = None,
        upgrade_prompt: Optional[Dict[str, Any]] = None
    ):
        self.allowed = allowed
        self.message = message
        self.current_tier = current_tier
        self.required_tiers = required_tiers or []
        self.upgrade_prompt = upgrade_prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'allowed': self.allowed,
            'message': self.message,
            'current_tier': self.current_tier,
            'required_tiers': self.required_tiers,
            'upgrade_prompt': self.upgrade_prompt
        }


class UpgradePrompt:
    """Structured upgrade prompt for UI rendering"""
    
    def __init__(
        self,
        feature_name: str,
        feature_description: str,
        current_tier: str,
        available_tiers: List[Dict[str, Any]],
        message: str
    ):
        self.feature_name = feature_name
        self.feature_description = feature_description
        self.current_tier = current_tier
        self.available_tiers = available_tiers
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'feature_name': self.feature_name,
            'feature_description': self.feature_description,
            'current_tier': self.current_tier,
            'available_tiers': self.available_tiers,
            'message': self.message
        }


class FeatureGateService:
    """Service for controlling feature access based on subscription tier"""
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize FeatureGateService.
        
        Args:
            db: Optional database session. If not provided, creates a new session.
        """
        self.db = db
        self._owns_session = db is None
        if self._owns_session:
            self.db = SessionLocal()
        
        # Initialize dependent services
        self.subscription_service = get_subscription_service(self.db)
        self.usage_tracker = get_usage_tracker(self.db)
    
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
    
    def _get_user_tier(self, user_id: str) -> str:
        """
        Get user's current subscription tier.
        
        Args:
            user_id: User ID
            
        Returns:
            Tier name (defaults to 'free' if no subscription)
        """
        try:
            subscription_info = self.subscription_service.get_user_subscription(user_id)
            if subscription_info and subscription_info.is_active:
                return subscription_info.tier
            return 'free'
        except Exception as e:
            logger.error(f"Error getting user tier for {user_id}: {e}")
            return 'free'
    
    def can_access_feature(self, user_id: str, feature: str) -> AccessResult:
        """
        Check if user can access a specific feature.
        
        Args:
            user_id: User ID
            feature: Feature name to check
            
        Returns:
            AccessResult object
        """
        try:
            # Get user's tier
            current_tier = self._get_user_tier(user_id)
            
            # Check if feature is available in current tier
            if has_feature(current_tier, feature):
                return AccessResult(
                    allowed=True,
                    message=f"Feature '{feature}' is available in your {current_tier} plan",
                    current_tier=current_tier
                )
            
            # Feature not available, find which tiers have it
            required_tiers = []
            for tier in get_all_tiers():
                if has_feature(tier, feature) and compare_tiers(current_tier, tier) == -1:
                    required_tiers.append(tier)
            
            if not required_tiers:
                return AccessResult(
                    allowed=False,
                    message=f"Feature '{feature}' is not available in any tier",
                    current_tier=current_tier
                )
            
            # Generate upgrade prompt
            upgrade_prompt = self.get_upgrade_prompt(user_id, feature)
            
            return AccessResult(
                allowed=False,
                message=f"Feature '{feature}' requires upgrade to {', '.join(required_tiers)}",
                current_tier=current_tier,
                required_tiers=required_tiers,
                upgrade_prompt=upgrade_prompt.to_dict() if upgrade_prompt else None
            )
            
        except Exception as e:
            logger.error(f"Error checking feature access for user {user_id}, feature {feature}: {e}")
            # Fail closed for security
            return AccessResult(
                allowed=False,
                message="Error checking feature access",
                current_tier='free'
            )
    
    def get_available_features(self, user_id: str) -> List[str]:
        """
        Get list of features accessible to the user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of feature names
        """
        try:
            current_tier = self._get_user_tier(user_id)
            features = get_tier_features(current_tier)
            
            if not features:
                return []
            
            # Return list of features that are enabled (True)
            return [feature_name for feature_name, enabled in features.items() if enabled]
            
        except Exception as e:
            logger.error(f"Error getting available features for user {user_id}: {e}")
            return []
    
    def check_query_limit(self, user_id: str) -> bool:
        """
        Check if user can submit a query (has not reached daily limit).
        
        Args:
            user_id: User ID
            
        Returns:
            True if query is allowed, False otherwise
        """
        try:
            limit_result = self.usage_tracker.check_limit(user_id)
            return limit_result.allowed
        except Exception as e:
            logger.error(f"Error checking query limit for user {user_id}: {e}")
            # Fail open for better user experience
            return True
    
    def check_diagram_access(self, user_id: str) -> bool:
        """
        Check if user can access diagram features.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.can_access_feature(user_id, 'diagrams')
        return result.allowed
    
    def check_image_solving_access(self, user_id: str) -> bool:
        """
        Check if user can access image-based doubt solving.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.can_access_feature(user_id, 'image_solving')
        return result.allowed
    
    def check_mock_test_access(self, user_id: str) -> bool:
        """
        Check if user can access mock tests.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.can_access_feature(user_id, 'mock_tests')
        return result.allowed
    
    def check_previous_papers_access(self, user_id: str, year: int) -> bool:
        """
        Check if user can access previous year papers for a specific year.
        
        Args:
            user_id: User ID
            year: Year of the paper
            
        Returns:
            True if access is allowed, False otherwise
        """
        try:
            current_tier = self._get_user_tier(user_id)
            
            # Check if previous_papers feature is available
            if not has_feature(current_tier, 'previous_papers'):
                return False
            
            # Get the year range for this tier
            features = get_tier_features(current_tier)
            if not features:
                return False
            
            year_range = features.get('previous_papers_years')
            if year_range is None:
                # If no year range specified, allow all years
                return True
            
            # Check if year is in the allowed range
            return year in year_range
            
        except Exception as e:
            logger.error(f"Error checking previous papers access for user {user_id}, year {year}: {e}")
            return False
    
    def check_progress_tracking_access(self, user_id: str) -> bool:
        """
        Check if user can access progress tracking features.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.can_access_feature(user_id, 'progress_tracking')
        return result.allowed
    
    def check_advanced_analytics_access(self, user_id: str) -> bool:
        """
        Check if user can access advanced analytics.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.can_access_feature(user_id, 'advanced_analytics')
        return result.allowed

    def check_prediction_access(self, user_id: str, prediction_type: str) -> AccessResult:
        """
        Check if user can access a specific prediction feature.
        
        Args:
            user_id: User ID
            prediction_type: Type of prediction (chapter_analysis, prediction_insights, 
                           smart_paper_generation, complete_paper_prediction)
            
        Returns:
            AccessResult object
        """
        try:
            # Get user's tier
            current_tier = self._get_user_tier(user_id)
            
            # Check if prediction feature is available in current tier
            if has_prediction_feature(current_tier, prediction_type):
                return AccessResult(
                    allowed=True,
                    message=f"Prediction feature '{prediction_type}' is available in your {current_tier} plan",
                    current_tier=current_tier
                )
            
            # Feature not available, find which tiers have it
            required_tiers = []
            for tier in get_all_tiers():
                if has_prediction_feature(tier, prediction_type) and compare_tiers(current_tier, tier) == -1:
                    required_tiers.append(tier)
            
            if not required_tiers:
                return AccessResult(
                    allowed=False,
                    message=f"Prediction feature '{prediction_type}' is not available in any tier",
                    current_tier=current_tier
                )
            
            # Generate upgrade prompt
            upgrade_prompt = self.get_upgrade_prompt(user_id, prediction_type)
            
            return AccessResult(
                allowed=False,
                message=f"Prediction feature '{prediction_type}' requires upgrade to {', '.join(required_tiers)}",
                current_tier=current_tier,
                required_tiers=required_tiers,
                upgrade_prompt=upgrade_prompt.to_dict() if upgrade_prompt else None
            )
            
        except Exception as e:
            logger.error(f"Error checking prediction access for user {user_id}, type {prediction_type}: {e}")
            # Fail closed for security
            return AccessResult(
                allowed=False,
                message="Error checking prediction access",
                current_tier='free'
            )
    
    def check_prediction_limit(self, user_id: str) -> Tuple[bool, int]:
        """
        Check if user has predictions remaining for the month.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (allowed: bool, predictions_remaining: int)
        """
        try:
            # Get user's tier
            current_tier = self._get_user_tier(user_id)
            
            # Get prediction features for tier
            pred_features = get_prediction_features(current_tier)
            if not pred_features:
                return (False, 0)
            
            predictions_per_month = pred_features.get('predictions_per_month', 0)
            
            # Unlimited predictions
            if predictions_per_month == -1:
                return (True, -1)
            
            # No predictions allowed
            if predictions_per_month == 0:
                return (False, 0)
            
            # Check current usage
            usage_info = self.usage_tracker.get_usage(user_id)
            if not usage_info:
                # No usage record, allow prediction
                return (True, predictions_per_month)
            
            predictions_remaining = usage_info.predictions_remaining
            
            if predictions_remaining > 0 or predictions_remaining == -1:
                return (True, predictions_remaining)
            else:
                return (False, 0)
            
        except Exception as e:
            logger.error(f"Error checking prediction limit for user {user_id}: {e}")
            # Fail closed for security
            return (False, 0)
    
    def check_chapter_analysis_access(self, user_id: str) -> bool:
        """
        Check if user can access chapter analysis feature.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.check_prediction_access(user_id, 'chapter_analysis')
        return result.allowed
    
    def check_prediction_insights_access(self, user_id: str) -> bool:
        """
        Check if user can access prediction insights.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.check_prediction_access(user_id, 'prediction_insights')
        return result.allowed
    
    def check_smart_paper_generation_access(self, user_id: str) -> bool:
        """
        Check if user can access smart paper generation.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.check_prediction_access(user_id, 'smart_paper_generation')
        return result.allowed
    
    def check_complete_paper_prediction_access(self, user_id: str) -> bool:
        """
        Check if user can access complete paper prediction.
        
        Args:
            user_id: User ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        result = self.check_prediction_access(user_id, 'complete_paper_prediction')
        return result.allowed

    def get_upgrade_prompt(self, user_id: str, feature: str) -> Optional[UpgradePrompt]:
        """
        Generate contextual upgrade prompt for a feature.
        
        Args:
            user_id: User ID
            feature: Feature name
            
        Returns:
            UpgradePrompt object or None if feature is already accessible
        """
        try:
            # Get user's current tier
            current_tier = self._get_user_tier(user_id)
            
            # Feature descriptions
            feature_descriptions = {
                'diagrams': 'Access to NCERT diagrams and visual learning aids',
                'previous_papers': 'Access to previous year JEE/NEET question papers',
                'image_solving': 'Upload images of questions for AI-powered solutions',
                'mock_tests': 'Generate and take JEE/NEET pattern mock tests',
                'progress_tracking': 'Track your learning progress and performance',
                'advanced_analytics': 'Detailed analytics and performance insights',
                'personalized_study_plans': 'AI-generated personalized study plans',
                'priority_support': 'Priority customer support',
                'chapter_analysis': 'Chapter-wise analysis and frequency data',
                'prediction_insights': 'AI-powered prediction insights',
                'smart_paper_generation': 'Generate personalized practice papers',
                'complete_paper_prediction': 'Complete predicted NEET/JEE papers'
            }
            
            feature_description = feature_descriptions.get(
                feature,
                f'Access to {feature} feature'
            )
            
            # Check if feature is already accessible
            # Try regular features first
            if has_feature(current_tier, feature):
                return None
            
            # Try prediction features
            if has_prediction_feature(current_tier, feature):
                return None
            
            # Find tiers that have this feature
            available_tiers = []
            
            # Check regular features
            for tier in get_all_tiers():
                if compare_tiers(current_tier, tier) == -1:  # Only higher tiers
                    if has_feature(tier, feature) or has_prediction_feature(tier, feature):
                        tier_config = get_tier_config(tier)
                        if tier_config:
                            tier_info = {
                                'tier': tier,
                                'name': tier_config['name'],
                                'price_monthly': tier_config['price_monthly'],
                                'price_yearly': tier_config['price_yearly'],
                                'queries_per_day': tier_config['queries_per_day']
                            }
                            
                            # Add prediction info if it's a prediction feature
                            pred_features = get_prediction_features(tier)
                            if pred_features:
                                tier_info['predictions_per_month'] = pred_features.get('predictions_per_month', 0)
                            
                            available_tiers.append(tier_info)
            
            if not available_tiers:
                return None
            
            # Generate message
            tier_names = [t['name'] for t in available_tiers]
            if len(tier_names) == 1:
                message = f"Upgrade to {tier_names[0]} to unlock this feature"
            else:
                message = f"Upgrade to {' or '.join(tier_names)} to unlock this feature"
            
            return UpgradePrompt(
                feature_name=feature,
                feature_description=feature_description,
                current_tier=current_tier,
                available_tiers=available_tiers,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error generating upgrade prompt for user {user_id}, feature {feature}: {e}")
            return None


# Global instance
_feature_gate_service = None


def get_feature_gate_service(db: Optional[Session] = None) -> FeatureGateService:
    """
    Get or create FeatureGateService instance.
    
    Args:
        db: Optional database session
        
    Returns:
        FeatureGateService instance
    """
    if db is not None:
        return FeatureGateService(db)
    
    global _feature_gate_service
    if _feature_gate_service is None:
        _feature_gate_service = FeatureGateService()
    return _feature_gate_service
