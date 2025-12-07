"""
Tier Configuration Module for VidyaTid Subscription System.

This module defines the subscription tiers, their features, limits, and pricing.
It provides helper functions to access tier configurations and validate tier names.
"""

from typing import Dict, Any, Optional, List


# Subscription Tier Configuration
TIER_CONFIG = {
    'free': {
        'name': 'Free',
        'price_monthly': 0,
        'price_yearly': 0,
        'queries_per_day': 10,
        'features': {
            'basic_queries': True,
            'ncert_content': True,
            'diagrams': False,
            'previous_papers': False,
            'image_solving': False,
            'mock_tests': False,
            'progress_tracking': False,
            'advanced_analytics': False,
            'quiz_questions': 2
        }
    },
    'starter': {
        'name': 'Starter',
        'price_monthly': 9900,  # ₹99 in paise
        'price_yearly': 99000,  # ₹990 (17% discount)
        'queries_per_day': 50,
        'features': {
            'basic_queries': True,
            'ncert_content': True,
            'diagrams': True,
            'previous_papers': True,
            'previous_papers_years': range(2015, 2025),
            'image_solving': False,
            'mock_tests': False,
            'progress_tracking': True,
            'advanced_analytics': False,
            'quiz_questions': 5
        }
    },
    'premium': {
        'name': 'Premium',
        'price_monthly': 29900,  # ₹299 in paise
        'price_yearly': 299000,  # ₹2990 (17% discount)
        'queries_per_day': 200,
        'features': {
            'basic_queries': True,
            'ncert_content': True,
            'diagrams': True,
            'previous_papers': True,
            'previous_papers_years': range(2010, 2025),
            'image_solving': True,
            'mock_tests': True,
            'progress_tracking': True,
            'advanced_analytics': True,
            'quiz_questions': 7
        }
    },
    'ultimate': {
        'name': 'Ultimate',
        'price_monthly': 49900,  # ₹499 in paise
        'price_yearly': 499000,  # ₹4990 (17% discount)
        'queries_per_day': -1,  # Unlimited
        'features': {
            'basic_queries': True,
            'ncert_content': True,
            'diagrams': True,
            'previous_papers': True,
            'previous_papers_years': range(2005, 2025),
            'image_solving': True,
            'mock_tests': True,
            'progress_tracking': True,
            'advanced_analytics': True,
            'personalized_study_plans': True,
            'priority_support': True,
            'quiz_questions': 10
        }
    }
}


# Prediction Features Configuration by Tier
PREDICTION_FEATURES = {
    'free': {
        'chapter_analysis': False,
        'prediction_insights': False,
        'smart_paper_generation': False,
        'complete_paper_prediction': False,
        'predictions_per_month': 0
    },
    'starter': {
        'chapter_analysis': True,
        'prediction_insights': True,
        'smart_paper_generation': False,
        'complete_paper_prediction': False,
        'predictions_per_month': 2
    },
    'premium': {
        'chapter_analysis': True,
        'prediction_insights': True,
        'smart_paper_generation': True,
        'complete_paper_prediction': True,
        'predictions_per_month': 10
    },
    'ultimate': {
        'chapter_analysis': True,
        'prediction_insights': True,
        'smart_paper_generation': True,
        'complete_paper_prediction': True,
        'predictions_per_month': -1  # Unlimited
    }
}


# Valid tier names
VALID_TIERS = ['free', 'starter', 'premium', 'ultimate']


def validate_tier(tier: str) -> bool:
    """
    Validate if a tier name is valid.
    
    Args:
        tier: The tier name to validate
        
    Returns:
        True if the tier is valid, False otherwise
    """
    return tier in VALID_TIERS


def get_tier_config(tier: str) -> Optional[Dict[str, Any]]:
    """
    Get the complete configuration for a specific tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        
    Returns:
        Dictionary containing tier configuration, or None if tier is invalid
    """
    if not validate_tier(tier):
        return None
    return TIER_CONFIG.get(tier)


def get_tier_features(tier: str) -> Optional[Dict[str, Any]]:
    """
    Get the features available for a specific tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        
    Returns:
        Dictionary containing tier features, or None if tier is invalid
    """
    config = get_tier_config(tier)
    if config is None:
        return None
    return config.get('features', {})


def get_tier_limits(tier: str) -> Optional[Dict[str, int]]:
    """
    Get the usage limits for a specific tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        
    Returns:
        Dictionary containing tier limits (queries_per_day, etc.), or None if tier is invalid
    """
    config = get_tier_config(tier)
    if config is None:
        return None
    
    return {
        'queries_per_day': config.get('queries_per_day', 0),
        'price_monthly': config.get('price_monthly', 0),
        'price_yearly': config.get('price_yearly', 0)
    }


def get_tier_price(tier: str, duration: str = 'monthly') -> Optional[int]:
    """
    Get the price for a specific tier and duration.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        duration: The billing duration ('monthly' or 'yearly')
        
    Returns:
        Price in paise, or None if tier is invalid
    """
    config = get_tier_config(tier)
    if config is None:
        return None
    
    if duration == 'yearly':
        return config.get('price_yearly', 0)
    return config.get('price_monthly', 0)


def get_queries_per_day(tier: str) -> int:
    """
    Get the daily query limit for a specific tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        
    Returns:
        Number of queries per day (-1 for unlimited), or 0 if tier is invalid
    """
    config = get_tier_config(tier)
    if config is None:
        return 0
    return config.get('queries_per_day', 0)


def get_prediction_features(tier: str) -> Optional[Dict[str, Any]]:
    """
    Get the prediction features available for a specific tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        
    Returns:
        Dictionary containing prediction features, or None if tier is invalid
    """
    if not validate_tier(tier):
        return None
    return PREDICTION_FEATURES.get(tier)


def get_predictions_per_month(tier: str) -> int:
    """
    Get the monthly prediction limit for a specific tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        
    Returns:
        Number of predictions per month (-1 for unlimited), or 0 if tier is invalid
    """
    pred_features = get_prediction_features(tier)
    if pred_features is None:
        return 0
    return pred_features.get('predictions_per_month', 0)


def has_feature(tier: str, feature: str) -> bool:
    """
    Check if a specific feature is available in a tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        feature: The feature name to check
        
    Returns:
        True if the feature is available, False otherwise
    """
    features = get_tier_features(tier)
    if features is None:
        return False
    return features.get(feature, False)


def has_prediction_feature(tier: str, feature: str) -> bool:
    """
    Check if a specific prediction feature is available in a tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        feature: The prediction feature name to check
        
    Returns:
        True if the prediction feature is available, False otherwise
    """
    pred_features = get_prediction_features(tier)
    if pred_features is None:
        return False
    return pred_features.get(feature, False)


def get_all_tiers() -> List[str]:
    """
    Get a list of all valid tier names.
    
    Returns:
        List of tier names
    """
    return VALID_TIERS.copy()


def get_tier_display_name(tier: str) -> str:
    """
    Get the display name for a tier.
    
    Args:
        tier: The tier name (free, starter, premium, ultimate)
        
    Returns:
        Display name of the tier, or empty string if tier is invalid
    """
    config = get_tier_config(tier)
    if config is None:
        return ''
    return config.get('name', '')


def compare_tiers(tier1: str, tier2: str) -> int:
    """
    Compare two tiers to determine which is higher.
    
    Args:
        tier1: First tier name
        tier2: Second tier name
        
    Returns:
        -1 if tier1 < tier2, 0 if equal, 1 if tier1 > tier2
        Returns 0 if either tier is invalid
    """
    if not validate_tier(tier1) or not validate_tier(tier2):
        return 0
    
    tier_order = {'free': 0, 'starter': 1, 'premium': 2, 'ultimate': 3}
    order1 = tier_order.get(tier1, -1)
    order2 = tier_order.get(tier2, -1)
    
    if order1 < order2:
        return -1
    elif order1 > order2:
        return 1
    return 0


def is_upgrade(from_tier: str, to_tier: str) -> bool:
    """
    Check if moving from one tier to another is an upgrade.
    
    Args:
        from_tier: Current tier name
        to_tier: Target tier name
        
    Returns:
        True if it's an upgrade, False otherwise
    """
    return compare_tiers(from_tier, to_tier) == -1


def is_downgrade(from_tier: str, to_tier: str) -> bool:
    """
    Check if moving from one tier to another is a downgrade.
    
    Args:
        from_tier: Current tier name
        to_tier: Target tier name
        
    Returns:
        True if it's a downgrade, False otherwise
    """
    return compare_tiers(from_tier, to_tier) == 1
