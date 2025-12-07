"""
Usage model for tracking daily query and prediction usage.
"""
from sqlalchemy import Column, String, DateTime, Integer, Date, JSON, Index, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, date
import uuid

from models.database import Base


class Usage(Base):
    """
    Usage tracking model for daily query and prediction limits.
    
    Tracks daily usage counters for queries and predictions per user,
    with automatic reset capabilities and feature-specific usage tracking.
    """
    
    __tablename__ = 'usage'
    
    usage_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    subscription_id = Column(String(36), ForeignKey('subscriptions.subscription_id'), nullable=True)
    date = Column(Date, nullable=False, index=True)
    query_count = Column(Integer, default=0, nullable=False)
    queries_limit = Column(Integer, nullable=False)
    prediction_count = Column(Integer, default=0, nullable=False)
    predictions_limit = Column(Integer, nullable=False)
    feature_usage = Column(JSON, default=dict, nullable=False)  # Track specific feature usage
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    
    # Composite index for efficient lookups by user and date
    __table_args__ = (
        Index('idx_user_date', 'user_id', 'date'),
    )
    
    # Relationships
    user = relationship("User", back_populates="usage_records")
    subscription = relationship("Subscription", back_populates="usage_records")
    
    def __init__(self, user_id, queries_limit, predictions_limit=0, subscription_id=None, date=None):
        """
        Initialize a new Usage record.
        
        Args:
            user_id: User ID this usage record belongs to
            queries_limit: Daily query limit for this user
            predictions_limit: Monthly prediction limit for this user
            subscription_id: Optional subscription ID
            date: Date for this usage record (defaults to today)
        """
        self.usage_id = str(uuid.uuid4())
        self.user_id = user_id
        self.subscription_id = subscription_id
        self.date = date or datetime.utcnow().date()
        self.query_count = 0
        self.queries_limit = queries_limit
        self.prediction_count = 0
        self.predictions_limit = predictions_limit
        self.feature_usage = {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def increment_query_count(self) -> bool:
        """
        Increment the query counter.
        
        Returns:
            True if increment was successful, False if limit reached
        """
        if self.query_count >= self.queries_limit and self.queries_limit != -1:
            return False
        self.query_count += 1
        self.updated_at = datetime.utcnow()
        return True
    
    def increment_prediction_count(self) -> bool:
        """
        Increment the prediction counter.
        
        Returns:
            True if increment was successful, False if limit reached
        """
        if self.prediction_count >= self.predictions_limit and self.predictions_limit != -1:
            return False
        self.prediction_count += 1
        self.updated_at = datetime.utcnow()
        return True
    
    def has_queries_remaining(self) -> bool:
        """
        Check if user has queries remaining.
        
        Returns:
            True if queries are available, False otherwise
        """
        if self.queries_limit == -1:  # Unlimited
            return True
        return self.query_count < self.queries_limit
    
    def has_predictions_remaining(self) -> bool:
        """
        Check if user has predictions remaining.
        
        Returns:
            True if predictions are available, False otherwise
        """
        if self.predictions_limit == -1:  # Unlimited
            return True
        return self.prediction_count < self.predictions_limit
    
    def get_remaining_queries(self) -> int:
        """
        Get number of queries remaining.
        
        Returns:
            Number of queries remaining, or -1 for unlimited
        """
        if self.queries_limit == -1:
            return -1
        return max(0, self.queries_limit - self.query_count)
    
    def get_remaining_predictions(self) -> int:
        """
        Get number of predictions remaining.
        
        Returns:
            Number of predictions remaining, or -1 for unlimited
        """
        if self.predictions_limit == -1:
            return -1
        return max(0, self.predictions_limit - self.prediction_count)
    
    def reset_counters(self, new_queries_limit=None, new_predictions_limit=None):
        """
        Reset usage counters to zero.
        
        Args:
            new_queries_limit: Optional new query limit
            new_predictions_limit: Optional new prediction limit
        """
        self.query_count = 0
        self.prediction_count = 0
        if new_queries_limit is not None:
            self.queries_limit = new_queries_limit
        if new_predictions_limit is not None:
            self.predictions_limit = new_predictions_limit
        self.updated_at = datetime.utcnow()
    
    def track_feature_usage(self, feature_name: str):
        """
        Track usage of a specific feature.
        
        Args:
            feature_name: Name of the feature being used
        """
        if not isinstance(self.feature_usage, dict):
            self.feature_usage = {}
        
        if feature_name in self.feature_usage:
            self.feature_usage[feature_name] += 1
        else:
            self.feature_usage[feature_name] = 1
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of usage record
        """
        return {
            'usage_id': self.usage_id,
            'user_id': self.user_id,
            'subscription_id': self.subscription_id,
            'date': self.date.isoformat() if self.date else None,
            'query_count': self.query_count,
            'queries_limit': self.queries_limit,
            'queries_remaining': self.get_remaining_queries(),
            'prediction_count': self.prediction_count,
            'predictions_limit': self.predictions_limit,
            'predictions_remaining': self.get_remaining_predictions(),
            'feature_usage': self.feature_usage,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f"<Usage(usage_id='{self.usage_id}', user_id='{self.user_id}', date='{self.date}', queries={self.query_count}/{self.queries_limit})>"
