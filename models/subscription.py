"""
Subscription model for user subscriptions.
"""
from sqlalchemy import Column, String, DateTime, Boolean, Integer, CheckConstraint, ForeignKey
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import uuid

from models.database import Base


class Subscription(Base):
    """User subscription model"""
    
    __tablename__ = 'subscriptions'
    
    subscription_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    tier = Column(String(20), nullable=False)  # free, starter, premium, ultimate
    status = Column(String(20), nullable=False, default='active')  # active, cancelled, expired, pending
    razorpay_subscription_id = Column(String(100), nullable=True)
    start_date = Column(DateTime, nullable=False, default=func.now())
    end_date = Column(DateTime, nullable=False)
    auto_renew = Column(Boolean, default=True)
    cancelled_at = Column(DateTime, nullable=True)
    scheduled_tier_change = Column(String(20), nullable=True)  # For scheduled downgrades
    scheduled_change_date = Column(DateTime, nullable=True)  # When the tier change should occur
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    
    # Add constraint to validate tier values
    __table_args__ = (
        CheckConstraint(
            "tier IN ('free', 'starter', 'premium', 'ultimate')",
            name='valid_tier_check'
        ),
    )
    
    # Relationships
    user = relationship("User", back_populates="subscription")
    usage_records = relationship("Usage", back_populates="subscription")
    
    @validates('tier')
    def validate_tier(self, key, tier):
        """Validate that tier is one of the allowed values."""
        valid_tiers = ['free', 'starter', 'premium', 'ultimate']
        if tier not in valid_tiers:
            raise ValueError(f"Invalid tier '{tier}'. Must be one of: {', '.join(valid_tiers)}")
        return tier
    
    @validates('scheduled_tier_change')
    def validate_scheduled_tier_change(self, key, scheduled_tier):
        """Validate that scheduled tier change is one of the allowed values."""
        if scheduled_tier is None:
            return scheduled_tier
        valid_tiers = ['free', 'starter', 'premium', 'ultimate']
        if scheduled_tier not in valid_tiers:
            raise ValueError(f"Invalid scheduled tier '{scheduled_tier}'. Must be one of: {', '.join(valid_tiers)}")
        return scheduled_tier
    
    def is_active(self) -> bool:
        """Check if subscription is currently active"""
        return (
            self.status == 'active' and
            self.end_date > datetime.utcnow()
        )
    
    def is_expired(self) -> bool:
        """Check if subscription has expired"""
        return self.end_date <= datetime.utcnow()
    
    def days_remaining(self) -> int:
        """Get number of days remaining"""
        if self.is_expired():
            return 0
        delta = self.end_date - datetime.utcnow()
        return delta.days
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'subscription_id': self.subscription_id,
            'user_id': self.user_id,
            'tier': self.tier,
            'status': self.status,
            'razorpay_subscription_id': self.razorpay_subscription_id,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'auto_renew': self.auto_renew,
            'is_active': self.is_active(),
            'days_remaining': self.days_remaining(),
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'scheduled_tier_change': self.scheduled_tier_change,
            'scheduled_change_date': self.scheduled_change_date.isoformat() if self.scheduled_change_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
