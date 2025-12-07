"""
Payment model for payment records.
"""
from sqlalchemy import Column, String, DateTime, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from models.database import Base


class Payment(Base):
    """Payment record model"""
    
    __tablename__ = 'payments'
    
    payment_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    subscription_id = Column(String(36), ForeignKey('subscriptions.subscription_id'), nullable=True)
    razorpay_payment_id = Column(String(100), nullable=False, unique=True)
    razorpay_order_id = Column(String(100), nullable=True)
    amount = Column(Integer, nullable=False)  # Amount in paise
    currency = Column(String(3), nullable=False, default='INR')
    status = Column(String(20), nullable=False)  # pending, captured, failed, refunded
    payment_method = Column(String(50), nullable=True)  # card, upi, netbanking, wallet
    payment_type = Column(String(20), nullable=False, default='subscription')  # subscription, upgrade, one_time
    payment_metadata = Column(JSON, default=dict, nullable=False)  # Additional payment information
    created_at = Column(DateTime, nullable=False, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="payments")
    subscription = relationship("Subscription")
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'payment_id': self.payment_id,
            'user_id': self.user_id,
            'subscription_id': self.subscription_id,
            'razorpay_payment_id': self.razorpay_payment_id,
            'razorpay_order_id': self.razorpay_order_id,
            'amount': self.amount,
            'amount_inr': self.amount / 100,  # Convert paise to rupees
            'currency': self.currency,
            'status': self.status,
            'payment_method': self.payment_method,
            'payment_type': self.payment_type,
            'metadata': self.payment_metadata,  # Return as 'metadata' for API compatibility
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
