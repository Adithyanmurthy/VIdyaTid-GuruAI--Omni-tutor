"""
User model for GuruAI application.
Handles user authentication and account management.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Integer
from sqlalchemy.orm import relationship
import bcrypt
from models.database import Base


class User(Base):
    """
    User model for storing user account information.
    
    Attributes:
        user_id: Unique identifier (UUID)
        username: Unique username for login
        password_hash: Bcrypt hashed password
        created_at: Account creation timestamp
        last_login: Last login timestamp
        preferences: JSON field for user preferences
        failed_login_attempts: Counter for failed login attempts
        locked_until: Timestamp until which account is locked
    """
    __tablename__ = 'users'
    
    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=True, index=True)
    full_name = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    preferences = Column(JSON, default=dict, nullable=False)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    
    # Relationships
    progress_records = relationship("Progress", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    usage_records = relationship("Usage", back_populates="user", cascade="all, delete-orphan")
    subscription = relationship("Subscription", back_populates="user", uselist=False)
    payments = relationship("Payment", back_populates="user")
    
    def __init__(self, username, password, email=None, preferences=None):
        """
        Initialize a new User.
        
        Args:
            username: Username for the account
            password: Plain text password (will be hashed)
            email: Optional email address
            preferences: Optional dictionary of user preferences
        """
        self.user_id = str(uuid.uuid4())
        self.username = username
        self.email = email
        self.password_hash = self._hash_password(password)
        self.created_at = datetime.utcnow()
        self.preferences = preferences or {}
        self.failed_login_attempts = 0
    
    @staticmethod
    def _hash_password(password):
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password as string
        """
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password):
        """
        Verify a password against the stored hash.
        
        Args:
            password: Plain text password to verify
            
        Returns:
            True if password matches, False otherwise
        """
        return bcrypt.checkpw(
            password.encode('utf-8'),
            self.password_hash.encode('utf-8')
        )
    
    def update_password(self, new_password):
        """
        Update the user's password.
        
        Args:
            new_password: New plain text password
        """
        self.password_hash = self._hash_password(new_password)
    
    def update_last_login(self):
        """Update the last login timestamp to current time."""
        self.last_login = datetime.utcnow()
    
    def increment_failed_login(self):
        """Increment the failed login attempts counter."""
        self.failed_login_attempts += 1
    
    def reset_failed_login(self):
        """Reset the failed login attempts counter."""
        self.failed_login_attempts = 0
    
    def lock_account(self, duration_seconds=900):
        """
        Lock the account for a specified duration.
        
        Args:
            duration_seconds: Duration to lock account (default 15 minutes)
        """
        from datetime import timedelta
        self.locked_until = datetime.utcnow() + timedelta(seconds=duration_seconds)
    
    def is_locked(self):
        """
        Check if the account is currently locked.
        
        Returns:
            True if account is locked, False otherwise
        """
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def unlock_account(self):
        """Unlock the account."""
        self.locked_until = None
        self.failed_login_attempts = 0
    
    def to_dict(self):
        """
        Convert user to dictionary (excluding sensitive data).
        
        Returns:
            Dictionary representation of user
        """
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'phone': self.phone,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'preferences': self.preferences
        }
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', username='{self.username}')>"
