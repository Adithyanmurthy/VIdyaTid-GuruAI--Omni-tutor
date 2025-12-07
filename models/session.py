"""
Session model for GuruAI application.
Handles user session management and authentication tokens.
"""
import uuid
import secrets
from datetime import datetime, timedelta
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from models.database import Base


class Session(Base):
    """
    Session model for managing user authentication sessions.
    
    Attributes:
        session_id: Unique session identifier (UUID)
        user_id: Foreign key to User
        session_token: Secure random token for authentication
        created_at: Session creation timestamp
        expires_at: Session expiration timestamp
        is_active: Whether the session is currently active
        last_activity: Last activity timestamp
    """
    __tablename__ = 'sessions'
    
    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    session_token = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="sessions")
    
    def __init__(self, user_id, duration_hours=24):
        """
        Initialize a new Session.
        
        Args:
            user_id: User ID this session belongs to
            duration_hours: Session duration in hours (default 24)
        """
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id
        self.session_token = self._generate_token()
        self.created_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
        self.is_active = True
        self.last_activity = datetime.utcnow()
    
    @staticmethod
    def _generate_token():
        """
        Generate a cryptographically secure random token.
        
        Returns:
            64-character hexadecimal token
        """
        return secrets.token_hex(32)
    
    def is_valid(self):
        """
        Check if the session is valid (active and not expired).
        
        Returns:
            True if session is valid, False otherwise
        """
        if not self.is_active:
            return False
        
        if datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def is_expired(self):
        """
        Check if the session has expired.
        
        Returns:
            True if session is expired, False otherwise
        """
        return datetime.utcnow() > self.expires_at
    
    def update_activity(self):
        """Update the last activity timestamp to current time."""
        self.last_activity = datetime.utcnow()
    
    def extend_session(self, hours=24):
        """
        Extend the session expiration time.
        
        Args:
            hours: Number of hours to extend the session
        """
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.update_activity()
    
    def invalidate(self):
        """Invalidate the session (logout)."""
        self.is_active = False
    
    def refresh_token(self):
        """
        Generate a new session token.
        Useful for security purposes after sensitive operations.
        """
        self.session_token = self._generate_token()
        self.update_activity()
    
    def to_dict(self):
        """
        Convert session to dictionary.
        
        Returns:
            Dictionary representation of session
        """
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'session_token': self.session_token,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'is_valid': self.is_valid(),
            'is_expired': self.is_expired()
        }
    
    @staticmethod
    def cleanup_expired_sessions(db_session):
        """
        Clean up expired sessions from the database.
        
        Args:
            db_session: SQLAlchemy database session
            
        Returns:
            Number of sessions deleted
        """
        expired_sessions = db_session.query(Session).filter(
            Session.expires_at < datetime.utcnow()
        ).all()
        
        count = len(expired_sessions)
        
        for session in expired_sessions:
            db_session.delete(session)
        
        db_session.commit()
        
        return count
    
    @staticmethod
    def cleanup_inactive_sessions(db_session, inactive_hours=24):
        """
        Clean up sessions that have been inactive for a specified period.
        
        Args:
            db_session: SQLAlchemy database session
            inactive_hours: Hours of inactivity before cleanup
            
        Returns:
            Number of sessions deleted
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=inactive_hours)
        
        inactive_sessions = db_session.query(Session).filter(
            Session.last_activity < cutoff_time
        ).all()
        
        count = len(inactive_sessions)
        
        for session in inactive_sessions:
            db_session.delete(session)
        
        db_session.commit()
        
        return count
    
    def __repr__(self):
        return (f"<Session(session_id='{self.session_id}', user_id='{self.user_id}', "
                f"is_active={self.is_active}, expires_at='{self.expires_at}')>")
