"""
Authentication API routes for GuruAI.
Handles user registration, login, logout, and session management.
"""
from flask import Blueprint, request, jsonify, render_template
from functools import wraps
import logging
from datetime import datetime

from models.database import SessionLocal
from models.user import User
from models.session import Session
from services.email_service import get_email_service

logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)


# Auth page route (no prefix)
@auth_bp.route('/auth')
def auth_page():
    """Render the login/register page"""
    return render_template('auth.html')


def get_session_token():
    """
    Extract session token from request headers or cookies.
    
    Returns:
        Session token string or None
    """
    # Try Authorization header first
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(' ')[1]
    
    # Try session cookie
    return request.cookies.get('session_token')


def require_auth(f):
    """
    Decorator to require authentication for a route.
    Validates session token and adds user_id to kwargs.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_session_token()
        
        if not token:
            return jsonify({
                'error': {
                    'code': 'MISSING_TOKEN',
                    'message': 'Authentication token is required'
                }
            }), 401
        
        db = SessionLocal()
        try:
            # Find session by token
            session = db.query(Session).filter_by(session_token=token).first()
            
            if not session:
                return jsonify({
                    'error': {
                        'code': 'INVALID_TOKEN',
                        'message': 'Invalid authentication token'
                    }
                }), 401
            
            # Check if session is valid
            if not session.is_valid():
                return jsonify({
                    'error': {
                        'code': 'SESSION_EXPIRED',
                        'message': 'Session has expired. Please log in again.'
                    }
                }), 401
            
            # Update last activity
            session.update_activity()
            db.commit()
            
            # Add user_id to kwargs
            kwargs['user_id'] = session.user_id
            kwargs['session'] = session
            
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return jsonify({
                'error': {
                    'code': 'AUTH_ERROR',
                    'message': 'Authentication error',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
    
    return decorated_function


@auth_bp.route('/api/auth/register', methods=['POST'])
def register():
    """
    Register a new user account.
    
    Request Body:
        {
            "username": "string",
            "password": "string",
            "preferences": {} (optional)
        }
    
    Returns:
        {
            "status": "success",
            "user": {
                "user_id": "string",
                "username": "string",
                "created_at": "string"
            },
            "message": "string"
        }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'Username and password are required'
                }
            }), 400
        
        username = data['username'].strip()
        password = data['password']
        preferences = data.get('preferences', {})
        
        # Validate username
        if len(username) < 3:
            return jsonify({
                'error': {
                    'code': 'INVALID_USERNAME',
                    'message': 'Username must be at least 3 characters long'
                }
            }), 400
        
        if len(username) > 50:
            return jsonify({
                'error': {
                    'code': 'INVALID_USERNAME',
                    'message': 'Username must be at most 50 characters long'
                }
            }), 400
        
        # Validate password
        if len(password) < 8:
            return jsonify({
                'error': {
                    'code': 'WEAK_PASSWORD',
                    'message': 'Password must be at least 8 characters long'
                }
            }), 400
        
        db = SessionLocal()
        try:
            # Check if username already exists
            existing_user = db.query(User).filter_by(username=username).first()
            if existing_user:
                return jsonify({
                    'error': {
                        'code': 'USERNAME_EXISTS',
                        'message': 'Username already exists'
                    }
                }), 409
            
            # Create new user
            new_user = User(
                username=username,
                password=password,
                preferences=preferences
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            logger.info(f"New user registered: {username}")
            
            # Send welcome email
            try:
                email_service = get_email_service()
                email_service.send_welcome_email(
                    user_email=username,
                    username=username
                )
            except Exception as e:
                logger.error(f"Failed to send welcome email: {e}")
            
            return jsonify({
                'status': 'success',
                'user': new_user.to_dict(),
                'message': 'User registered successfully'
            }), 201
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error registering user: {e}")
            return jsonify({
                'error': {
                    'code': 'REGISTRATION_FAILED',
                    'message': 'Failed to register user',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in register endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500


@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    """
    Authenticate user and create session.
    
    Request Body:
        {
            "email": "string",
            "password": "string"
        }
    
    Returns:
        {
            "status": "success",
            "user": {
                "user_id": "string",
                "username": "string",
                "email": "string",
                "last_login": "string"
            },
            "session": {
                "session_token": "string",
                "expires_at": "string"
            },
            "message": "string"
        }
    """
    try:
        data = request.get_json()
        
        # Validate required fields - accept both email and username for backward compatibility
        if not data or ('email' not in data and 'username' not in data) or 'password' not in data:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'Email/Username and password are required'
                }
            }), 400
        
        # Use email if provided, otherwise fall back to username
        email_or_username = data.get('email', data.get('username', '')).strip()
        password = data['password']
        
        db = SessionLocal()
        try:
            # Find user by email first, then by username for backward compatibility
            user = db.query(User).filter_by(email=email_or_username).first()
            if not user:
                user = db.query(User).filter_by(username=email_or_username).first()
            
            if not user:
                return jsonify({
                    'error': {
                        'code': 'INVALID_CREDENTIALS',
                        'message': 'Invalid email or password'
                    }
                }), 401
            
            # Check if account is locked
            if user.is_locked():
                locked_until = user.locked_until.isoformat() if user.locked_until else 'unknown'
                return jsonify({
                    'error': {
                        'code': 'ACCOUNT_LOCKED',
                        'message': f'Account is locked due to too many failed login attempts. Try again after {locked_until}',
                        'locked_until': locked_until
                    }
                }), 403
            
            # Verify password
            if not user.verify_password(password):
                # Increment failed login attempts
                user.increment_failed_login()
                
                # Lock account if too many failed attempts
                from config import Config
                if user.failed_login_attempts >= Config.MAX_LOGIN_ATTEMPTS:
                    user.lock_account(duration_seconds=Config.ACCOUNT_LOCKOUT_DURATION)
                    db.commit()
                    
                    logger.warning(f"Account locked for user: {email_or_username}")
                    
                    return jsonify({
                        'error': {
                            'code': 'ACCOUNT_LOCKED',
                            'message': f'Account locked due to {Config.MAX_LOGIN_ATTEMPTS} failed login attempts. Try again in {Config.ACCOUNT_LOCKOUT_DURATION // 60} minutes.',
                            'locked_until': user.locked_until.isoformat() if user.locked_until else None
                        }
                    }), 403
                
                db.commit()
                
                remaining_attempts = Config.MAX_LOGIN_ATTEMPTS - user.failed_login_attempts
                
                return jsonify({
                    'error': {
                        'code': 'INVALID_CREDENTIALS',
                        'message': f'Invalid email or password. {remaining_attempts} attempts remaining.',
                        'remaining_attempts': remaining_attempts
                    }
                }), 401
            
            # Password is correct - reset failed login attempts
            user.reset_failed_login()
            user.update_last_login()
            
            # Create new session
            new_session = Session(user_id=user.user_id)
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            
            logger.info(f"User logged in: {email_or_username}")
            
            response = jsonify({
                'status': 'success',
                'user': user.to_dict(),
                'session': {
                    'session_token': new_session.session_token,
                    'expires_at': new_session.expires_at.isoformat()
                },
                'message': 'Login successful'
            })
            
            # Set session cookie
            response.set_cookie(
                'session_token',
                new_session.session_token,
                httponly=True,
                secure=False,  # Set to True in production with HTTPS
                samesite='Lax',
                max_age=86400  # 24 hours
            )
            
            return response, 200
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error during login: {e}")
            return jsonify({
                'error': {
                    'code': 'LOGIN_FAILED',
                    'message': 'Login failed',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in login endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500


@auth_bp.route('/api/auth/logout', methods=['POST'])
@require_auth
def logout(user_id, session):
    """
    Logout user and invalidate session.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "status": "success",
            "message": "string"
        }
    """
    try:
        db = SessionLocal()
        try:
            # Invalidate the session
            db_session = db.query(Session).filter_by(session_id=session.session_id).first()
            if db_session:
                db_session.invalidate()
                db.commit()
            
            logger.info(f"User logged out: {user_id}")
            
            response = jsonify({
                'status': 'success',
                'message': 'Logout successful'
            })
            
            # Clear session cookie
            response.set_cookie('session_token', '', expires=0)
            
            return response, 200
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error during logout: {e}")
            return jsonify({
                'error': {
                    'code': 'LOGOUT_FAILED',
                    'message': 'Logout failed',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in logout endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500


@auth_bp.route('/api/auth/check', methods=['GET'])
def check_auth():
    """
    Check if user is authenticated (without requiring auth).
    Returns user info if authenticated, or indicates not authenticated.
    
    Returns:
        {
            "authenticated": boolean,
            "user": {...} (if authenticated)
        }
    """
    try:
        token = get_session_token()
        
        if not token:
            return jsonify({
                'authenticated': False
            }), 200
        
        db = SessionLocal()
        try:
            # Find session by token
            session = db.query(Session).filter_by(session_token=token).first()
            
            if not session or not session.is_valid():
                return jsonify({
                    'authenticated': False
                }), 200
            
            # Update last activity
            session.update_activity()
            db.commit()
            
            # Get user info
            user = db.query(User).filter_by(user_id=session.user_id).first()
            
            if not user:
                return jsonify({
                    'authenticated': False
                }), 200
            
            return jsonify({
                'authenticated': True,
                'user': user.to_dict()
            }), 200
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error checking auth: {e}")
        return jsonify({
            'authenticated': False
        }), 200


@auth_bp.route('/validate', methods=['GET'])
@require_auth
def validate_session(user_id, session):
    """
    Validate current session and return user info.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "status": "valid",
            "user": {
                "user_id": "string",
                "username": "string",
                "last_login": "string"
            },
            "session": {
                "session_id": "string",
                "expires_at": "string",
                "last_activity": "string"
            }
        }
    """
    try:
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                return jsonify({
                    'error': {
                        'code': 'USER_NOT_FOUND',
                        'message': 'User not found'
                    }
                }), 404
            
            return jsonify({
                'status': 'valid',
                'user': user.to_dict(),
                'session': session.to_dict()
            }), 200
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error validating session: {e}")
        return jsonify({
            'error': {
                'code': 'VALIDATION_FAILED',
                'message': 'Session validation failed',
                'details': str(e)
            }
        }), 500


@auth_bp.route('/change-password', methods=['POST'])
@require_auth
def change_password(user_id, session):
    """
    Change user password.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "current_password": "string",
            "new_password": "string"
        }
    
    Returns:
        {
            "status": "success",
            "message": "string"
        }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'current_password' not in data or 'new_password' not in data:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'Current password and new password are required'
                }
            }), 400
        
        current_password = data['current_password']
        new_password = data['new_password']
        
        # Validate new password
        if len(new_password) < 8:
            return jsonify({
                'error': {
                    'code': 'WEAK_PASSWORD',
                    'message': 'New password must be at least 8 characters long'
                }
            }), 400
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                return jsonify({
                    'error': {
                        'code': 'USER_NOT_FOUND',
                        'message': 'User not found'
                    }
                }), 404
            
            # Verify current password
            if not user.verify_password(current_password):
                return jsonify({
                    'error': {
                        'code': 'INVALID_PASSWORD',
                        'message': 'Current password is incorrect'
                    }
                }), 401
            
            # Update password
            user.update_password(new_password)
            db.commit()
            
            logger.info(f"Password changed for user: {user_id}")
            
            return jsonify({
                'status': 'success',
                'message': 'Password changed successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error changing password: {e}")
            return jsonify({
                'error': {
                    'code': 'PASSWORD_CHANGE_FAILED',
                    'message': 'Failed to change password',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in change-password endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500


@auth_bp.route('/user/<user_id>', methods=['GET'])
@require_auth
def get_user_info(user_id, session):
    """
    Get user information.
    
    Requires authentication token in header or cookie.
    User can only access their own information.
    
    Returns:
        {
            "user": {
                "user_id": "string",
                "username": "string",
                "created_at": "string",
                "last_login": "string",
                "preferences": {}
            }
        }
    """
    try:
        # Verify user is accessing their own data
        if session.user_id != user_id:
            return jsonify({
                'error': {
                    'code': 'UNAUTHORIZED',
                    'message': 'You can only access your own user information'
                }
            }), 403
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                return jsonify({
                    'error': {
                        'code': 'USER_NOT_FOUND',
                        'message': 'User not found'
                    }
                }), 404
            
            return jsonify({
                'user': user.to_dict()
            }), 200
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch user information',
                'details': str(e)
            }
        }), 500


@auth_bp.route('/user/<user_id>/preferences', methods=['PUT'])
@require_auth
def update_preferences(user_id, session):
    """
    Update user preferences.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "preferences": {}
        }
    
    Returns:
        {
            "status": "success",
            "preferences": {},
            "message": "string"
        }
    """
    try:
        # Verify user is updating their own data
        if session.user_id != user_id:
            return jsonify({
                'error': {
                    'code': 'UNAUTHORIZED',
                    'message': 'You can only update your own preferences'
                }
            }), 403
        
        data = request.get_json()
        
        if not data or 'preferences' not in data:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'Preferences field is required'
                }
            }), 400
        
        preferences = data['preferences']
        
        if not isinstance(preferences, dict):
            return jsonify({
                'error': {
                    'code': 'INVALID_DATA',
                    'message': 'Preferences must be a JSON object'
                }
            }), 400
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                return jsonify({
                    'error': {
                        'code': 'USER_NOT_FOUND',
                        'message': 'User not found'
                    }
                }), 404
            
            # Update preferences
            user.preferences = preferences
            db.commit()
            
            logger.info(f"Preferences updated for user: {user_id}")
            
            return jsonify({
                'status': 'success',
                'preferences': user.preferences,
                'message': 'Preferences updated successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating preferences: {e}")
            return jsonify({
                'error': {
                    'code': 'UPDATE_FAILED',
                    'message': 'Failed to update preferences',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in update-preferences endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500


@auth_bp.route('/sessions', methods=['GET'])
@require_auth
def get_user_sessions(user_id, session):
    """
    Get all active sessions for the current user.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "sessions": [
                {
                    "session_id": "string",
                    "created_at": "string",
                    "expires_at": "string",
                    "last_activity": "string",
                    "is_current": boolean
                }
            ]
        }
    """
    try:
        db = SessionLocal()
        try:
            sessions = db.query(Session).filter_by(
                user_id=user_id,
                is_active=True
            ).all()
            
            sessions_data = []
            for s in sessions:
                session_dict = s.to_dict()
                session_dict['is_current'] = (s.session_id == session.session_id)
                sessions_data.append(session_dict)
            
            return jsonify({
                'sessions': sessions_data
            }), 200
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch sessions',
                'details': str(e)
            }
        }), 500


@auth_bp.route('/sessions/<session_id>', methods=['DELETE'])
@require_auth
def revoke_session(session_id, user_id, session):
    """
    Revoke a specific session.
    
    Requires authentication token in header or cookie.
    User can only revoke their own sessions.
    
    Returns:
        {
            "status": "success",
            "message": "string"
        }
    """
    try:
        db = SessionLocal()
        try:
            target_session = db.query(Session).filter_by(session_id=session_id).first()
            
            if not target_session:
                return jsonify({
                    'error': {
                        'code': 'SESSION_NOT_FOUND',
                        'message': 'Session not found'
                    }
                }), 404
            
            # Verify user owns this session
            if target_session.user_id != user_id:
                return jsonify({
                    'error': {
                        'code': 'UNAUTHORIZED',
                        'message': 'You can only revoke your own sessions'
                    }
                }), 403
            
            # Invalidate the session
            target_session.invalidate()
            db.commit()
            
            logger.info(f"Session revoked: {session_id} for user: {user_id}")
            
            return jsonify({
                'status': 'success',
                'message': 'Session revoked successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error revoking session: {e}")
            return jsonify({
                'error': {
                    'code': 'REVOKE_FAILED',
                    'message': 'Failed to revoke session',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in revoke-session endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500
