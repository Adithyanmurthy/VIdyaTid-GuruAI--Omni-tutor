"""
User profile API routes for VidyaTid.
Handles user profile management and updates.
"""
from flask import Blueprint, request, jsonify
import logging

from routes.auth_routes import require_auth
from models.user import User
from models.database import SessionLocal

logger = logging.getLogger(__name__)

# Create blueprint
user_bp = Blueprint('user', __name__)


@user_bp.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile(user_id, session):
    """
    Get user's profile information.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "user": {
                "user_id": "string",
                "username": "string",
                "email": "string",
                "full_name": "string",
                "phone": "string",
                "created_at": "string",
                "last_login": "string"
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
                'user': user.to_dict()
            }), 200
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting user profile for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'PROFILE_ERROR',
                'message': 'Failed to get user profile',
                'details': str(e)
            }
        }), 500


@user_bp.route('/api/user/update', methods=['PUT'])
@require_auth
def update_user_profile(user_id, session):
    """
    Update user's profile information.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "username": "string" (optional),
            "email": "string" (optional),
            "full_name": "string" (optional),
            "phone": "string" (optional)
        }
    
    Returns:
        {
            "status": "success",
            "user": {user_dict},
            "message": "Profile updated successfully"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': {
                    'code': 'MISSING_DATA',
                    'message': 'No data provided'
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
            
            # Update fields if provided
            if 'username' in data and data['username']:
                # Check if username is already taken by another user
                existing_user = db.query(User).filter(
                    User.username == data['username'],
                    User.user_id != user_id
                ).first()
                
                if existing_user:
                    return jsonify({
                        'error': {
                            'code': 'USERNAME_TAKEN',
                            'message': 'Username is already taken'
                        }
                    }), 400
                
                user.username = data['username'].strip()
            
            if 'email' in data:
                user.email = data['email'].strip() if data['email'] else None
            
            if 'full_name' in data:
                user.full_name = data['full_name'].strip() if data['full_name'] else None
            
            if 'phone' in data:
                user.phone = data['phone'].strip() if data['phone'] else None
            
            db.commit()
            db.refresh(user)
            
            logger.info(f"User profile updated for user {user_id}")
            
            return jsonify({
                'status': 'success',
                'user': user.to_dict(),
                'message': 'Profile updated successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error updating user profile for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'UPDATE_ERROR',
                'message': 'Failed to update profile',
                'details': str(e)
            }
        }), 500


@user_bp.route('/api/user/change-password', methods=['POST'])
@require_auth
def change_password(user_id, session):
    """
    Change user's password.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "current_password": "string",
            "new_password": "string"
        }
    
    Returns:
        {
            "status": "success",
            "message": "Password changed successfully"
        }
    """
    try:
        data = request.get_json()
        
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
                    'code': 'INVALID_PASSWORD',
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
            
            logger.info(f"Password changed for user {user_id}")
            
            return jsonify({
                'status': 'success',
                'message': 'Password changed successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error changing password for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'PASSWORD_CHANGE_ERROR',
                'message': 'Failed to change password',
                'details': str(e)
            }
        }), 500


@user_bp.route('/api/user/preferences', methods=['PUT'])
@require_auth
def update_preferences(user_id, session):
    """
    Update user preferences.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "preference_key": "preference_value",
            ...
        }
    
    Returns:
        {
            "status": "success",
            "preferences": {dict},
            "message": "Preferences updated successfully"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': {
                    'code': 'MISSING_DATA',
                    'message': 'No preferences provided'
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
            if user.preferences is None:
                user.preferences = {}
            
            user.preferences.update(data)
            db.commit()
            db.refresh(user)
            
            logger.info(f"Preferences updated for user {user_id}")
            
            return jsonify({
                'status': 'success',
                'preferences': user.preferences,
                'message': 'Preferences updated successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error updating preferences for user {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'PREFERENCES_ERROR',
                'message': 'Failed to update preferences',
                'details': str(e)
            }
        }), 500


@user_bp.route('/api/user/delete', methods=['DELETE'])
@require_auth
def delete_user_account(user_id, session):
    """
    Delete user account and all associated data.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "status": "success",
            "message": "Account deleted successfully"
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
            
            # Delete user (cascade will delete related records)
            db.delete(user)
            db.commit()
            
            logger.info(f"User account deleted: {user_id}")
            
            return jsonify({
                'status': 'success',
                'message': 'Account deleted successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error deleting user account {user_id}: {e}")
        return jsonify({
            'error': {
                'code': 'DELETE_ERROR',
                'message': 'Failed to delete account',
                'details': str(e)
            }
        }), 500
