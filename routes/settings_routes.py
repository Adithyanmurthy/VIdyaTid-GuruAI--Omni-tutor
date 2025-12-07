"""
Settings API routes for GuruAI.
Handles system configuration, user preferences, and performance settings.
"""
from flask import Blueprint, request, jsonify, render_template
from functools import wraps
import logging
import json
import os
import psutil
from pathlib import Path
from datetime import datetime

from models.database import SessionLocal
from models.user import User
from routes.auth_routes import require_auth
from config import Config

logger = logging.getLogger(__name__)

# Create blueprint
settings_bp = Blueprint('settings', __name__, url_prefix='/api')

# Default settings
DEFAULT_SETTINGS = {
    # General
    'target_exam': 'JEE-Main',
    'default_language': 'en',
    'auto_save': True,
    'show_hints': True,
    
    # Performance
    'memory_limit': 6,  # GB
    'idle_timeout': 10,  # minutes
    'response_speed': 'balanced',
    'cache_responses': True,
    
    # Appearance
    'theme': 'dark',
    'font_size': 'medium',
    'compact_mode': False,
    'show_animations': True,
    
    # Voice
    'enable_voice_input': True,
    'voice_language': 'hinglish',
    'enable_tts': False,
    'speech_speed': 1.0,
    'voice_type': 'female',
    
    # Privacy
    'peer_comparison': False,
    'usage_analytics': False
}


def validate_settings(settings):
    """
    Validate settings values.
    
    Args:
        settings: Dictionary of settings to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Validate memory limit
    if 'memory_limit' in settings:
        memory_limit = settings['memory_limit']
        if not isinstance(memory_limit, (int, float)) or memory_limit < 2 or memory_limit > 16:
            return False, 'Memory limit must be between 2 and 16 GB'
    
    # Validate idle timeout
    if 'idle_timeout' in settings:
        idle_timeout = settings['idle_timeout']
        if not isinstance(idle_timeout, (int, float)) or idle_timeout < 1 or idle_timeout > 60:
            return False, 'Idle timeout must be between 1 and 60 minutes'
    
    # Validate response speed
    if 'response_speed' in settings:
        if settings['response_speed'] not in ['fast', 'balanced', 'quality']:
            return False, 'Response speed must be fast, balanced, or quality'
    
    # Validate theme
    if 'theme' in settings:
        if settings['theme'] not in ['light', 'dark', 'auto']:
            return False, 'Theme must be light, dark, or auto'
    
    # Validate font size
    if 'font_size' in settings:
        if settings['font_size'] not in ['small', 'medium', 'large', 'extra-large']:
            return False, 'Font size must be small, medium, large, or extra-large'
    
    # Validate speech speed
    if 'speech_speed' in settings:
        speech_speed = settings['speech_speed']
        if not isinstance(speech_speed, (int, float)) or speech_speed < 0.5 or speech_speed > 2.0:
            return False, 'Speech speed must be between 0.5 and 2.0'
    
    # Validate voice language
    if 'voice_language' in settings:
        if settings['voice_language'] not in ['en', 'hi', 'hinglish']:
            return False, 'Voice language must be en, hi, or hinglish'
    
    # Validate voice type
    if 'voice_type' in settings:
        if settings['voice_type'] not in ['male', 'female']:
            return False, 'Voice type must be male or female'
    
    # Validate target exam
    if 'target_exam' in settings:
        if settings['target_exam'] not in ['JEE-Main', 'JEE-Advanced', 'NEET', 'Both']:
            return False, 'Target exam must be JEE-Main, JEE-Advanced, NEET, or Both'
    
    # Validate default language
    if 'default_language' in settings:
        if settings['default_language'] not in ['en', 'hi', 'hinglish']:
            return False, 'Default language must be en, hi, or hinglish'
    
    return True, None


@settings_bp.route('/settings', methods=['GET'])
@require_auth
def get_settings(user_id, session):
    """
    Get user settings.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "settings": {
                "target_exam": "string",
                "default_language": "string",
                ...
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
            
            # Get user preferences (settings are stored in preferences)
            user_settings = user.preferences.get('settings', {})
            
            # Merge with defaults
            settings = {**DEFAULT_SETTINGS, **user_settings}
            
            return jsonify({
                'settings': settings
            }), 200
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch settings',
                'details': str(e)
            }
        }), 500


@settings_bp.route('/settings', methods=['POST'])
@require_auth
def save_settings(user_id, session):
    """
    Save user settings.
    
    Requires authentication token in header or cookie.
    
    Request Body:
        {
            "settings": {
                "target_exam": "string",
                "default_language": "string",
                ...
            }
        }
    
    Returns:
        {
            "status": "success",
            "settings": {},
            "message": "string"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'settings' not in data:
            return jsonify({
                'error': {
                    'code': 'MISSING_FIELDS',
                    'message': 'Settings field is required'
                }
            }), 400
        
        new_settings = data['settings']
        
        if not isinstance(new_settings, dict):
            return jsonify({
                'error': {
                    'code': 'INVALID_DATA',
                    'message': 'Settings must be a JSON object'
                }
            }), 400
        
        # Validate settings
        is_valid, error_message = validate_settings(new_settings)
        if not is_valid:
            return jsonify({
                'error': {
                    'code': 'INVALID_SETTINGS',
                    'message': error_message
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
            
            # Update user preferences with new settings
            if user.preferences is None:
                user.preferences = {}
            
            user.preferences['settings'] = new_settings
            db.commit()
            
            logger.info(f"Settings saved for user: {user_id}")
            
            return jsonify({
                'status': 'success',
                'settings': new_settings,
                'message': 'Settings saved successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving settings: {e}")
            return jsonify({
                'error': {
                    'code': 'SAVE_FAILED',
                    'message': 'Failed to save settings',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in save-settings endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500


@settings_bp.route('/settings/reset', methods=['POST'])
@require_auth
def reset_settings(user_id, session):
    """
    Reset user settings to defaults.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "status": "success",
            "settings": {},
            "message": "string"
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
            
            # Reset to default settings
            if user.preferences is None:
                user.preferences = {}
            
            user.preferences['settings'] = DEFAULT_SETTINGS.copy()
            db.commit()
            
            logger.info(f"Settings reset to defaults for user: {user_id}")
            
            return jsonify({
                'status': 'success',
                'settings': DEFAULT_SETTINGS,
                'message': 'Settings reset to defaults'
            }), 200
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error resetting settings: {e}")
            return jsonify({
                'error': {
                    'code': 'RESET_FAILED',
                    'message': 'Failed to reset settings',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in reset-settings endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500


@settings_bp.route('/system/memory', methods=['GET'])
@require_auth
def get_memory_usage(user_id, session):
    """
    Get current system memory usage.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "total": float,  # Total RAM in GB
            "used": float,   # Used RAM in GB
            "available": float,  # Available RAM in GB
            "percent": float  # Usage percentage
        }
    """
    try:
        memory = psutil.virtual_memory()
        
        return jsonify({
            'total': memory.total / (1024 ** 3),  # Convert to GB
            'used': memory.used / (1024 ** 3),
            'available': memory.available / (1024 ** 3),
            'percent': memory.percent
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch memory usage',
                'details': str(e)
            }
        }), 500


@settings_bp.route('/system/info', methods=['GET'])
@require_auth
def get_system_info(user_id, session):
    """
    Get system information.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "data_location": "string",
            "diagram_count": int,
            "ncert_content_size": float,  # Size in MB
            "database_size": float,  # Size in MB
            "cache_size": float  # Size in MB
        }
    """
    try:
        # Get data location
        data_location = str(Config.BASE_DIR)
        
        # Count diagrams
        diagram_count = 0
        if Config.DIAGRAMS_DIR.exists():
            diagram_count = sum(1 for _ in Config.DIAGRAMS_DIR.rglob('*.png')) + \
                           sum(1 for _ in Config.DIAGRAMS_DIR.rglob('*.jpg'))
        
        # Get NCERT content size
        ncert_size = 0
        if Config.NCERT_CONTENT_DIR.exists():
            ncert_size = sum(f.stat().st_size for f in Config.NCERT_CONTENT_DIR.rglob('*') if f.is_file())
            ncert_size = ncert_size / (1024 ** 2)  # Convert to MB
        
        # Get database size
        db_size = 0
        db_path = Path(Config.SQLALCHEMY_DATABASE_URI.replace('sqlite:///', ''))
        if db_path.exists():
            db_size = db_path.stat().st_size / (1024 ** 2)  # Convert to MB
        
        # Get cache size (ChromaDB)
        cache_size = 0
        if Config.CHROMA_DB_PATH.exists():
            cache_size = sum(f.stat().st_size for f in Config.CHROMA_DB_PATH.rglob('*') if f.is_file())
            cache_size = cache_size / (1024 ** 2)  # Convert to MB
        
        return jsonify({
            'data_location': data_location,
            'diagram_count': diagram_count,
            'ncert_content_size': round(ncert_size, 2),
            'database_size': round(db_size, 2),
            'cache_size': round(cache_size, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({
            'error': {
                'code': 'FETCH_FAILED',
                'message': 'Failed to fetch system info',
                'details': str(e)
            }
        }), 500


@settings_bp.route('/data/export', methods=['GET'])
@require_auth
def export_data(user_id, session):
    """
    Export user data (progress, settings, history).
    
    Requires authentication token in header or cookie.
    
    Returns:
        ZIP file containing user data
    """
    try:
        import zipfile
        import io
        from flask import send_file
        
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
            
            # Create in-memory ZIP file
            memory_file = io.BytesIO()
            
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Export user info
                user_data = {
                    'username': user.username,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'preferences': user.preferences
                }
                zf.writestr('user_info.json', json.dumps(user_data, indent=2))
                
                # Export progress data
                from models.progress import Progress
                progress_records = db.query(Progress).filter_by(user_id=user_id).all()
                progress_data = [p.to_dict() for p in progress_records]
                zf.writestr('progress.json', json.dumps(progress_data, indent=2))
                
                # Export README
                readme = f"""GuruAI Data Export
                
User: {user.username}
Export Date: {datetime.now().isoformat()}

This archive contains:
- user_info.json: Your account information and settings
- progress.json: Your learning progress and statistics

To import this data, use the import feature in GuruAI settings.
"""
                zf.writestr('README.txt', readme)
            
            memory_file.seek(0)
            
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'guruai_data_{user.username}_{datetime.now().strftime("%Y%m%d")}.zip'
            )
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({
            'error': {
                'code': 'EXPORT_FAILED',
                'message': 'Failed to export data',
                'details': str(e)
            }
        }), 500


@settings_bp.route('/cache/clear', methods=['POST'])
@require_auth
def clear_cache(user_id, session):
    """
    Clear application cache.
    
    Requires authentication token in header or cookie.
    
    Returns:
        {
            "status": "success",
            "freed_space": float,  # Space freed in MB
            "message": "string"
        }
    """
    try:
        freed_space = 0
        
        # Clear ChromaDB cache (if applicable)
        # Note: We don't actually delete the vector store, just temporary cache files
        cache_dir = Config.BASE_DIR / 'cache'
        if cache_dir.exists():
            for file in cache_dir.rglob('*'):
                if file.is_file():
                    size = file.stat().st_size
                    file.unlink()
                    freed_space += size
        
        freed_space_mb = freed_space / (1024 ** 2)
        
        logger.info(f"Cache cleared for user: {user_id}, freed {freed_space_mb:.2f} MB")
        
        return jsonify({
            'status': 'success',
            'freed_space': round(freed_space_mb, 2),
            'message': f'Cache cleared successfully. Freed {freed_space_mb:.2f} MB'
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'error': {
                'code': 'CLEAR_FAILED',
                'message': 'Failed to clear cache',
                'details': str(e)
            }
        }), 500


@settings_bp.route('/data/delete-all', methods=['POST'])
@require_auth
def delete_all_data(user_id, session):
    """
    Delete all user data permanently.
    
    Requires authentication token in header or cookie.
    This action cannot be undone.
    
    Returns:
        {
            "status": "success",
            "message": "string"
        }
    """
    try:
        db = SessionLocal()
        try:
            # Delete progress records
            from models.progress import Progress
            db.query(Progress).filter_by(user_id=user_id).delete()
            
            # Delete sessions
            from models.session import Session
            db.query(Session).filter_by(user_id=user_id).delete()
            
            # Delete user
            user = db.query(User).filter_by(user_id=user_id).first()
            if user:
                username = user.username
                db.delete(user)
            
            db.commit()
            
            logger.info(f"All data deleted for user: {user_id}")
            
            return jsonify({
                'status': 'success',
                'message': 'All data deleted successfully'
            }), 200
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting all data: {e}")
            return jsonify({
                'error': {
                    'code': 'DELETE_FAILED',
                    'message': 'Failed to delete data',
                    'details': str(e)
                }
            }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in delete-all-data endpoint: {e}")
        return jsonify({
            'error': {
                'code': 'SERVER_ERROR',
                'message': 'Internal server error',
                'details': str(e)
            }
        }), 500


# Settings page route
@settings_bp.route('/settings-page', methods=['GET'])
def settings_page():
    """Render the settings page."""
    return render_template('settings.html')
