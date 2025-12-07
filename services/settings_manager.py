"""
Settings Manager for GuruAI - Manages application settings and configuration.
Provides centralized access to user settings and system configuration.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from models.database import SessionLocal
from models.user import User
from config import Config

logger = logging.getLogger(__name__)


class SettingsManager:
    """
    Manages application settings and configuration.
    
    Features:
    - User-specific settings persistence
    - Settings validation
    - Default settings management
    - Integration with model manager for dynamic configuration
    """
    
    # Default settings
    DEFAULT_SETTINGS = {
        # General
        'target_exam': 'JEE-Main',
        'default_language': 'en',
        'auto_save': True,
        'show_hints': True,
        
        # Performance
        'memory_limit': Config.DEFAULT_MEMORY_LIMIT,  # GB
        'idle_timeout': Config.DEFAULT_IDLE_TIMEOUT,  # minutes
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
    
    def __init__(self):
        """Initialize the Settings Manager."""
        self._cache = {}  # Cache for user settings
        logger.info("SettingsManager initialized")
    
    def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Get settings for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of user settings (merged with defaults)
        """
        # Check cache first
        if user_id in self._cache:
            return self._cache[user_id]
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                logger.warning(f"User not found: {user_id}, using defaults")
                return self.DEFAULT_SETTINGS.copy()
            
            # Get user preferences
            user_settings = user.preferences.get('settings', {}) if user.preferences else {}
            
            # Merge with defaults
            settings = {**self.DEFAULT_SETTINGS, **user_settings}
            
            # Cache the settings
            self._cache[user_id] = settings
            
            return settings
            
        except Exception as e:
            logger.error(f"Error getting user settings: {e}")
            return self.DEFAULT_SETTINGS.copy()
        finally:
            db.close()
    
    def save_user_settings(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """
        Save settings for a specific user.
        
        Args:
            user_id: User ID
            settings: Dictionary of settings to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        # Validate settings
        is_valid, error_message = self.validate_settings(settings)
        if not is_valid:
            logger.error(f"Invalid settings: {error_message}")
            return False
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                logger.error(f"User not found: {user_id}")
                return False
            
            # Update user preferences
            if user.preferences is None:
                user.preferences = {}
            
            user.preferences['settings'] = settings
            db.commit()
            
            # Update cache
            self._cache[user_id] = settings
            
            logger.info(f"Settings saved for user: {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving user settings: {e}")
            return False
        finally:
            db.close()
    
    def reset_user_settings(self, user_id: str) -> bool:
        """
        Reset user settings to defaults.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if reset successfully, False otherwise
        """
        return self.save_user_settings(user_id, self.DEFAULT_SETTINGS.copy())
    
    def get_model_config(self, user_id: str) -> Dict[str, Any]:
        """
        Get model configuration based on user settings.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of model configuration
        """
        settings = self.get_user_settings(user_id)
        
        # Convert settings to model config
        config = {
            'idle_timeout': settings.get('idle_timeout', Config.DEFAULT_IDLE_TIMEOUT) * 60,  # Convert to seconds
            'n_ctx': Config.LLM_N_CTX,
            'n_gpu_layers': Config.LLM_N_GPU_LAYERS,
            'temperature': Config.LLM_TEMPERATURE,
            'max_tokens': Config.LLM_MAX_TOKENS
        }
        
        # Adjust based on response speed setting
        response_speed = settings.get('response_speed', 'balanced')
        if response_speed == 'fast':
            config['max_tokens'] = 256
            config['temperature'] = 0.5
        elif response_speed == 'quality':
            config['max_tokens'] = 1024
            config['temperature'] = 0.8
        
        return config
    
    def validate_settings(self, settings: Dict[str, Any]) -> tuple:
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
            if not isinstance(memory_limit, (int, float)):
                return False, 'Memory limit must be a number'
            if memory_limit < Config.MIN_MEMORY_LIMIT or memory_limit > Config.MAX_MEMORY_LIMIT:
                return False, f'Memory limit must be between {Config.MIN_MEMORY_LIMIT} and {Config.MAX_MEMORY_LIMIT} GB'
        
        # Validate idle timeout
        if 'idle_timeout' in settings:
            idle_timeout = settings['idle_timeout']
            if not isinstance(idle_timeout, (int, float)):
                return False, 'Idle timeout must be a number'
            if idle_timeout < Config.MIN_IDLE_TIMEOUT or idle_timeout > Config.MAX_IDLE_TIMEOUT:
                return False, f'Idle timeout must be between {Config.MIN_IDLE_TIMEOUT} and {Config.MAX_IDLE_TIMEOUT} minutes'
        
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
            if not isinstance(speech_speed, (int, float)):
                return False, 'Speech speed must be a number'
            if speech_speed < 0.5 or speech_speed > 2.0:
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
        
        # Validate boolean fields
        boolean_fields = [
            'auto_save', 'show_hints', 'cache_responses', 'compact_mode',
            'show_animations', 'enable_voice_input', 'enable_tts',
            'peer_comparison', 'usage_analytics'
        ]
        
        for field in boolean_fields:
            if field in settings and not isinstance(settings[field], bool):
                return False, f'{field} must be a boolean value'
        
        return True, None
    
    def clear_cache(self, user_id: Optional[str] = None):
        """
        Clear settings cache.
        
        Args:
            user_id: User ID to clear cache for (None to clear all)
        """
        if user_id:
            self._cache.pop(user_id, None)
            logger.info(f"Settings cache cleared for user: {user_id}")
        else:
            self._cache.clear()
            logger.info("Settings cache cleared for all users")


# Singleton instance
_settings_manager_instance: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """
    Get the singleton SettingsManager instance.
    
    Returns:
        SettingsManager instance
    """
    global _settings_manager_instance
    
    if _settings_manager_instance is None:
        _settings_manager_instance = SettingsManager()
    
    return _settings_manager_instance
