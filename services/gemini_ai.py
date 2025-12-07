"""
Google Gemini AI Service
Free tier: 15 requests per minute, 1500 requests per day
Get API key from: https://makersuite.google.com/app/apikey

Supports multiple API keys with automatic rotation when rate limited.
"""
import os
import logging
import time
import google.generativeai as genai
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class GeminiAI:
    """Google Gemini AI service for chat, problem solving, and content generation"""
    
    def __init__(self):
        # Load multiple API keys (comma-separated) or fall back to single key
        api_keys_str = os.getenv("GEMINI_API_KEYS", "")
        self.api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        
        # Fall back to single key if multiple keys not set
        if not self.api_keys:
            single_key = os.getenv("GEMINI_API_KEY", "")
            if single_key:
                self.api_keys = [single_key]
        
        if not self.api_keys:
            raise ValueError(
                "No Gemini API keys found. Set GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY in .env\n"
                "Get free key from: https://makersuite.google.com/app/apikey"
            )
        
        self.current_key_index = 0
        self.key_cooldowns = {}  # Track rate-limited keys
        self.cooldown_duration = 60  # seconds to wait before retrying a rate-limited key
        
        # Use gemini-2.0-flash as it's the latest available model
        # Note: gemini-1.5-flash may not be available in all regions/API versions
        self.model_name = 'gemini-2.0-flash'
        
        try:
            self._configure_current_key()
            self.enabled = True
            
            logger.info(f"✅ Gemini AI initialized with {len(self.api_keys)} API key(s) ({self.model_name})")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def _configure_current_key(self):
        """Configure genai with the current API key"""
        api_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        logger.debug(f"Using API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _rotate_key(self) -> bool:
        """
        Rotate to the next available API key.
        Returns True if a new key was found, False if all keys are rate-limited.
        """
        if len(self.api_keys) <= 1:
            return False
        
        current_time = time.time()
        original_index = self.current_key_index
        
        # Mark current key as rate-limited
        self.key_cooldowns[self.current_key_index] = current_time
        
        # Try to find an available key
        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            # Check if this key is still in cooldown
            cooldown_time = self.key_cooldowns.get(self.current_key_index, 0)
            if current_time - cooldown_time >= self.cooldown_duration:
                self._configure_current_key()
                logger.info(f"🔄 Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
                return True
        
        # All keys are rate-limited, stay on original
        self.current_key_index = original_index
        return False
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, 
                 stop: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate text completion using Gemini
        Compatible with ModelManager interface
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences (optional)
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            Dictionary with 'text', 'success', 'tokens_used', 'error'
        """
        max_retries = len(self.api_keys)
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Configure generation
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    stop_sequences=stop if stop else None
                )
                
                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract text - handle different response formats with robust error handling
                text = ""
                
                # Try to get text directly first
                try:
                    if hasattr(response, 'text'):
                        text = response.text
                except (IndexError, AttributeError, ValueError) as e:
                    logger.debug(f"Could not access response.text directly: {e}")
                
                # If that didn't work, try candidates
                if not text:
                    try:
                        if hasattr(response, 'candidates') and response.candidates:
                            if len(response.candidates) > 0:
                                candidate = response.candidates[0]
                                if hasattr(candidate, 'content'):
                                    content = candidate.content
                                    if hasattr(content, 'parts') and content.parts:
                                        if len(content.parts) > 0:
                                            part = content.parts[0]
                                            if hasattr(part, 'text'):
                                                text = part.text
                    except (IndexError, AttributeError, ValueError) as e:
                        logger.debug(f"Could not access response via candidates: {e}")
                
                # Final check
                if not text:
                    logger.warning("Empty or inaccessible response from Gemini")
                    return {
                        'text': '',
                        'success': False,
                        'tokens_used': 0,
                        'error': 'Empty response from API'
                    }
                
                # Estimate tokens (rough approximation)
                tokens_used = len(text.split()) + len(prompt.split())
                
                result = {
                    'text': text,
                    'success': True,
                    'tokens_used': tokens_used,
                    'error': None
                }
                
                logger.info(f"✅ Gemini generation successful (~{tokens_used} tokens)")
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                
                # Check if this is a rate limit error
                is_rate_limit = any(term in error_str for term in [
                    'rate limit', 'quota', '429', 'resource exhausted', 
                    'too many requests', 'rate_limit'
                ])
                
                if is_rate_limit and self._rotate_key():
                    logger.warning(f"⚠️ Rate limit hit, rotating to next API key (attempt {attempt + 1}/{max_retries})")
                    continue
                
                # Not a rate limit error or no more keys available
                break
        
        logger.error(f"❌ Gemini generation failed: {last_error}")
        
        # Try fallback to Cloudflare AI
        try:
            from services.cloudflare_ai import get_cloudflare_ai, is_cloudflare_ai_enabled
            if is_cloudflare_ai_enabled():
                logger.info("🔄 Falling back to Cloudflare AI...")
                cf_ai = get_cloudflare_ai()
                result = cf_ai.generate(prompt, temperature, max_tokens, stop=stop)
                if result.get('success'):
                    logger.info("✅ Cloudflare AI fallback successful")
                    return result
        except Exception as fallback_error:
            logger.error(f"Cloudflare fallback also failed: {fallback_error}")
        
        return {
            'text': '',
            'success': False,
            'tokens_used': 0,
            'error': str(last_error)
        }
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, 
             max_tokens: int = 1024) -> str:
        """
        Generate chat response using Gemini
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Convert messages to prompt format
            prompt = self._messages_to_prompt(messages)
            
            # Generate
            result = self.generate(prompt, temperature, max_tokens)
            
            if result['success']:
                return result['text']
            else:
                raise Exception(result['error'])
                
        except Exception as e:
            logger.error(f"Gemini chat failed: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt format"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System Instructions: {content}\n")
            elif role == 'user':
                prompt_parts.append(f"User: {content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant: ")
        return "\n".join(prompt_parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status info (compatible with ModelManager)"""
        return {
            'enabled': self.enabled,
            'model': self.model_name,
            'provider': 'Google Gemini',
            'tier': 'Free',
            'rate_limit': '15 requests/minute, 1500 requests/day',
            'quality': 'Excellent',
            'api_keys_count': len(self.api_keys),
            'current_key_index': self.current_key_index + 1,
            'key_rotation': 'enabled' if len(self.api_keys) > 1 else 'disabled'
        }


def get_gemini_ai():
    """Get or create Gemini AI instance"""
    global _gemini_ai
    if '_gemini_ai' not in globals():
        _gemini_ai = GeminiAI()
    return _gemini_ai


def is_gemini_enabled():
    """Check if Gemini is enabled"""
    return os.getenv('USE_GEMINI', 'false').lower() == 'true'
