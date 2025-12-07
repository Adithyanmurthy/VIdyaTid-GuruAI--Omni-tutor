"""
Cloudflare Workers AI integration for VidyaTid.
Provides chat, embeddings, and image recognition using Cloudflare AI models.
Includes fallback to local models for reliability.
"""
import requests
import logging
from typing import List, Dict, Any, Optional
import time

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from config import Config

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
TIMEOUT = 30  # seconds


class CloudflareAI:
    """
    Cloudflare Workers AI client with fallback support.
    
    Features:
    - Automatic retry with exponential backoff
    - Fallback to local models on failure
    - Request timeout handling
    - Error logging and monitoring
    """
    
    def __init__(self, enable_fallback=True):
        """
        Initialize Cloudflare AI client.
        
        Args:
            enable_fallback: Whether to fallback to local models on failure
        """
        self.account_id = Config.CLOUDFLARE_ACCOUNT_ID
        self.api_token = Config.CLOUDFLARE_API_TOKEN
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"
        self.enable_fallback = enable_fallback
        self.local_model = None
        
        if not self.account_id or not self.api_token:
            logger.warning("Cloudflare AI credentials not configured. Using local models.")
            self.enabled = False
            if enable_fallback:
                self._init_local_fallback()
        else:
            self.enabled = True
            logger.info("Cloudflare AI initialized successfully")
    
    def _init_local_fallback(self):
        """Initialize local model fallback"""
        try:
            from services.model_manager import ModelManager
            self.local_model = ModelManager()
            logger.info("Local model fallback initialized")
        except Exception as e:
            logger.error(f"Failed to initialize local model fallback: {e}")
            self.local_model = None
    
    def _make_request(self, model: str, data: Dict[str, Any], retries=MAX_RETRIES) -> Dict[str, Any]:
        """
        Make request to Cloudflare AI API with retry logic.
        
        Args:
            model: Model identifier
            data: Request payload
            retries: Number of retries remaining
        
        Returns:
            API response as dictionary
        
        Raises:
            Exception: If all retries fail
        """
        if not self.enabled:
            raise Exception("Cloudflare AI not configured")
        
        url = f"{self.base_url}/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        last_error = None
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=data, 
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Cloudflare AI request timeout (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                    
            except requests.exceptions.HTTPError as e:
                last_error = e
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    logger.error(f"Cloudflare AI client error: {e}")
                    raise
                logger.warning(f"Cloudflare AI server error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"Cloudflare AI request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
        
        # All retries failed
        logger.error(f"Cloudflare AI request failed after {retries} attempts: {last_error}")
        raise Exception(f"Cloudflare AI request failed: {last_error}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        use_fallback: bool = True
    ) -> str:
        """
        Generate chat response using Llama 3.1 8B with fallback support.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            use_fallback: Whether to use local model fallback on failure
            
        Returns:
            Generated response text
        """
        try:
            data = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            result = self._make_request(Config.CLOUDFLARE_CHAT_MODEL, data)
            
            # Extract response text
            if 'result' in result and 'response' in result['result']:
                return result['result']['response']
            elif 'result' in result and 'choices' in result['result']:
                return result['result']['choices'][0]['message']['content']
            else:
                logger.error(f"Unexpected response format: {result}")
                raise Exception("Could not parse response")
                
        except Exception as e:
            logger.error(f"Cloudflare chat generation failed: {e}")
            
            # Try fallback to local model
            if use_fallback and self.enable_fallback and self.local_model:
                logger.info("Falling back to local model for chat generation")
                try:
                    return self._chat_fallback(messages, temperature, max_tokens)
                except Exception as fallback_error:
                    logger.error(f"Local model fallback also failed: {fallback_error}")
            
            raise Exception(f"Chat generation failed: {e}")
    
    def _chat_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Fallback to local model for chat generation.
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            Generated response text
        """
        if not self.local_model:
            raise Exception("Local model not available")
        
        # Convert messages to prompt format for local model
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        
        # Generate using local model
        response = self.local_model.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        use_fallback: bool = True,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text completion using Cloudflare AI (compatible with ModelManager interface).
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            use_fallback: Whether to use local model fallback on failure
            stop: Stop sequences (ignored for Cloudflare AI, kept for compatibility)
            **kwargs: Additional arguments (ignored, kept for compatibility)
            
        Returns:
            Dictionary with 'text', 'success', and 'tokens_used' keys
        """
        try:
            # Convert prompt to chat format
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response_text = self.chat(messages, temperature, max_tokens, use_fallback)
            
            return {
                'text': response_text,
                'tokens_used': len(response_text.split()),  # Approximate
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Generate failed: {e}")
            return {
                'text': '',
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information (compatible with ModelManager interface).
        
        Returns:
            Dictionary with status information
        """
        return {
            'enabled': self.enabled,
            'model': 'cloudflare-workers-ai',
            'chat_model': Config.CLOUDFLARE_CHAT_MODEL,
            'embedding_model': Config.CLOUDFLARE_EMBEDDING_MODEL,
            'fallback_enabled': self.enable_fallback,
            'fallback_available': self.local_model is not None
        }
    
    def generate_embeddings(self, text: str, use_fallback: bool = True) -> List[float]:
        """
        Generate embeddings using BGE-Base-en-v1.5 with fallback support.
        
        Args:
            text: Text to embed
            use_fallback: Whether to use local model fallback on failure
            
        Returns:
            List of embedding values (768 dimensions for Cloudflare, 384 for local)
        """
        try:
            data = {"text": text}
            result = self._make_request(Config.CLOUDFLARE_EMBEDDING_MODEL, data)
            
            # Extract embeddings
            if 'result' in result and 'data' in result['result']:
                return result['result']['data'][0]
            else:
                logger.error(f"Unexpected embedding format: {result}")
                raise Exception("Could not parse embeddings")
                
        except Exception as e:
            logger.error(f"Cloudflare embedding generation failed: {e}")
            
            # Try fallback to local model
            if use_fallback and self.enable_fallback:
                logger.info("Falling back to local embeddings")
                try:
                    return self._embeddings_fallback(text)
                except Exception as fallback_error:
                    logger.error(f"Local embeddings fallback also failed: {fallback_error}")
            
            raise Exception(f"Embedding generation failed: {e}")
    
    def _embeddings_fallback(self, text: str) -> List[float]:
        """
        Fallback to local embeddings model.
        
        Args:
            text: Text to embed
        
        Returns:
            List of embedding values
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load local embedding model (cached after first load)
            if not hasattr(self, '_local_embedding_model'):
                logger.info("Loading local embedding model...")
                self._local_embedding_model = SentenceTransformer(
                    Config.EMBEDDING_MODEL_NAME
                )
            
            # Generate embeddings
            embeddings = self._local_embedding_model.encode(text)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise
    
    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze image using ResNet-50
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dict with classification results
        """
        try:
            # Convert bytes to list of integers for API
            image_array = list(image_bytes)
            
            data = {"image": image_array}
            result = self._make_request(Config.CLOUDFLARE_IMAGE_MODEL, data)
            
            return result.get('result', {})
                
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    def chat_with_context(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate answer with RAG (Retrieval Augmented Generation)
        
        Args:
            question: User's question
            context: Retrieved context from NCERT
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = (
                "You are VidyaTid, an AI tutor for JEE & NEET preparation. "
                "Answer questions based on NCERT content. "
                "Provide clear, accurate, and educational responses."
            )
        
        user_prompt = f"""Context from NCERT:
{context}

Question: {question}

Provide a clear, accurate answer based on the NCERT content above. 
Include relevant examples and explanations."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.chat(messages)


# Global instance
_cloudflare_ai = None


def get_cloudflare_ai() -> CloudflareAI:
    """Get or create CloudflareAI instance"""
    global _cloudflare_ai
    if _cloudflare_ai is None:
        _cloudflare_ai = CloudflareAI()
    return _cloudflare_ai


def is_cloudflare_ai_enabled() -> bool:
    """Check if Cloudflare AI is enabled and configured"""
    return Config.USE_CLOUDFLARE_AI and get_cloudflare_ai().enabled


async def health_check() -> Dict[str, Any]:
    """
    Perform health check on Cloudflare AI service.
    
    Returns:
        Dictionary with health status and details
    """
    ai = get_cloudflare_ai()
    
    health_status = {
        'cloudflare_ai': {
            'enabled': ai.enabled,
            'configured': bool(ai.account_id and ai.api_token),
            'status': 'unknown'
        },
        'local_fallback': {
            'available': ai.local_model is not None,
            'status': 'unknown'
        }
    }
    
    # Test Cloudflare AI
    if ai.enabled:
        try:
            test_messages = [
                {"role": "user", "content": "Hello"}
            ]
            response = ai.chat(test_messages, max_tokens=10, use_fallback=False)
            health_status['cloudflare_ai']['status'] = 'healthy'
            health_status['cloudflare_ai']['test_response'] = response[:50]
        except Exception as e:
            health_status['cloudflare_ai']['status'] = 'unhealthy'
            health_status['cloudflare_ai']['error'] = str(e)
    
    # Test local fallback
    if ai.local_model:
        try:
            test_messages = [
                {"role": "user", "content": "Hello"}
            ]
            response = ai._chat_fallback(test_messages, 0.7, 10)
            health_status['local_fallback']['status'] = 'healthy'
        except Exception as e:
            health_status['local_fallback']['status'] = 'unhealthy'
            health_status['local_fallback']['error'] = str(e)
    
    return health_status


def get_service_info() -> Dict[str, Any]:
    """
    Get information about the Cloudflare AI service configuration.
    
    Returns:
        Dictionary with service information
    """
    ai = get_cloudflare_ai()
    
    return {
        'service': 'Cloudflare Workers AI',
        'enabled': ai.enabled,
        'fallback_enabled': ai.enable_fallback,
        'models': {
            'chat': Config.CLOUDFLARE_CHAT_MODEL,
            'embeddings': Config.CLOUDFLARE_EMBEDDING_MODEL,
            'image': Config.CLOUDFLARE_IMAGE_MODEL
        },
        'configuration': {
            'account_configured': bool(ai.account_id),
            'token_configured': bool(ai.api_token),
            'base_url': ai.base_url if ai.enabled else None
        },
        'retry_config': {
            'max_retries': MAX_RETRIES,
            'retry_delay': RETRY_DELAY,
            'timeout': TIMEOUT
        }
    }
