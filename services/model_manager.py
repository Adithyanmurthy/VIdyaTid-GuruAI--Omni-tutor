"""
Model Manager for GuruAI - Handles AI model lifecycle and resource management.
Implements lazy loading and automatic unloading based on idle timeout.
"""
import time
import gc
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import os

# Optional llama_cpp import (not available on cloud deployments)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the lifecycle of the local LLM model.
    
    Features:
    - Lazy loading: Model is loaded only when first needed
    - Automatic unloading: Model is unloaded after idle timeout
    - Memory monitoring: Tracks model status and memory usage
    - Thread-safe: Can be used in multi-threaded Flask application
    """
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize the Model Manager.
        
        Args:
            model_path: Path to the GGUF model file
            config: Configuration dictionary with settings:
                - idle_timeout: Seconds before unloading idle model (default: 600)
                - n_ctx: Context window size (default: 2048)
                - n_gpu_layers: GPU layers to offload (default: 0)
                - temperature: Default temperature (default: 0.7)
                - max_tokens: Default max tokens (default: 512)
        """
        self.model_path = Path(model_path)
        self.config = config
        
        # Model state
        self.model: Optional[Llama] = None
        self.last_used: Optional[float] = None
        self.is_loading: bool = False
        self.load_lock = threading.Lock()
        
        # Configuration
        self.idle_timeout = config.get('idle_timeout', 600)  # 10 minutes
        self.n_ctx = config.get('n_ctx', 2048)
        self.n_gpu_layers = config.get('n_gpu_layers', 0)
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 512)
        
        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"ModelManager initialized with model: {self.model_path}")
        logger.info(f"Idle timeout: {self.idle_timeout}s, Context: {self.n_ctx}")
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self.model is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current model status.
        
        Returns:
            Dictionary with status information:
                - loaded: Whether model is loaded
                - loading: Whether model is currently loading
                - last_used: Timestamp of last use (None if never used)
                - idle_time: Seconds since last use (None if not loaded)
                - model_path: Path to model file
        """
        status = {
            'loaded': self.is_loaded(),
            'loading': self.is_loading,
            'last_used': self.last_used,
            'idle_time': None,
            'model_path': str(self.model_path)
        }
        
        if self.is_loaded() and self.last_used is not None:
            status['idle_time'] = time.time() - self.last_used
        
        return status
    
    def load_model(self) -> bool:
        """
        Load the model into memory.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        with self.load_lock:
            # If already loaded, just update timestamp
            if self.is_loaded():
                self.last_used = time.time()
                return True
            
            # If currently loading, wait
            if self.is_loading:
                logger.info("Model is already being loaded, waiting...")
                return False
            
            try:
                self.is_loading = True
                logger.info(f"Loading model from: {self.model_path}")
                start_time = time.time()
                
                # Load the model
                self.model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )
                
                load_time = time.time() - start_time
                self.last_used = time.time()
                self.is_loading = False
                
                logger.info(f"Model loaded successfully in {load_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
                self.is_loading = False
                return False
    
    def unload_model(self) -> bool:
        """
        Unload the model from memory to free resources.
        
        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        with self.load_lock:
            if not self.is_loaded():
                logger.info("Model is not loaded, nothing to unload")
                return True
            
            try:
                logger.info("Unloading model from memory...")
                del self.model
                self.model = None
                self.last_used = None
                
                # Force garbage collection
                gc.collect()
                
                logger.info("Model unloaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload model: {e}")
                return False
    
    def check_idle_and_unload(self) -> bool:
        """
        Check if model has been idle and unload if necessary.
        
        Returns:
            bool: True if model was unloaded, False otherwise
        """
        if not self.is_loaded():
            return False
        
        if self.last_used is None:
            return False
        
        idle_time = time.time() - self.last_used
        
        if idle_time > self.idle_timeout:
            logger.info(f"Model idle for {idle_time:.0f}s, unloading...")
            return self.unload_model()
        
        return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (uses default if None)
            temperature: Sampling temperature (uses default if None)
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
        
        Returns:
            Dictionary with generation results:
                - text: Generated text
                - tokens_used: Number of tokens generated
                - success: Whether generation succeeded
                - error: Error message if failed
        """
        # Ensure model is loaded
        if not self.is_loaded():
            logger.info("Model not loaded, loading now...")
            if not self.load_model():
                return {
                    'text': '',
                    'tokens_used': 0,
                    'success': False,
                    'error': 'Failed to load model'
                }
        
        # Update last used timestamp
        self.last_used = time.time()
        
        # Use defaults if not specified
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        
        try:
            logger.info(f"Generating response (max_tokens={max_tokens}, temp={temperature})")
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            tokens_used = response['usage']['completion_tokens']
            
            logger.info(f"Generated {tokens_used} tokens")
            
            return {
                'text': generated_text,
                'tokens_used': tokens_used,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                'text': '',
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str = "You are a helpful AI tutor specializing in NCERT content for JEE and NEET preparation.",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using query and retrieved context.
        
        Args:
            query: User's question
            context: Retrieved NCERT context
            system_prompt: System instruction for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dictionary with generation results (same as generate())
        """
        # Construct prompt with context
        prompt = f"""{system_prompt}

Context from NCERT textbooks:
{context}

Student's question: {query}

Provide a clear, accurate answer based ONLY on the context provided above. If the context doesn't contain enough information, say so.

Answer:"""
        
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.is_loaded():
            logger.info("ModelManager being destroyed, unloading model...")
            self.unload_model()


class ModelManagerSingleton:
    """
    Singleton wrapper for ModelManager to ensure only one instance exists.
    """
    _instance: Optional[ModelManager] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, model_path: str = None, config: Dict[str, Any] = None) -> ModelManager:
        """
        Get or create the ModelManager instance.
        
        Args:
            model_path: Path to model file (required for first call)
            config: Configuration dictionary (required for first call)
        
        Returns:
            ModelManager instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if model_path is None or config is None:
                        raise ValueError("model_path and config required for first initialization")
                    cls._instance = ModelManager(model_path, config)
        
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unload_model()
            cls._instance = None
