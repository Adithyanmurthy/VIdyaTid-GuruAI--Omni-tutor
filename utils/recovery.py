"""
Error Recovery Mechanisms for GuruAI.

Provides automatic recovery strategies for common failure scenarios
such as model loading failures, database connection issues, etc.

Requirements: Error handling requirements
"""

import logging
import time
from typing import Callable, Any, Optional, Dict
from functools import wraps

logger = logging.getLogger(__name__)


class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def __init__(self, max_attempts: int = 3, delay_seconds: float = 1.0):
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with recovery strategy.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
        
        Returns:
            Function result
        
        Raises:
            Exception: If all recovery attempts fail
        """
        raise NotImplementedError("Subclasses must implement execute()")


class ExponentialBackoffStrategy(RecoveryStrategy):
    """Recovery strategy with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, initial_delay: float = 1.0, 
                 backoff_factor: float = 2.0, max_delay: float = 60.0):
        super().__init__(max_attempts, initial_delay)
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with exponential backoff."""
        delay = self.delay_seconds
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.debug(f"Attempt {attempt}/{self.max_attempts} for {func.__name__}")
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_attempts:
                    logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    
                    # Exponential backoff
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    logger.error(
                        f"All {self.max_attempts} attempts failed for {func.__name__}: {str(e)}"
                    )
        
        raise last_exception


class CircuitBreakerStrategy(RecoveryStrategy):
    """
    Circuit breaker pattern for preventing cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """
    
    STATE_CLOSED = 'CLOSED'
    STATE_OPEN = 'OPEN'
    STATE_HALF_OPEN = 'HALF_OPEN'
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: float = 60.0):
        super().__init__()
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.STATE_CLOSED
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with circuit breaker."""
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == self.STATE_OPEN:
            if time.time() - self.last_failure_time >= self.timeout_seconds:
                logger.info(f"Circuit breaker transitioning to HALF_OPEN for {func.__name__}")
                self.state = self.STATE_HALF_OPEN
            else:
                raise Exception(
                    f"Circuit breaker is OPEN for {func.__name__}. "
                    f"Service unavailable. Try again later."
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit breaker
            if self.state == self.STATE_HALF_OPEN:
                logger.info(f"Circuit breaker closing for {func.__name__}")
                self.state = self.STATE_CLOSED
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                f"Failure {self.failure_count}/{self.failure_threshold} "
                f"for {func.__name__}: {str(e)}"
            )
            
            # Open circuit if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker opening for {func.__name__} "
                    f"after {self.failure_count} failures"
                )
                self.state = self.STATE_OPEN
            
            raise


class FallbackStrategy(RecoveryStrategy):
    """Recovery strategy with fallback function."""
    
    def __init__(self, fallback_func: Callable, 
                 fallback_args: tuple = (), 
                 fallback_kwargs: dict = None):
        super().__init__()
        self.fallback_func = fallback_func
        self.fallback_args = fallback_args
        self.fallback_kwargs = fallback_kwargs or {}
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with fallback."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Primary function {func.__name__} failed: {str(e)}. "
                f"Using fallback: {self.fallback_func.__name__}"
            )
            return self.fallback_func(*self.fallback_args, **self.fallback_kwargs)


def with_recovery(strategy: RecoveryStrategy):
    """
    Decorator to add recovery strategy to a function.
    
    Args:
        strategy: Recovery strategy to use
    
    Usage:
        @with_recovery(ExponentialBackoffStrategy(max_attempts=3))
        def my_function():
            # Your code here
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return strategy.execute(func, *args, **kwargs)
        return wrapper
    return decorator


class ModelLoadingRecovery:
    """Specialized recovery for AI model loading failures."""
    
    @staticmethod
    def recover_model_loading(model_path: str, config: Dict[str, Any]) -> Any:
        """
        Attempt to recover from model loading failure.
        
        Args:
            model_path: Path to model file
            config: Model configuration
        
        Returns:
            Loaded model or None
        """
        logger.info(f"Attempting to recover model loading from {model_path}")
        
        # Strategy 1: Clear cache and retry
        try:
            import gc
            gc.collect()
            logger.info("Cleared memory cache, retrying model load...")
            
            # Import here to avoid circular dependency
            from services.model_manager import ModelManagerSingleton
            model_manager = ModelManagerSingleton.get_instance(model_path, config)
            return model_manager
        
        except Exception as e:
            logger.error(f"Cache clearing recovery failed: {e}")
        
        # Strategy 2: Try with reduced context size
        try:
            logger.info("Attempting model load with reduced context size...")
            reduced_config = config.copy()
            reduced_config['n_ctx'] = min(config.get('n_ctx', 2048), 1024)
            
            from services.model_manager import ModelManagerSingleton
            model_manager = ModelManagerSingleton.get_instance(model_path, reduced_config)
            logger.warning("Model loaded with reduced context size")
            return model_manager
        
        except Exception as e:
            logger.error(f"Reduced context recovery failed: {e}")
        
        logger.error("All model loading recovery strategies failed")
        return None


class DatabaseRecovery:
    """Specialized recovery for database connection failures."""
    
    @staticmethod
    def recover_database_connection(max_attempts: int = 3) -> bool:
        """
        Attempt to recover database connection.
        
        Args:
            max_attempts: Maximum recovery attempts
        
        Returns:
            True if recovery successful, False otherwise
        """
        logger.info("Attempting to recover database connection")
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Reinitialize database
                from models.database import init_db, create_tables
                init_db()
                create_tables()
                
                logger.info(f"Database connection recovered on attempt {attempt}")
                return True
            
            except Exception as e:
                logger.warning(f"Database recovery attempt {attempt} failed: {e}")
                
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("Database connection recovery failed")
        return False


class VectorStoreRecovery:
    """Specialized recovery for vector store failures."""
    
    @staticmethod
    def recover_vector_store(collection_name: str) -> bool:
        """
        Attempt to recover vector store connection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            True if recovery successful, False otherwise
        """
        logger.info(f"Attempting to recover vector store: {collection_name}")
        
        try:
            # Try to reconnect to ChromaDB
            from services.rag_system import RAGSystem
            rag = RAGSystem()
            
            # Test connection
            stats = rag.get_stats()
            
            if stats.get('total_documents', 0) > 0:
                logger.info("Vector store connection recovered")
                return True
            else:
                logger.warning("Vector store connected but no documents found")
                return False
        
        except Exception as e:
            logger.error(f"Vector store recovery failed: {e}")
            return False


def graceful_degradation(fallback_value: Any = None, 
                        log_level: str = 'ERROR'):
    """
    Decorator for graceful degradation - return fallback value on error.
    
    Args:
        fallback_value: Value to return on error
        log_level: Logging level for errors
    
    Usage:
        @graceful_degradation(fallback_value={})
        def my_function():
            # Your code here
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_func = getattr(logger, log_level.lower(), logger.error)
                log_func(
                    f"Function {func.__name__} failed, using fallback value: {str(e)}"
                )
                return fallback_value
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, 
                default_value: Any = None, 
                log_errors: bool = True, 
                **kwargs) -> Any:
    """
    Safely execute a function and return default value on error.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_value: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default_value
