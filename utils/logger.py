"""
Logging Configuration for GuruAI.

Provides centralized logging setup with rotation, formatting,
and different log levels for different components.

Requirements: Error handling requirements
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logging(
    app_name: str = 'guruai',
    log_dir: Optional[Path] = None,
    log_level: str = 'INFO',
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        app_name: Name of the application (used for log file naming)
        log_dir: Directory to store log files (default: ./logs)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
        log_format: Custom log format string
    
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / 'logs'
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default log format
    if log_format is None:
        log_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[%(filename)s:%(lineno)d] - %(message)s'
        )
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        # Main log file
        log_file = log_dir / f'{app_name}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        # Use standard formatter for file
        file_formatter = logging.Formatter(
            log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error log file (only ERROR and CRITICAL)
        error_log_file = log_dir / f'{app_name}_error.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"Logging initialized for {app_name}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_dir}")
    
    return logger


def setup_component_logger(
    component_name: str,
    parent_logger: Optional[logging.Logger] = None,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger for a specific component.
    
    Args:
        component_name: Name of the component
        parent_logger: Parent logger to inherit from
        log_level: Optional log level override
    
    Returns:
        Component logger
    """
    if parent_logger:
        logger_name = f"{parent_logger.name}.{component_name}"
    else:
        logger_name = component_name
    
    logger = logging.getLogger(logger_name)
    
    if log_level:
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and results.
    
    Args:
        logger: Logger instance to use
    
    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            # Your code here
    """
    from functools import wraps
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Log function call
            args_str = ', '.join([repr(arg) for arg in args])
            kwargs_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            all_args = ', '.join(filter(None, [args_str, kwargs_str]))
            
            logger.debug(f"Calling {f.__name__}({all_args})")
            
            try:
                result = f(*args, **kwargs)
                logger.debug(f"{f.__name__} returned: {repr(result)}")
                return result
            except Exception as e:
                logger.error(f"{f.__name__} raised {type(e).__name__}: {str(e)}")
                raise
        
        return wrapped
    return decorator


def log_performance(logger: logging.Logger, threshold_seconds: float = 1.0):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use
        threshold_seconds: Log warning if execution exceeds this threshold
    
    Usage:
        @log_performance(logger, threshold_seconds=2.0)
        def my_slow_function():
            # Your code here
    """
    from functools import wraps
    import time
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                
                if elapsed > threshold_seconds:
                    logger.warning(
                        f"{f.__name__} took {elapsed:.2f}s (threshold: {threshold_seconds}s)"
                    )
                else:
                    logger.debug(f"{f.__name__} took {elapsed:.2f}s")
        
        return wrapped
    return decorator


class RequestLogger:
    """Context manager for logging request processing."""
    
    def __init__(self, logger: logging.Logger, request_id: str, endpoint: str):
        self.logger = logger
        self.request_id = request_id
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(
            f"[{self.request_id}] Processing request: {self.endpoint}"
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                f"[{self.request_id}] Request completed successfully in {elapsed:.2f}s"
            )
        else:
            self.logger.error(
                f"[{self.request_id}] Request failed after {elapsed:.2f}s: "
                f"{exc_type.__name__}: {str(exc_val)}"
            )
        
        # Don't suppress exceptions
        return False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: str):
    """
    Set log level for a logger.
    
    Args:
        logger: Logger instance
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Also update all handlers
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def cleanup_old_logs(log_dir: Path, days_to_keep: int = 30):
    """
    Clean up log files older than specified days.
    
    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days to keep logs
    """
    import time
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return
    
    cutoff_time = time.time() - (days_to_keep * 86400)
    
    for log_file in log_dir.glob('*.log*'):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                logging.info(f"Deleted old log file: {log_file}")
            except Exception as e:
                logging.error(f"Failed to delete log file {log_file}: {e}")


# Initialize default logger for the module
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get the default application logger."""
    global _default_logger
    
    if _default_logger is None:
        _default_logger = setup_logging()
    
    return _default_logger
