"""
Error Handler Utilities for GuruAI.

Provides standardized error handling, error response formatting,
and error recovery mechanisms across the application.

Requirements: Error handling requirements
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from functools import wraps
from flask import jsonify, request

logger = logging.getLogger(__name__)


class GuruAIError(Exception):
    """Base exception class for GuruAI application."""
    
    def __init__(self, message: str, code: str = "GENERAL_ERROR", 
                 details: Optional[str] = None, status_code: int = 500):
        self.message = message
        self.code = code
        self.details = details
        self.status_code = status_code
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        error_dict = {
            'code': self.code,
            'message': self.message,
            'timestamp': self.timestamp
        }
        if self.details:
            error_dict['details'] = self.details
        return error_dict


class ValidationError(GuruAIError):
    """Exception for input validation errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=details,
            status_code=400
        )


class AuthenticationError(GuruAIError):
    """Exception for authentication errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            details=details,
            status_code=401
        )


class AuthorizationError(GuruAIError):
    """Exception for authorization errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            details=details,
            status_code=403
        )


class ResourceNotFoundError(GuruAIError):
    """Exception for resource not found errors."""
    
    def __init__(self, resource: str, details: Optional[str] = None):
        super().__init__(
            message=f"{resource} not found",
            code="RESOURCE_NOT_FOUND",
            details=details,
            status_code=404
        )


class ModelError(GuruAIError):
    """Exception for AI model errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            code="MODEL_ERROR",
            details=details,
            status_code=500
        )


class DatabaseError(GuruAIError):
    """Exception for database errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            code="DATABASE_ERROR",
            details=details,
            status_code=500
        )


class ImageProcessingError(GuruAIError):
    """Exception for image processing errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            code="IMAGE_PROCESSING_ERROR",
            details=details,
            status_code=400
        )


class ContentNotFoundError(GuruAIError):
    """Exception for NCERT content not found errors."""
    
    def __init__(self, message: str = "Content not found in NCERT textbooks", 
                 details: Optional[str] = None):
        super().__init__(
            message=message,
            code="CONTENT_NOT_FOUND",
            details=details,
            status_code=404
        )


class RateLimitError(GuruAIError):
    """Exception for rate limiting errors."""
    
    def __init__(self, message: str = "Rate limit exceeded", 
                 details: Optional[str] = None):
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            details=details,
            status_code=429
        )


def format_error_response(error: Exception, 
                         include_traceback: bool = False) -> Tuple[Dict[str, Any], int]:
    """
    Format an exception into a standardized error response.
    
    Args:
        error: The exception to format
        include_traceback: Whether to include traceback in response (for debugging)
    
    Returns:
        Tuple of (response_dict, status_code)
    """
    if isinstance(error, GuruAIError):
        response = {
            'error': error.to_dict()
        }
        status_code = error.status_code
    else:
        # Generic error
        response = {
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': 'An unexpected error occurred',
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        status_code = 500
        
        # Log the full error
        logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    
    # Add traceback for debugging if requested
    if include_traceback:
        response['error']['traceback'] = traceback.format_exc()
    
    return response, status_code


def handle_errors(include_traceback: bool = False):
    """
    Decorator to handle errors in route handlers.
    
    Args:
        include_traceback: Whether to include traceback in error responses
    
    Usage:
        @app.route('/api/endpoint')
        @handle_errors()
        def my_endpoint():
            # Your code here
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except GuruAIError as e:
                # Log custom errors at appropriate level
                if e.status_code >= 500:
                    logger.error(f"Error in {f.__name__}: {e.message}", exc_info=True)
                else:
                    logger.warning(f"Client error in {f.__name__}: {e.message}")
                
                response, status_code = format_error_response(e, include_traceback)
                return jsonify(response), status_code
            
            except Exception as e:
                # Log unexpected errors
                logger.error(f"Unexpected error in {f.__name__}: {str(e)}", exc_info=True)
                
                response, status_code = format_error_response(e, include_traceback)
                return jsonify(response), status_code
        
        return wrapped
    return decorator


def validate_request_data(required_fields: list, 
                         optional_fields: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate request JSON data and extract fields.
    
    Args:
        required_fields: List of required field names
        optional_fields: List of optional field names
    
    Returns:
        Dictionary of validated data
    
    Raises:
        ValidationError: If validation fails
    """
    data = request.get_json()
    
    if not data:
        raise ValidationError("No JSON data provided")
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            details=f"Required fields: {', '.join(required_fields)}"
        )
    
    # Extract fields
    result = {}
    for field in required_fields:
        result[field] = data[field]
    
    if optional_fields:
        for field in optional_fields:
            if field in data:
                result[field] = data[field]
    
    return result


def validate_file_upload(file_key: str = 'file', 
                        allowed_extensions: Optional[set] = None,
                        max_size_mb: Optional[int] = None) -> bytes:
    """
    Validate file upload from request.
    
    Args:
        file_key: Key name for the file in request.files
        allowed_extensions: Set of allowed file extensions (e.g., {'png', 'jpg'})
        max_size_mb: Maximum file size in megabytes
    
    Returns:
        File data as bytes
    
    Raises:
        ValidationError: If validation fails
    """
    if file_key not in request.files:
        raise ValidationError(f"No file provided with key '{file_key}'")
    
    file = request.files[file_key]
    
    if file.filename == '':
        raise ValidationError("Empty filename")
    
    # Check file extension
    if allowed_extensions:
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            raise ValidationError(
                f"Invalid file type: .{file_ext}",
                details=f"Allowed types: {', '.join(allowed_extensions)}"
            )
    
    # Read file data
    file_data = file.read()
    
    # Check file size
    if max_size_mb:
        max_size_bytes = max_size_mb * 1024 * 1024
        if len(file_data) > max_size_bytes:
            raise ValidationError(
                f"File too large: {len(file_data) / (1024*1024):.2f}MB",
                details=f"Maximum allowed size: {max_size_mb}MB"
            )
    
    return file_data


def retry_on_failure(max_retries: int = 3, 
                    delay_seconds: float = 1.0,
                    exceptions: tuple = (Exception,)):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay_seconds: Delay between retries in seconds
        exceptions: Tuple of exception types to catch and retry
    
    Usage:
        @retry_on_failure(max_retries=3, delay_seconds=2.0)
        def my_function():
            # Your code here
    """
    import time
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {f.__name__}: {str(e)}. "
                            f"Retrying in {delay_seconds}s..."
                        )
                        time.sleep(delay_seconds)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {f.__name__}: {str(e)}"
                        )
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapped
    return decorator


def log_request_info():
    """
    Log information about the current request.
    Useful for debugging and monitoring.
    """
    logger.info(
        f"Request: {request.method} {request.path} "
        f"from {request.remote_addr} "
        f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}"
    )


def create_success_response(data: Dict[str, Any], 
                           message: Optional[str] = None,
                           status_code: int = 200) -> Tuple[Dict[str, Any], int]:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code
    
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'success': True,
        **data
    }
    
    if message:
        response['message'] = message
    
    return response, status_code


def sanitize_error_message(message: str, include_details: bool = False) -> str:
    """
    Sanitize error message to remove sensitive information.
    
    Args:
        message: Original error message
        include_details: Whether to include technical details
    
    Returns:
        Sanitized error message
    """
    # Remove file paths
    import re
    message = re.sub(r'[A-Za-z]:\\[^\s]+', '[PATH]', message)
    message = re.sub(r'/[^\s]+', '[PATH]', message)
    
    # Remove potential sensitive data patterns
    message = re.sub(r'\b\d{10,}\b', '[REDACTED]', message)  # Long numbers
    message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', message)
    
    if not include_details:
        # Generic message for production
        return "An error occurred while processing your request"
    
    return message
