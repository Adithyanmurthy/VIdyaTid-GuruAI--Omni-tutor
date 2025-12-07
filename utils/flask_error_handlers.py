"""
Flask Error Handlers for GuruAI.

Registers global error handlers for Flask application to provide
consistent error responses across all endpoints.

Requirements: Error handling requirements
"""

import logging
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from sqlalchemy.exc import SQLAlchemyError

from utils.error_handler import (
    GuruAIError,
    format_error_response,
    sanitize_error_message
)

logger = logging.getLogger(__name__)


def register_error_handlers(app: Flask, include_traceback: bool = False):
    """
    Register error handlers for Flask application.
    
    Args:
        app: Flask application instance
        include_traceback: Whether to include traceback in error responses
    """
    
    @app.errorhandler(GuruAIError)
    def handle_guruai_error(error: GuruAIError):
        """Handle custom GuruAI errors."""
        # Log based on severity
        if error.status_code >= 500:
            logger.error(
                f"GuruAI Error: {error.code} - {error.message}",
                exc_info=True,
                extra={
                    'error_code': error.code,
                    'status_code': error.status_code,
                    'endpoint': request.endpoint,
                    'method': request.method,
                    'path': request.path
                }
            )
        else:
            logger.warning(
                f"Client Error: {error.code} - {error.message}",
                extra={
                    'error_code': error.code,
                    'status_code': error.status_code,
                    'endpoint': request.endpoint
                }
            )
        
        response, status_code = format_error_response(error, include_traceback)
        return jsonify(response), status_code
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        """Handle Werkzeug HTTP exceptions."""
        logger.warning(
            f"HTTP Exception: {error.code} - {error.name}",
            extra={
                'status_code': error.code,
                'endpoint': request.endpoint,
                'method': request.method,
                'path': request.path
            }
        )
        
        response = {
            'error': {
                'code': error.name.upper().replace(' ', '_'),
                'message': error.description or error.name,
                'status_code': error.code
            }
        }
        
        return jsonify(response), error.code
    
    @app.errorhandler(SQLAlchemyError)
    def handle_database_error(error: SQLAlchemyError):
        """Handle database errors."""
        logger.error(
            f"Database Error: {str(error)}",
            exc_info=True,
            extra={
                'endpoint': request.endpoint,
                'method': request.method,
                'path': request.path
            }
        )
        
        # Don't expose database details to client
        response = {
            'error': {
                'code': 'DATABASE_ERROR',
                'message': 'A database error occurred',
                'details': sanitize_error_message(str(error), include_traceback)
            }
        }
        
        return jsonify(response), 500
    
    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError):
        """Handle value errors (often from invalid input)."""
        logger.warning(
            f"Value Error: {str(error)}",
            extra={
                'endpoint': request.endpoint,
                'method': request.method
            }
        )
        
        response = {
            'error': {
                'code': 'INVALID_VALUE',
                'message': 'Invalid value provided',
                'details': str(error) if include_traceback else None
            }
        }
        
        return jsonify(response), 400
    
    @app.errorhandler(KeyError)
    def handle_key_error(error: KeyError):
        """Handle key errors (often from missing required fields)."""
        logger.warning(
            f"Key Error: {str(error)}",
            extra={
                'endpoint': request.endpoint,
                'method': request.method
            }
        )
        
        response = {
            'error': {
                'code': 'MISSING_FIELD',
                'message': f'Required field missing: {str(error)}',
                'details': 'Check API documentation for required fields'
            }
        }
        
        return jsonify(response), 400
    
    @app.errorhandler(TypeError)
    def handle_type_error(error: TypeError):
        """Handle type errors."""
        logger.error(
            f"Type Error: {str(error)}",
            exc_info=True,
            extra={
                'endpoint': request.endpoint,
                'method': request.method
            }
        )
        
        response = {
            'error': {
                'code': 'TYPE_ERROR',
                'message': 'Invalid data type',
                'details': str(error) if include_traceback else None
            }
        }
        
        return jsonify(response), 400
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 Not Found errors."""
        from datetime import datetime
        
        logger.info(
            f"404 Not Found: {request.path}",
            extra={
                'method': request.method,
                'path': request.path
            }
        )
        
        response = {
            'error': {
                'code': 'NOT_FOUND',
                'message': f'Endpoint not found: {request.path}',
                'available_endpoints': 'See API documentation',
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        return jsonify(response), 404
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 Method Not Allowed errors."""
        logger.warning(
            f"405 Method Not Allowed: {request.method} {request.path}",
            extra={
                'method': request.method,
                'path': request.path
            }
        )
        
        response = {
            'error': {
                'code': 'METHOD_NOT_ALLOWED',
                'message': f'Method {request.method} not allowed for {request.path}',
                'allowed_methods': error.valid_methods if hasattr(error, 'valid_methods') else None
            }
        }
        
        return jsonify(response), 405
    
    @app.errorhandler(413)
    def handle_request_entity_too_large(error):
        """Handle 413 Request Entity Too Large errors."""
        logger.warning(
            f"413 Request Too Large: {request.path}",
            extra={
                'method': request.method,
                'path': request.path,
                'content_length': request.content_length
            }
        )
        
        response = {
            'error': {
                'code': 'REQUEST_TOO_LARGE',
                'message': 'Request entity too large',
                'details': f'Maximum allowed size: {app.config.get("MAX_CONTENT_LENGTH", "unknown")} bytes'
            }
        }
        
        return jsonify(response), 413
    
    @app.errorhandler(429)
    def handle_too_many_requests(error):
        """Handle 429 Too Many Requests errors."""
        logger.warning(
            f"429 Too Many Requests: {request.path}",
            extra={
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr
            }
        )
        
        response = {
            'error': {
                'code': 'RATE_LIMIT_EXCEEDED',
                'message': 'Too many requests',
                'details': 'Please slow down and try again later'
            }
        }
        
        return jsonify(response), 429
    
    @app.errorhandler(500)
    def handle_internal_server_error(error):
        """Handle 500 Internal Server Error."""
        logger.error(
            f"500 Internal Server Error: {str(error)}",
            exc_info=True,
            extra={
                'endpoint': request.endpoint,
                'method': request.method,
                'path': request.path
            }
        )
        
        response = {
            'error': {
                'code': 'INTERNAL_SERVER_ERROR',
                'message': 'An internal server error occurred',
                'details': sanitize_error_message(str(error), include_traceback)
            }
        }
        
        if include_traceback:
            import traceback
            response['error']['traceback'] = traceback.format_exc()
        
        return jsonify(response), 500
    
    @app.errorhandler(503)
    def handle_service_unavailable(error):
        """Handle 503 Service Unavailable errors."""
        logger.error(
            f"503 Service Unavailable: {str(error)}",
            extra={
                'endpoint': request.endpoint,
                'method': request.method,
                'path': request.path
            }
        )
        
        response = {
            'error': {
                'code': 'SERVICE_UNAVAILABLE',
                'message': 'Service temporarily unavailable',
                'details': 'Please try again later'
            }
        }
        
        return jsonify(response), 503
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        """Handle any unexpected errors."""
        from datetime import datetime
        
        logger.critical(
            f"Unexpected Error: {type(error).__name__} - {str(error)}",
            exc_info=True,
            extra={
                'endpoint': request.endpoint,
                'method': request.method,
                'path': request.path,
                'error_type': type(error).__name__
            }
        )
        
        response = {
            'error': {
                'code': 'UNEXPECTED_ERROR',
                'message': 'An unexpected error occurred',
                'error_type': type(error).__name__,
                'details': sanitize_error_message(str(error), include_traceback),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        if include_traceback:
            import traceback
            response['error']['traceback'] = traceback.format_exc()
        
        return jsonify(response), 500
    
    logger.info("Error handlers registered successfully")


def register_request_logging(app: Flask):
    """
    Register request/response logging middleware.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def log_request():
        """Log incoming requests."""
        logger.info(
            f"Incoming request: {request.method} {request.path}",
            extra={
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', 'Unknown')
            }
        )
    
    @app.after_request
    def log_response(response):
        """Log outgoing responses."""
        logger.info(
            f"Response: {response.status_code} for {request.method} {request.path}",
            extra={
                'method': request.method,
                'path': request.path,
                'status_code': response.status_code,
                'content_length': response.content_length
            }
        )
        return response
    
    logger.info("Request logging middleware registered")
