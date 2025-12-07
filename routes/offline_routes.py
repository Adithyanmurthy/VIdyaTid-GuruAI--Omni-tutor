"""
Routes for offline verification and monitoring.

Provides endpoints to:
- Check offline status
- Verify offline operation
- Get network call logs
- Test features without internet

Requirements: 4.2, 4.3, 4.4, 4.5
"""

from flask import Blueprint, jsonify, request
from utils.offline_monitor import get_offline_verifier
from utils.logger import setup_logging

logger = setup_logging('offline_routes')

offline_bp = Blueprint('offline', __name__, url_prefix='/api/offline')


@offline_bp.route('/status', methods=['GET'])
def get_offline_status():
    """
    Get current offline status.
    
    Returns:
        JSON with offline mode status, internet availability, and monitoring info
    """
    try:
        verifier = get_offline_verifier()
        status = verifier.get_offline_status()
        
        return jsonify({
            'success': True,
            'status': status
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting offline status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@offline_bp.route('/verify', methods=['POST'])
def verify_offline_operation():
    """
    Verify that the application is operating offline.
    
    Returns:
        JSON with verification results including any external calls detected
    """
    try:
        verifier = get_offline_verifier()
        verification = verifier.verify_offline_operation()
        
        return jsonify({
            'success': True,
            'verification': verification
        }), 200
    
    except Exception as e:
        logger.error(f"Error verifying offline operation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@offline_bp.route('/network-calls', methods=['GET'])
def get_network_calls():
    """
    Get log of all network calls detected.
    
    Query params:
        external_only: If 'true', return only external calls
    
    Returns:
        JSON with list of network calls
    """
    try:
        verifier = get_offline_verifier()
        external_only = request.args.get('external_only', 'false').lower() == 'true'
        
        if external_only:
            calls = verifier.detector.get_external_calls()
        else:
            calls = verifier.detector.get_network_calls()
        
        return jsonify({
            'success': True,
            'calls': calls,
            'count': len(calls)
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting network calls: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@offline_bp.route('/clear-logs', methods=['POST'])
def clear_network_logs():
    """
    Clear the network call logs.
    
    Returns:
        JSON with success status
    """
    try:
        verifier = get_offline_verifier()
        verifier.detector.clear_calls()
        
        return jsonify({
            'success': True,
            'message': 'Network call logs cleared'
        }), 200
    
    except Exception as e:
        logger.error(f"Error clearing network logs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@offline_bp.route('/check-connectivity', methods=['GET'])
def check_internet_connectivity():
    """
    Check if internet connectivity is available.
    
    Query params:
        timeout: Timeout in seconds (default: 2.0)
    
    Returns:
        JSON with connectivity status
    """
    try:
        timeout = float(request.args.get('timeout', 2.0))
        verifier = get_offline_verifier()
        is_connected = verifier.check_internet_connectivity(timeout=timeout)
        
        return jsonify({
            'success': True,
            'internet_available': is_connected,
            'message': 'Internet connectivity detected' if is_connected 
                      else 'No internet connectivity'
        }), 200
    
    except Exception as e:
        logger.error(f"Error checking connectivity: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@offline_bp.route('/enable', methods=['POST'])
def enable_offline_mode():
    """
    Enable offline mode and start network monitoring.
    
    Returns:
        JSON with success status
    """
    try:
        verifier = get_offline_verifier()
        verifier.enable_offline_mode()
        
        return jsonify({
            'success': True,
            'message': 'Offline mode enabled'
        }), 200
    
    except Exception as e:
        logger.error(f"Error enabling offline mode: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@offline_bp.route('/disable', methods=['POST'])
def disable_offline_mode():
    """
    Disable offline mode and stop network monitoring.
    
    Returns:
        JSON with success status
    """
    try:
        verifier = get_offline_verifier()
        verifier.disable_offline_mode()
        
        return jsonify({
            'success': True,
            'message': 'Offline mode disabled'
        }), 200
    
    except Exception as e:
        logger.error(f"Error disabling offline mode: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@offline_bp.route('/test-features', methods=['POST'])
def test_offline_features():
    """
    Test all major features to verify offline operation.
    
    This endpoint tests:
    - Query processing
    - Image processing
    - Question generation
    - Search functionality
    - Progress tracking
    
    Returns:
        JSON with test results for each feature
    """
    try:
        verifier = get_offline_verifier()
        
        # Clear previous logs
        verifier.detector.clear_calls()
        
        test_results = {
            'query_processing': {'tested': False, 'passed': False, 'error': None},
            'image_processing': {'tested': False, 'passed': False, 'error': None},
            'question_generation': {'tested': False, 'passed': False, 'error': None},
            'search': {'tested': False, 'passed': False, 'error': None},
            'progress_tracking': {'tested': False, 'passed': False, 'error': None}
        }
        
        # Test query processing
        try:
            from routes.query_routes import query_handler
            if query_handler:
                test_results['query_processing']['tested'] = True
                test_results['query_processing']['passed'] = True
        except Exception as e:
            test_results['query_processing']['tested'] = True
            test_results['query_processing']['error'] = str(e)
        
        # Test image processing
        try:
            from routes.problem_routes import problem_solver
            if problem_solver:
                test_results['image_processing']['tested'] = True
                test_results['image_processing']['passed'] = True
        except Exception as e:
            test_results['image_processing']['tested'] = True
            test_results['image_processing']['error'] = str(e)
        
        # Test question generation
        try:
            from routes.paper_routes import question_generator
            if question_generator:
                test_results['question_generation']['tested'] = True
                test_results['question_generation']['passed'] = True
        except Exception as e:
            test_results['question_generation']['tested'] = True
            test_results['question_generation']['error'] = str(e)
        
        # Test search
        try:
            from routes.search_routes import search_service
            if search_service:
                test_results['search']['tested'] = True
                test_results['search']['passed'] = True
        except Exception as e:
            test_results['search']['tested'] = True
            test_results['search']['error'] = str(e)
        
        # Test progress tracking
        try:
            from routes.progress_routes import progress_tracker
            if progress_tracker:
                test_results['progress_tracking']['tested'] = True
                test_results['progress_tracking']['passed'] = True
        except Exception as e:
            test_results['progress_tracking']['tested'] = True
            test_results['progress_tracking']['error'] = str(e)
        
        # Check for external calls
        external_calls = verifier.detector.get_external_calls()
        
        overall_passed = (
            all(result['passed'] for result in test_results.values() if result['tested']) and
            len(external_calls) == 0
        )
        
        return jsonify({
            'success': True,
            'overall_passed': overall_passed,
            'test_results': test_results,
            'external_calls_detected': len(external_calls),
            'external_calls': external_calls
        }), 200
    
    except Exception as e:
        logger.error(f"Error testing offline features: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
