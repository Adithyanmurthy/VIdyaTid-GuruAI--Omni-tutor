"""
Voice Routes
Handles voice input transcription (Gemini) and text-to-speech (ElevenLabs)
"""

from flask import Blueprint, request, jsonify, session, send_file
import logging
import io

voice_bp = Blueprint('voice', __name__, url_prefix='/api/voice')
logger = logging.getLogger(__name__)


@voice_bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio to text.
    
    Accepts audio file and returns transcribed text.
    """
    try:
        # Check if user is logged in (optional for now during development)
        user_id = session.get('user_id', 'anonymous')
        
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided',
                'text': ''
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty audio file',
                'text': ''
            }), 400
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Get MIME type
        mime_type = audio_file.content_type or 'audio/wav'
        
        logger.info(f"Audio transcription requested by user {user_id}")
        logger.info(f"Audio size: {len(audio_data)} bytes, type: {mime_type}")
        
        # Use Gemini for transcription
        from services.voice_service import get_voice_service
        voice_service = get_voice_service()
        
        result = voice_service.transcribe_audio(audio_data, mime_type)
        
        if result['success']:
            return jsonify({
                'success': True,
                'text': result['text'],
                'language': result.get('language', 'en')
            }), 200
        else:
            return jsonify({
                'success': False,
                'text': '',
                'error': result.get('error', 'Transcription failed')
            }), 500
    
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'text': ''
        }), 500


@voice_bp.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """
    Convert text to speech.
    
    Accepts text and returns audio file.
    """
    try:
        # Check if user is logged in
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get voice settings
        voice_id = data.get('voice_id', None)  # ElevenLabs voice ID
        
        logger.info(f"TTS requested by user {user_id}")
        logger.info(f"Text length: {len(text)} characters")
        
        # Use ElevenLabs for TTS
        from services.voice_service import get_voice_service
        voice_service = get_voice_service()
        
        result = voice_service.synthesize_speech(text, voice_id)
        
        if result['success']:
            # Return audio file
            audio_io = io.BytesIO(result['audio'])
            audio_io.seek(0)
            
            return send_file(
                audio_io,
                mimetype='audio/mpeg',
                as_attachment=False,
                download_name='speech.mp3'
            )
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'TTS failed')
            }), 500
    
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500


@voice_bp.route('/status', methods=['GET'])
def voice_status():
    """
    Get voice features status.
    """
    from services.voice_service import get_voice_service
    voice_service = get_voice_service()
    
    return jsonify({
        'transcription': {
            'available': True,
            'provider': 'Google Gemini',
            'status': 'active',
            'message': 'Voice input powered by Gemini'
        },
        'tts': {
            'available': bool(voice_service.elevenlabs_client),
            'provider': 'ElevenLabs',
            'status': 'active' if voice_service.elevenlabs_client else 'not_configured',
            'message': 'Text-to-speech powered by ElevenLabs'
        },
        'supported_languages': ['en', 'hi', 'hinglish'],
        'supported_voices': voice_service.get_available_voices()
    })


@voice_bp.route('/voices', methods=['GET'])
def list_voices():
    """
    Get list of available voices from ElevenLabs.
    """
    try:
        from services.voice_service import get_voice_service
        voice_service = get_voice_service()
        
        voices = voice_service.get_available_voices()
        
        return jsonify({
            'success': True,
            'voices': voices
        })
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
