"""
Voice Service - Gemini for transcription, ElevenLabs for TTS
"""
import os
import logging
import google.generativeai as genai
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

logger = logging.getLogger(__name__)

class VoiceService:
    """Voice service using Gemini for input and ElevenLabs for output"""
    
    def __init__(self):
        # Initialize Gemini for voice input (supports multiple keys)
        api_keys_str = os.getenv("GEMINI_API_KEYS", "")
        gemini_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        
        # Fall back to single key if multiple keys not set
        if not gemini_keys:
            single_key = os.getenv("GEMINI_API_KEY", "")
            if single_key:
                gemini_keys = [single_key]
        
        self.gemini_key = gemini_keys[0] if gemini_keys else None
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            logger.info("✅ Gemini configured for voice transcription")
        
        # Initialize ElevenLabs for voice output
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        if self.elevenlabs_key:
            self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_key)
            logger.info("✅ ElevenLabs configured for voice synthesis")
        else:
            self.elevenlabs_client = None
            logger.warning("⚠️ ElevenLabs API key not found")
    
    def transcribe_audio(self, audio_data: bytes, mime_type: str = "audio/wav") -> dict:
        """
        Transcribe audio - Currently disabled due to API limitations
        
        Args:
            audio_data: Audio file bytes
            mime_type: MIME type of audio
        
        Returns:
            Dictionary with transcription result
        """
        # Voice input temporarily disabled - Gemini audio API has rate limits
        logger.warning("Voice input feature temporarily disabled")
        return {
            'success': False,
            'text': '',
            'error': 'Voice input is temporarily disabled. Please type your question instead.'
        }
    
    def synthesize_speech(self, text: str, voice_id: str = None) -> dict:
        """
        Convert text to speech using ElevenLabs
        
        Args:
            text: Text to convert
            voice_id: ElevenLabs voice ID (optional)
        
        Returns:
            Dictionary with audio data
        """
        try:
            if not self.elevenlabs_client:
                return {
                    'success': False,
                    'audio': None,
                    'error': 'ElevenLabs API key not configured'
                }
            
            # Default voice (Rachel - clear female voice)
            if not voice_id:
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
            
            # Generate speech using text_to_speech
            audio = self.elevenlabs_client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            
            # Convert generator to bytes
            audio_bytes = b"".join(audio)
            
            logger.info(f"✅ Generated {len(audio_bytes)} bytes of audio")
            
            return {
                'success': True,
                'audio': audio_bytes,
                'format': 'mp3'
            }
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return {
                'success': False,
                'audio': None,
                'error': str(e)
            }
    
    def get_available_voices(self) -> list:
        """Get list of available ElevenLabs voices"""
        try:
            if not self.elevenlabs_client:
                return self._get_default_voices()
            
            # Try to fetch voices from API
            voices = self.elevenlabs_client.voices.get_all()
            
            return [
                {
                    'id': voice.voice_id,
                    'name': voice.name,
                    'category': voice.category if hasattr(voice, 'category') else 'premade'
                }
                for voice in voices.voices
            ]
            
        except Exception as e:
            logger.warning(f"Could not fetch voices from API: {e}")
            # Return default list as fallback
            return self._get_default_voices()
    
    def _get_default_voices(self) -> list:
        """Get default voice list (fallback)"""
        return [
            {'id': '21m00Tcm4TlvDq8ikWAM', 'name': 'Rachel', 'category': 'premade'},
            {'id': 'AZnzlk1XvdvUeBnXmlld', 'name': 'Domi', 'category': 'premade'},
            {'id': 'EXAVITQu4vr4xnSDxMaL', 'name': 'Bella', 'category': 'premade'},
            {'id': 'ErXwobaYiN019PkySvjV', 'name': 'Antoni', 'category': 'premade'},
            {'id': 'MF3mGyEYCl7XYWbV9V6O', 'name': 'Elli', 'category': 'premade'},
            {'id': 'TxGEqnHWrfWFTfGW9XjX', 'name': 'Josh', 'category': 'premade'},
            {'id': 'VR6AewLTigWG4xSOukaG', 'name': 'Arnold', 'category': 'premade'},
            {'id': 'pNInz6obpgDQGcFmaJgB', 'name': 'Adam', 'category': 'premade'},
            {'id': 'yoZ06aMxZJJ28mfd3POQ', 'name': 'Sam', 'category': 'premade'}
        ]


# Global instance
_voice_service = None

def get_voice_service():
    """Get or create voice service instance"""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service
