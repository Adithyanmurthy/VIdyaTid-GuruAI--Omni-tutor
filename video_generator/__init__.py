"""
Video Generator Module
=======================
A portable, self-contained module for generating YouTube-style 2D animated explainer videos.

Features:
- Large animated presenter character with gestures and expressions
- Topic-specific visualizations (circuits, DNA, atoms, etc.)
- Professional audio narration (ElevenLabs API or gTTS fallback)
- AI-powered script generation (Gemini/OpenAI)
- Smooth transitions and animations

Quick Start:
-----------
1. Copy this folder to your project
2. pip install -r video_generator/requirements.txt
3. Set API keys in .env (GEMINI_API_KEY, ELEVENLABS_API_KEY)
4. Register blueprint: app.register_blueprint(video_bp)

Usage:
------
from video_generator import VideoGeneratorService

service = VideoGeneratorService()
result = service.generate_video(topic="Electric Current", subject="Physics")
"""

__version__ = "1.0.0"
__author__ = "Video Generator"

from .video_service import VideoGeneratorService
from .script_generator import ScriptGenerator
from .explainer_engine import ExplainerVideoEngine
from .character_animator import CharacterAnimator, DiagramGenerator, TopicVisualizer
from .audio_service import AudioService
from .routes import video_bp

__all__ = [
    'VideoGeneratorService', 
    'ScriptGenerator', 
    'ExplainerVideoEngine',
    'CharacterAnimator',
    'DiagramGenerator',
    'TopicVisualizer',
    'AudioService',
    'video_bp',  # Flask blueprint for easy integration
]


def setup_flask_app(app, url_prefix='/video'):
    """
    Helper function to set up the video generator with a Flask app.
    
    Usage:
        from video_generator import setup_flask_app
        setup_flask_app(app)
    
    Args:
        app: Flask application instance
        url_prefix: URL prefix for video routes (default: '/video')
    """
    from .routes import video_bp
    
    # Update prefix if different from default
    if url_prefix != '/video':
        video_bp.url_prefix = url_prefix
    
    app.register_blueprint(video_bp)
    print(f"✓ Video Generator registered at {url_prefix}/")
