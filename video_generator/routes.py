"""
Flask Routes for Video Generator
API endpoints for video generation feature
"""

from flask import Blueprint, request, jsonify, render_template, send_file
from pathlib import Path
import os

# Create blueprint
video_bp = Blueprint('video', __name__, url_prefix='/video')

# Video service instance (initialized lazily)
_video_service = None

def get_video_service():
    """Get or create video service instance"""
    global _video_service
    if _video_service is None:
        from .video_service import VideoGeneratorService
        _video_service = VideoGeneratorService()
    return _video_service


@video_bp.route('/')
def video_generator_page():
    """Render the video generator page"""
    # Use the main templates folder video_generator.html which matches VidyaTid styling
    return render_template('video_generator.html')


@video_bp.route('/api/generate', methods=['POST'])
def generate_video():
    """
    Generate a new explainer video.
    
    Request JSON:
    {
        "topic": "DNA Replication",
        "subject": "Biology",
        "duration": 120  // optional, in seconds
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'topic' not in data:
            return jsonify({
                "success": False,
                "error": "Topic is required"
            }), 400
        
        topic = data.get('topic')
        subject = data.get('subject')
        duration = data.get('duration')
        
        service = get_video_service()
        result = service.generate_video(
            topic=topic,
            subject=subject,
            target_duration=duration
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@video_bp.route('/api/preview', methods=['POST'])
def preview_script():
    """
    Preview the script without generating video.
    
    Request JSON:
    {
        "topic": "Photosynthesis",
        "subject": "Biology"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'topic' not in data:
            return jsonify({
                "success": False,
                "error": "Topic is required"
            }), 400
        
        service = get_video_service()
        result = service.get_script_preview(
            topic=data.get('topic'),
            subject=data.get('subject')
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@video_bp.route('/api/estimate', methods=['POST'])
def estimate_duration():
    """
    Estimate video duration for a topic.
    
    Request JSON:
    {
        "topic": "Newton's Laws"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'topic' not in data:
            return jsonify({
                "success": False,
                "error": "Topic is required"
            }), 400
        
        service = get_video_service()
        result = service.estimate_duration(data.get('topic'))
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@video_bp.route('/api/videos', methods=['GET'])
def list_videos():
    """List all generated videos"""
    try:
        service = get_video_service()
        videos = service.list_generated_videos()
        
        return jsonify({
            "success": True,
            "videos": videos,
            "count": len(videos)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@video_bp.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Get available subjects"""
    try:
        service = get_video_service()
        subjects = service.get_available_subjects()
        
        return jsonify({
            "success": True,
            "subjects": subjects
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@video_bp.route('/api/topics', methods=['GET'])
def get_sample_topics():
    """Get sample topics for a subject"""
    try:
        subject = request.args.get('subject')
        
        service = get_video_service()
        topics = service.get_sample_topics(subject)
        
        return jsonify({
            "success": True,
            "topics": topics
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@video_bp.route('/api/download/<filename>')
def download_video(filename):
    """Download a generated video"""
    try:
        service = get_video_service()
        video_path = service.output_dir / filename
        
        if not video_path.exists():
            return jsonify({
                "success": False,
                "error": "Video not found"
            }), 404
        
        return send_file(
            video_path,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@video_bp.route('/api/status', methods=['GET'])
def get_status():
    """Get video generator service status"""
    try:
        service = get_video_service()
        
        # Check dependencies
        try:
            from PIL import Image
            pil_available = True
        except:
            pil_available = False
        
        try:
            from moviepy import VideoClip
            moviepy_available = True
        except:
            try:
                from moviepy.editor import VideoClip
                moviepy_available = True
            except:
                moviepy_available = False
        
        return jsonify({
            "success": True,
            "status": "ready",
            "dependencies": {
                "PIL": pil_available,
                "MoviePy": moviepy_available,
                "Gemini_API": service.script_generator.gemini_model is not None,
                "OpenAI_API": service.script_generator.openai_client is not None
            },
            "output_directory": str(service.output_dir),
            "videos_generated": len(service.history)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
