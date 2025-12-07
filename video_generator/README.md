# Video Generator Module

A portable, self-contained module for generating YouTube-style 2D animated explainer videos.

## Features

- 🎬 **Animated Character** - Large presenter with gestures, expressions, and lip sync
- 📊 **Topic Visualizations** - Auto-generates circuits, DNA helixes, atomic structures, etc.
- 🎙️ **Audio Narration** - ElevenLabs API (high quality) or gTTS fallback (free)
- 📝 **AI Script Generation** - Uses Gemini or OpenAI to create educational scripts
- 🎨 **Professional Styling** - Subject-specific color schemes and animations

## Quick Setup

### 1. Copy the folder
Copy the entire `video_generator` folder to your project.

### 2. Install dependencies
```bash
pip install -r video_generator/requirements.txt
```

### 3. Set environment variables
Add these to your `.env` file:
```
# Required for AI script generation (at least one)
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Optional - for high-quality voice (falls back to gTTS if not set)
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

### 4. Register the Flask blueprint
In your main Flask app:
```python
from video_generator.routes import video_bp

app = Flask(__name__)
app.register_blueprint(video_bp)
```

### 5. Copy the templates and static files
Copy these files to your project:
- `templates/video_generator.html` → your `templates/` folder
- `static/video_generator.css` → your `static/` folder
- `static/video_generator.js` → your `static/` folder

### 6. Access the video generator
Navigate to `/video` in your app to use the video generator.

## Usage (Programmatic)

```python
from video_generator import VideoGeneratorService

# Initialize
service = VideoGeneratorService(output_dir="generated_videos")

# Generate a video
result = service.generate_video(
    topic="Electric Current",
    subject="Physics",
    target_duration=120  # seconds (optional)
)

if result["success"]:
    print(f"Video saved: {result['video_path']}")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video/` | GET | Video generator UI |
| `/video/api/generate` | POST | Generate a video |
| `/video/api/preview` | POST | Preview script only |
| `/video/api/videos` | GET | List generated videos |
| `/video/api/download/<filename>` | GET | Download a video |
| `/video/api/status` | GET | Check service status |

## File Structure

```
video_generator/
├── __init__.py           # Module exports
├── video_service.py      # Main service (entry point)
├── script_generator.py   # AI-powered script generation
├── explainer_engine.py   # Video frame creation
├── character_animator.py # Character & diagram drawing
├── audio_service.py      # Text-to-speech
├── routes.py             # Flask API routes
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Configuration

The module auto-detects available services:
- If `GEMINI_API_KEY` is set → Uses Gemini for scripts
- If `OPENAI_API_KEY` is set → Uses OpenAI as fallback
- If `ELEVENLABS_API_KEY` is set → Uses ElevenLabs for voice
- Otherwise → Uses gTTS (free, lower quality)

## Output

Videos are saved to the `output_dir` (default: `generated_videos/`):
- `explainer_<topic>_<timestamp>.mp4` - The video file
- `thumb_<timestamp>.png` - Thumbnail image
- `audio/scene_XX.mp3` - Audio files (can be deleted after)
