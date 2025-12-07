"""
Animation Engine for 2D Explainer Videos
Generates animated video frames and compiles them into videos

Uses MoviePy and PIL for video generation
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Try to import video/image libraries
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

try:
    # MoviePy 2.x imports
    from moviepy import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips, ColorClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        # MoviePy 1.x imports (fallback)
        from moviepy.editor import (
            ImageClip, TextClip, CompositeVideoClip, 
            concatenate_videoclips, ColorClip
        )
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False
        print("Warning: MoviePy not available. Install with: pip install moviepy")


class AnimationEngine:
    """
    Creates 2D animated explainer videos from scripts.
    Generates frames, applies animations, and compiles final video.
    """
    
    # Color schemes for different subjects
    COLOR_SCHEMES = {
        "physics": {
            "primary": "#3498db",
            "secondary": "#2980b9",
            "accent": "#e74c3c",
            "background": "#1a1a2e",
            "text": "#ffffff"
        },
        "chemistry": {
            "primary": "#27ae60",
            "secondary": "#229954",
            "accent": "#f39c12",
            "background": "#0d1117",
            "text": "#ffffff"
        },
        "biology": {
            "primary": "#2ecc71",
            "secondary": "#27ae60",
            "accent": "#e74c3c",
            "background": "#1a2f1a",
            "text": "#ffffff"
        },
        "maths": {
            "primary": "#9b59b6",
            "secondary": "#8e44ad",
            "accent": "#3498db",
            "background": "#1a1a2e",
            "text": "#ffffff"
        },
        "default": {
            "primary": "#667eea",
            "secondary": "#764ba2",
            "accent": "#f093fb",
            "background": "#0a0e27",
            "text": "#ffffff"
        }
    }
    
    # Video settings
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    FPS = 30
    
    def __init__(self, output_dir: str = "generated_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Font settings
        self.title_font_size = 72
        self.body_font_size = 48
        self.caption_font_size = 36
        
        # Try to load fonts
        self.fonts = self._load_fonts()
        
    def _load_fonts(self) -> dict:
        """Load available fonts"""
        fonts = {}
        
        if not PIL_AVAILABLE:
            return fonts
        
        # Try common font paths
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
            "/System/Library/Fonts/SFNSDisplay.ttf",  # macOS SF
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    fonts['title'] = ImageFont.truetype(path, self.title_font_size)
                    fonts['body'] = ImageFont.truetype(path, self.body_font_size)
                    fonts['caption'] = ImageFont.truetype(path, self.caption_font_size)
                    break
                except:
                    continue
        
        # Fallback to default font
        if not fonts:
            fonts['title'] = ImageFont.load_default()
            fonts['body'] = ImageFont.load_default()
            fonts['caption'] = ImageFont.load_default()
        
        return fonts
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def get_color_scheme(self, subject: str) -> dict:
        """Get color scheme for subject"""
        subject_lower = (subject or "").lower()
        for key in self.COLOR_SCHEMES:
            if key in subject_lower:
                return self.COLOR_SCHEMES[key]
        return self.COLOR_SCHEMES["default"]
    
    def create_gradient_background(self, width: int, height: int, 
                                   color1: str, color2: str) -> Image.Image:
        """Create a gradient background image"""
        if not PIL_AVAILABLE:
            return None
        
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        r1, g1, b1 = self.hex_to_rgb(color1)
        r2, g2, b2 = self.hex_to_rgb(color2)
        
        for y in range(height):
            ratio = y / height
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        return img

    
    def create_scene_frame(self, scene: dict, colors: dict, frame_type: str = "main") -> Image.Image:
        """
        Create a single frame for a scene.
        
        Args:
            scene: Scene data from script
            colors: Color scheme
            frame_type: 'intro', 'main', 'outro'
        """
        if not PIL_AVAILABLE:
            return None
        
        # Create gradient background
        bg_color = scene.get("background_color", colors["background"])
        img = self.create_gradient_background(
            self.VIDEO_WIDTH, self.VIDEO_HEIGHT,
            bg_color, colors["secondary"]
        )
        
        draw = ImageDraw.Draw(img)
        
        # Add decorative elements
        self._add_decorative_elements(draw, colors)
        
        # Add scene title
        title = scene.get("scene_title", "")
        if title:
            self._draw_text_with_shadow(
                draw, title,
                (self.VIDEO_WIDTH // 2, 100),
                self.fonts.get('title'),
                colors["text"],
                anchor="mm"
            )
        
        # Add key points
        key_points = scene.get("key_points", [])
        y_offset = 300
        for i, point in enumerate(key_points[:5]):
            bullet = f"• {point}"
            self._draw_text_with_shadow(
                draw, bullet,
                (200, y_offset + i * 80),
                self.fonts.get('body'),
                colors["text"],
                anchor="lm"
            )
        
        # Add visual elements indicator
        visual_elements = scene.get("visual_elements", [])
        if visual_elements:
            elements_text = "Visual: " + ", ".join(visual_elements[:3])
            self._draw_text_with_shadow(
                draw, elements_text,
                (self.VIDEO_WIDTH // 2, self.VIDEO_HEIGHT - 100),
                self.fonts.get('caption'),
                colors["accent"],
                anchor="mm"
            )
        
        return img
    
    def _add_decorative_elements(self, draw: ImageDraw, colors: dict):
        """Add decorative shapes to frame"""
        # Add corner accents
        accent_color = self.hex_to_rgb(colors["accent"])
        
        # Top-left corner
        draw.polygon([(0, 0), (150, 0), (0, 150)], fill=(*accent_color, 100))
        
        # Bottom-right corner
        draw.polygon([
            (self.VIDEO_WIDTH, self.VIDEO_HEIGHT),
            (self.VIDEO_WIDTH - 150, self.VIDEO_HEIGHT),
            (self.VIDEO_WIDTH, self.VIDEO_HEIGHT - 150)
        ], fill=(*accent_color, 100))
        
        # Add subtle grid lines
        grid_color = (*self.hex_to_rgb(colors["primary"]), 30)
        for x in range(0, self.VIDEO_WIDTH, 100):
            draw.line([(x, 0), (x, self.VIDEO_HEIGHT)], fill=grid_color, width=1)
        for y in range(0, self.VIDEO_HEIGHT, 100):
            draw.line([(0, y), (self.VIDEO_WIDTH, y)], fill=grid_color, width=1)
    
    def _draw_text_with_shadow(self, draw: ImageDraw, text: str, 
                               position: Tuple[int, int], font,
                               color: str, anchor: str = "mm"):
        """Draw text with shadow effect"""
        x, y = position
        shadow_color = (0, 0, 0, 128)
        text_color = self.hex_to_rgb(color) if isinstance(color, str) else color
        
        # Draw shadow
        draw.text((x + 3, y + 3), text, font=font, fill=shadow_color, anchor=anchor)
        # Draw main text
        draw.text((x, y), text, font=font, fill=text_color, anchor=anchor)
    
    def generate_scene_frames(self, scene: dict, colors: dict) -> List[Image.Image]:
        """Generate all frames for a scene with animations"""
        frames = []
        duration = scene.get("duration_seconds", 10)
        total_frames = duration * self.FPS
        animation_type = scene.get("animation_type", "fade_in")
        
        # Generate base frame
        base_frame = self.create_scene_frame(scene, colors)
        if not base_frame:
            return frames
        
        # Apply animation based on type
        if animation_type == "fade_in":
            frames = self._apply_fade_in(base_frame, total_frames)
        elif animation_type == "slide":
            frames = self._apply_slide(base_frame, total_frames)
        elif animation_type == "zoom":
            frames = self._apply_zoom(base_frame, total_frames)
        else:
            # Static frames
            frames = [base_frame] * total_frames
        
        return frames
    
    def _apply_fade_in(self, frame: Image.Image, total_frames: int) -> List[Image.Image]:
        """Apply fade-in animation"""
        frames = []
        fade_frames = min(30, total_frames // 4)  # Fade in over first quarter
        
        for i in range(total_frames):
            if i < fade_frames:
                # Fade in
                alpha = int(255 * (i / fade_frames))
                faded = frame.copy()
                faded.putalpha(alpha)
                # Composite with black background
                bg = Image.new('RGBA', frame.size, (0, 0, 0, 255))
                bg.paste(faded, mask=faded.split()[-1] if faded.mode == 'RGBA' else None)
                frames.append(bg.convert('RGB'))
            else:
                frames.append(frame)
        
        return frames
    
    def _apply_slide(self, frame: Image.Image, total_frames: int) -> List[Image.Image]:
        """Apply slide-in animation"""
        frames = []
        slide_frames = min(30, total_frames // 4)
        
        for i in range(total_frames):
            if i < slide_frames:
                # Slide from left
                offset = int(self.VIDEO_WIDTH * (1 - i / slide_frames))
                canvas = Image.new('RGB', frame.size, (10, 14, 39))
                canvas.paste(frame, (-offset, 0))
                frames.append(canvas)
            else:
                frames.append(frame)
        
        return frames
    
    def _apply_zoom(self, frame: Image.Image, total_frames: int) -> List[Image.Image]:
        """Apply zoom-in animation"""
        frames = []
        zoom_frames = min(30, total_frames // 4)
        
        for i in range(total_frames):
            if i < zoom_frames:
                # Zoom from center
                scale = 0.5 + 0.5 * (i / zoom_frames)
                new_size = (int(frame.width * scale), int(frame.height * scale))
                zoomed = frame.resize(new_size, Image.Resampling.LANCZOS)
                
                canvas = Image.new('RGB', frame.size, (10, 14, 39))
                paste_x = (frame.width - zoomed.width) // 2
                paste_y = (frame.height - zoomed.height) // 2
                canvas.paste(zoomed, (paste_x, paste_y))
                frames.append(canvas)
            else:
                frames.append(frame)
        
        return frames

    
    def compile_video(self, script: dict, output_filename: str = None) -> str:
        """
        Compile all scenes into a final video.
        
        Args:
            script: Complete video script with scenes
            output_filename: Output filename (optional)
        
        Returns:
            Path to generated video file
        """
        if not MOVIEPY_AVAILABLE:
            return self._compile_video_fallback(script, output_filename)
        
        subject = script.get("subject", "default")
        colors = self.get_color_scheme(subject)
        
        clips = []
        
        for scene in script.get("scenes", []):
            try:
                # Create scene clip
                scene_clip = self._create_scene_clip(scene, colors)
                if scene_clip:
                    clips.append(scene_clip)
            except Exception as e:
                print(f"Error creating scene {scene.get('scene_number')}: {e}")
                continue
        
        if not clips:
            return None
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Generate output filename
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = script.get("topic", "video").replace(" ", "_")[:30]
            output_filename = f"{topic_slug}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Write video file
        final_video.write_videofile(
            str(output_path),
            fps=self.FPS,
            codec='libx264',
            audio=False,  # No audio for now
            preset='medium',
            threads=4
        )
        
        # Cleanup
        final_video.close()
        for clip in clips:
            clip.close()
        
        return str(output_path)
    
    def _create_scene_clip(self, scene: dict, colors: dict):
        """Create a MoviePy clip for a scene"""
        duration = scene.get("duration_seconds", 10)
        
        # Create base frame
        frame = self.create_scene_frame(scene, colors)
        if not frame:
            # Create a simple color clip as fallback
            bg_color = self.hex_to_rgb(colors["background"])
            return ColorClip(
                size=(self.VIDEO_WIDTH, self.VIDEO_HEIGHT),
                color=bg_color,
                duration=duration
            )
        
        # Convert PIL image to numpy array for MoviePy
        import numpy as np
        frame_array = np.array(frame)
        
        # Create image clip
        clip = ImageClip(frame_array, duration=duration)
        
        # Apply animation effects (MoviePy 2.x compatible)
        animation_type = scene.get("animation_type", "fade_in")
        
        try:
            if animation_type == "fade_in":
                # Try MoviePy 2.x method first
                if hasattr(clip, 'with_effects'):
                    from moviepy.video.fx import CrossFadeIn
                    clip = clip.with_effects([CrossFadeIn(1.0)])
                elif hasattr(clip, 'crossfadein'):
                    clip = clip.crossfadein(1.0)
            elif animation_type == "zoom":
                # Skip zoom for now - static is fine
                pass
        except Exception as e:
            # If animation fails, just use static clip
            print(f"Animation effect skipped: {e}")
        
        return clip
    
    def _compile_video_fallback(self, script: dict, output_filename: str) -> str:
        """
        Fallback video compilation when MoviePy is not available.
        Generates individual frame images instead.
        """
        if not PIL_AVAILABLE:
            return None
        
        subject = script.get("subject", "default")
        colors = self.get_color_scheme(subject)
        
        # Create frames directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = script.get("topic", "video").replace(" ", "_")[:30]
        frames_dir = self.output_dir / f"{topic_slug}_{timestamp}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        
        for scene in script.get("scenes", []):
            # Create scene frame
            frame = self.create_scene_frame(scene, colors)
            if frame:
                # Save frame
                frame_path = frames_dir / f"frame_{frame_count:04d}.png"
                frame.save(frame_path)
                frame_count += 1
        
        # Create a summary JSON
        summary = {
            "topic": script.get("topic"),
            "total_frames": frame_count,
            "frames_directory": str(frames_dir),
            "note": "MoviePy not available. Frames saved as images. Use ffmpeg to compile: ffmpeg -framerate 1 -i frame_%04d.png -c:v libx264 output.mp4"
        }
        
        summary_path = frames_dir / "video_info.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(frames_dir)
    
    def generate_thumbnail(self, script: dict) -> str:
        """Generate a thumbnail image for the video"""
        if not PIL_AVAILABLE:
            return None
        
        subject = script.get("subject", "default")
        colors = self.get_color_scheme(subject)
        
        # Create thumbnail (16:9 aspect ratio)
        thumb_width = 1280
        thumb_height = 720
        
        img = self.create_gradient_background(
            thumb_width, thumb_height,
            colors["background"], colors["secondary"]
        )
        
        draw = ImageDraw.Draw(img)
        
        # Add title
        title = script.get("title", script.get("topic", "Educational Video"))
        self._draw_text_with_shadow(
            draw, title,
            (thumb_width // 2, thumb_height // 2 - 50),
            self.fonts.get('title'),
            colors["text"],
            anchor="mm"
        )
        
        # Add subject badge
        subject_text = script.get("subject", "NCERT")
        self._draw_text_with_shadow(
            draw, subject_text,
            (thumb_width // 2, thumb_height // 2 + 50),
            self.fonts.get('body'),
            colors["accent"],
            anchor="mm"
        )
        
        # Add play button overlay
        center_x, center_y = thumb_width // 2, thumb_height // 2 + 150
        play_size = 60
        draw.polygon([
            (center_x - play_size//2, center_y - play_size//2),
            (center_x - play_size//2, center_y + play_size//2),
            (center_x + play_size//2, center_y)
        ], fill=self.hex_to_rgb(colors["accent"]))
        
        # Save thumbnail
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = script.get("topic", "video").replace(" ", "_")[:30]
        thumb_path = self.output_dir / f"{topic_slug}_{timestamp}_thumb.png"
        img.save(thumb_path)
        
        return str(thumb_path)
