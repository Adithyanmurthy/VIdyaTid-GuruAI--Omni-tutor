"""
Video Generator Service
Main service for generating YouTube-style 2D animated explainer videos

Orchestrates:
- AI-powered script generation
- Topic-specific visualizations
- Character animation
- Audio narration (ElevenLabs)
- Video compilation
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from .script_generator import ScriptGenerator
from .explainer_engine import ExplainerVideoEngine


class VideoGeneratorService:
    """
    Main service for generating educational explainer videos.
    
    Creates YouTube-style videos with:
    - Large animated presenter character
    - Topic-specific visualizations (circuits, diagrams, etc.)
    - Professional audio narration
    - Smooth transitions and animations
    """
    
    def __init__(self, output_dir: str = "generated_videos"):
        self.script_generator = ScriptGenerator()
        self.explainer_engine = ExplainerVideoEngine(output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation history
        self.history_file = self.output_dir / "generation_history.json"
        self.history = self._load_history()
        
        print("✓ Video Generator Service ready")
    
    def _load_history(self) -> list:
        """Load generation history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_history(self):
        """Save generation history."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history[-50:], f, indent=2)
        except:
            pass

    def generate_video(self, topic: str, subject: str = None,
                      target_duration: int = None,
                      generate_thumbnail: bool = True) -> Dict[str, Any]:
        """
        Generate a complete YouTube-style explainer video.
        
        Args:
            topic: The topic to explain (e.g., "Electric Current", "DNA Replication")
            subject: Subject area (Physics, Chemistry, Biology, Maths)
            target_duration: Target video duration in seconds
            generate_thumbnail: Whether to generate a thumbnail
        
        Returns:
            dict with video path, script, thumbnail, and metadata
        """
        result = {
            "success": False,
            "topic": topic,
            "subject": subject,
            "timestamp": datetime.now().isoformat(),
            "video_path": None,
            "thumbnail_path": None,
            "script": None,
            "duration_seconds": 0,
            "error": None
        }
        
        try:
            print(f"\n{'='*60}")
            print(f"🎬 Generating Explainer Video")
            print(f"   Topic: {topic}")
            print(f"   Subject: {subject or 'Auto-detect'}")
            print(f"{'='*60}")
            
            # Step 1: Generate script
            print("\n📝 Step 1: Generating script...")
            script = self.script_generator.generate_script(
                topic=topic,
                subject=subject,
                target_duration=target_duration
            )
            
            if not script:
                result["error"] = "Failed to generate script"
                return result
            
            result["script"] = script
            result["subject"] = script.get("subject", subject)
            
            print(f"   ✓ Script: {script.get('total_scenes', 0)} scenes")
            print(f"   ✓ Duration: {script.get('duration_info', {}).get('formatted', 'N/A')}")
            
            # Step 2: Generate thumbnail
            if generate_thumbnail:
                print("\n🖼️  Step 2: Generating thumbnail...")
                thumbnail_path = self._generate_thumbnail(script)
                if thumbnail_path:
                    result["thumbnail_path"] = thumbnail_path
                    print(f"   ✓ Thumbnail saved")
            
            # Step 3: Compile video
            print("\n🎥 Step 3: Creating video...")
            video_path = self.explainer_engine.compile_explainer_video(script)
            
            if video_path:
                result["video_path"] = video_path
                result["success"] = True
                result["duration_seconds"] = script.get("duration_info", {}).get("total_seconds", 0)
                print(f"\n✅ Video generated successfully!")
            else:
                result["error"] = "Failed to compile video"
            
            # Save to history
            self.history.append({
                "topic": topic,
                "subject": result["subject"],
                "timestamp": result["timestamp"],
                "video_path": result["video_path"],
                "success": result["success"]
            })
            self._save_history()
            
        except Exception as e:
            result["error"] = str(e)
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        return result

    def get_script_preview(self, topic: str, subject: str = None) -> Dict[str, Any]:
        """Generate just the script without creating the video."""
        try:
            script = self.script_generator.generate_script(
                topic=topic,
                subject=subject
            )
            
            return {
                "success": True,
                "script": script,
                "estimated_duration": script.get("duration_info", {}),
                "scenes_count": script.get("total_scenes", 0)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def estimate_duration(self, topic: str) -> Dict[str, Any]:
        """Estimate video duration for a topic."""
        ncert_content = self.script_generator.get_ncert_content(topic)
        duration_info = self.script_generator.estimate_video_duration(
            topic, len(ncert_content)
        )
        
        return {
            "topic": topic,
            "content_found": len(ncert_content) > 100,
            "duration": duration_info
        }
    
    def list_generated_videos(self) -> list:
        """List all generated videos."""
        videos = []
        
        for ext in ['*.mp4', '*.avi', '*.mov']:
            for video_file in self.output_dir.glob(ext):
                videos.append({
                    "filename": video_file.name,
                    "path": str(video_file),
                    "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(
                        video_file.stat().st_ctime
                    ).isoformat()
                })
        
        return sorted(videos, key=lambda x: x["created"], reverse=True)
    
    def get_available_subjects(self) -> list:
        """Get list of available subjects."""
        return [
            {"id": "physics", "name": "Physics", "icon": "⚛️"},
            {"id": "chemistry", "name": "Chemistry", "icon": "⚗️"},
            {"id": "biology", "name": "Biology", "icon": "🧬"},
            {"id": "maths", "name": "Mathematics", "icon": "📐"}
        ]
    
    def get_sample_topics(self, subject: str = None) -> list:
        """Get sample topics for video generation."""
        topics = {
            "physics": [
                "Electric Current",
                "Newton's Laws of Motion",
                "Electromagnetic Induction",
                "Wave Optics",
                "Thermodynamics"
            ],
            "chemistry": [
                "Chemical Bonding",
                "Periodic Table",
                "Organic Chemistry Basics",
                "Electrochemistry",
                "Chemical Equilibrium"
            ],
            "biology": [
                "Cell Structure",
                "DNA Replication",
                "Photosynthesis",
                "Human Digestive System",
                "Genetics and Heredity"
            ],
            "maths": [
                "Quadratic Equations",
                "Trigonometry",
                "Calculus Basics",
                "Probability",
                "Matrices"
            ]
        }
        
        if subject and subject.lower() in topics:
            return topics[subject.lower()]
        
        all_topics = []
        for subj, topic_list in topics.items():
            for topic in topic_list:
                all_topics.append({"topic": topic, "subject": subj})
        return all_topics

    def _generate_thumbnail(self, script: dict) -> Optional[str]:
        """Generate a thumbnail for the video."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            colors = self.explainer_engine.get_colors(script.get("subject", "default"))
            
            # Create thumbnail (1280x720)
            thumb = Image.new('RGB', (1280, 720))
            draw = ImageDraw.Draw(thumb)
            
            # Background gradient
            c1, c2 = colors["bg_gradient"]
            for y in range(720):
                ratio = y / 720
                r = int(c1[0] + (c2[0] - c1[0]) * ratio)
                g = int(c1[1] + (c2[1] - c1[1]) * ratio)
                b = int(c1[2] + (c2[2] - c1[2]) * ratio)
                draw.line([(0, y), (1280, y)], fill=(r, g, b))
            
            # Load font
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]
            
            title_font = None
            subtitle_font = None
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        title_font = ImageFont.truetype(path, 72)
                        subtitle_font = ImageFont.truetype(path, 36)
                        break
                    except:
                        continue
            
            # Title
            title = script.get("title", script.get("topic", "Educational Video"))
            if len(title) > 40:
                title = title[:37] + "..."
            
            if title_font:
                bbox = draw.textbbox((0, 0), title, font=title_font)
                text_width = bbox[2] - bbox[0]
                x = 640 - text_width // 2
                # Shadow
                draw.text((x + 3, 253), title, fill=(0, 0, 0), font=title_font)
                draw.text((x, 250), title, fill=colors["text"], font=title_font)
            else:
                draw.text((640 - len(title) * 15, 250), title, fill=colors["text"])
            
            # Subject badge
            subject = script.get("subject", "NCERT")
            badge_text = f"📚 {subject}"
            if subtitle_font:
                bbox = draw.textbbox((0, 0), badge_text, font=subtitle_font)
                text_width = bbox[2] - bbox[0]
                badge_x = 640 - text_width // 2
                
                # Badge background
                draw.rounded_rectangle([badge_x - 20, 360, badge_x + text_width + 20, 420],
                                      radius=10, fill=colors["primary"])
                draw.text((badge_x, 370), badge_text, fill=colors["text"], font=subtitle_font)
            
            # Play button
            play_x, play_y = 640, 550
            # Circle
            draw.ellipse([play_x - 50, play_y - 50, play_x + 50, play_y + 50],
                        fill=colors["accent"])
            # Triangle
            draw.polygon([(play_x - 15, play_y - 25), 
                         (play_x - 15, play_y + 25), 
                         (play_x + 25, play_y)],
                        fill=(255, 255, 255))
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            thumb_path = self.output_dir / f"thumb_{timestamp}.png"
            thumb.save(thumb_path)
            
            return str(thumb_path)
            
        except Exception as e:
            print(f"Thumbnail generation failed: {e}")
            return None
