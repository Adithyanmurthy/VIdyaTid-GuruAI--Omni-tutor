"""
Professional Explainer Video Engine
Creates YouTube-style educational videos with:
- Large animated presenter character
- Topic-specific visualizations (circuits, diagrams, etc.)
- Smooth transitions and animations
- Synchronized audio narration
"""

import os
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .character_animator import CharacterAnimator, DiagramGenerator, TopicVisualizer
from .audio_service import AudioService


class ExplainerVideoEngine:
    """
    Creates professional YouTube-style explainer videos.
    """
    
    # Video settings
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080
    FPS = 30
    
    # Character positions for different layouts
    CHARACTER_POSITIONS = {
        "left": (350, 950),      # Character on left side
        "right": (1570, 950),    # Character on right side
        "center": (960, 950),    # Character in center
        "corner_left": (250, 980),   # Small in corner
        "corner_right": (1670, 980), # Small in corner
    }
    
    # Content area positions
    CONTENT_POSITIONS = {
        "left": (1200, 500),     # Content on right when char is left
        "right": (720, 500),     # Content on left when char is right
        "center": (960, 400),    # Content above center character
        "full": (960, 540),      # Full screen content
    }

    # Color schemes by subject
    COLOR_SCHEMES = {
        "physics": {
            "bg_gradient": [(15, 20, 50), (30, 45, 90)],
            "primary": (52, 152, 219),
            "secondary": (41, 128, 185),
            "accent": (231, 76, 60),
            "text": (255, 255, 255),
            "highlight": (255, 200, 100)
        },
        "chemistry": {
            "bg_gradient": [(10, 30, 25), (25, 55, 45)],
            "primary": (39, 174, 96),
            "secondary": (34, 153, 84),
            "accent": (243, 156, 18),
            "text": (255, 255, 255),
            "highlight": (255, 220, 100)
        },
        "biology": {
            "bg_gradient": [(15, 35, 25), (30, 65, 45)],
            "primary": (46, 204, 113),
            "secondary": (39, 174, 96),
            "accent": (231, 76, 60),
            "text": (255, 255, 255),
            "highlight": (255, 180, 100)
        },
        "maths": {
            "bg_gradient": [(25, 18, 45), (50, 35, 80)],
            "primary": (155, 89, 182),
            "secondary": (142, 68, 173),
            "accent": (52, 152, 219),
            "text": (255, 255, 255),
            "highlight": (255, 200, 150)
        },
        "default": {
            "bg_gradient": [(12, 16, 42), (28, 35, 70)],
            "primary": (102, 126, 234),
            "secondary": (118, 75, 162),
            "accent": (240, 147, 251),
            "text": (255, 255, 255),
            "highlight": (255, 220, 150)
        }
    }
    
    def __init__(self, output_dir: str = "generated_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components with larger character
        self.character = CharacterAnimator(scale=1.8, style="professional")
        self.diagram_gen = DiagramGenerator()
        self.topic_visualizer = TopicVisualizer()
        self.audio_service = AudioService()
        
        # Load fonts
        self.fonts = self._load_fonts()
        
        print("✓ Explainer Video Engine initialized (YouTube-style)")
    
    def _load_fonts(self) -> dict:
        """Load fonts for text rendering"""
        fonts = {}
        
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    fonts['title'] = ImageFont.truetype(path, 64)
                    fonts['subtitle'] = ImageFont.truetype(path, 42)
                    fonts['body'] = ImageFont.truetype(path, 32)
                    fonts['caption'] = ImageFont.truetype(path, 26)
                    fonts['small'] = ImageFont.truetype(path, 20)
                    break
                except:
                    continue
        
        if not fonts:
            default = ImageFont.load_default()
            fonts = {k: default for k in ['title', 'subtitle', 'body', 'caption', 'small']}
        
        return fonts
    
    def get_colors(self, subject: str) -> dict:
        """Get color scheme for subject"""
        subject_lower = (subject or "").lower()
        for key in self.COLOR_SCHEMES:
            if key in subject_lower:
                return self.COLOR_SCHEMES[key]
        return self.COLOR_SCHEMES["default"]

    def create_background(self, colors: dict, frame: int = 0) -> Image.Image:
        """Create animated gradient background"""
        img = Image.new('RGB', (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
        draw = ImageDraw.Draw(img)
        
        c1, c2 = colors["bg_gradient"]
        
        # Animated gradient shift
        shift = int(math.sin(frame * 0.02) * 10)
        
        for y in range(self.VIDEO_HEIGHT):
            ratio = (y + shift) / self.VIDEO_HEIGHT
            ratio = max(0, min(1, ratio))
            r = int(c1[0] + (c2[0] - c1[0]) * ratio)
            g = int(c1[1] + (c2[1] - c1[1]) * ratio)
            b = int(c1[2] + (c2[2] - c1[2]) * ratio)
            draw.line([(0, y), (self.VIDEO_WIDTH, y)], fill=(r, g, b))
        
        # Subtle animated grid
        grid_alpha = 15 + int(math.sin(frame * 0.05) * 5)
        grid_color = (*colors["primary"][:3],)
        
        # Vertical lines
        for x in range(0, self.VIDEO_WIDTH, 100):
            x_offset = x + int(math.sin(frame * 0.01 + x * 0.01) * 2)
            draw.line([(x_offset, 0), (x_offset, self.VIDEO_HEIGHT)], 
                     fill=tuple(max(0, min(255, c + grid_alpha - 255)) for c in grid_color), 
                     width=1)
        
        # Horizontal lines
        for y in range(0, self.VIDEO_HEIGHT, 100):
            draw.line([(0, y), (self.VIDEO_WIDTH, y)], 
                     fill=tuple(max(0, min(255, c + grid_alpha - 255)) for c in grid_color), 
                     width=1)
        
        # Add subtle glow orbs
        self._add_background_orbs(draw, colors, frame)
        
        return img
    
    def _add_background_orbs(self, draw: ImageDraw, colors: dict, frame: int):
        """Add animated glowing orbs to background"""
        orb_positions = [
            (200, 200, 150),
            (1700, 300, 120),
            (400, 800, 100),
            (1500, 700, 130),
        ]
        
        for i, (ox, oy, size) in enumerate(orb_positions):
            # Animate position
            ox += int(math.sin(frame * 0.02 + i) * 30)
            oy += int(math.cos(frame * 0.015 + i) * 20)
            
            # Draw soft glow (multiple circles with decreasing opacity)
            for r in range(size, 0, -20):
                alpha = int(15 * (r / size))
                color = tuple(min(255, c + 30) for c in colors["primary"][:3])
                draw.ellipse([ox - r, oy - r, ox + r, oy + r], 
                            fill=color, outline=None)

    def create_intro_frame(self, script: dict, colors: dict, 
                          frame: int, total_frames: int) -> Image.Image:
        """Create animated introduction frame"""
        img = self.create_background(colors, frame)
        draw = ImageDraw.Draw(img)
        
        title = script.get("title", script.get("topic", "Educational Video"))
        subject = script.get("subject", "NCERT")
        
        # Animation progress (0 to 1)
        progress = min(1.0, frame / (total_frames * 0.7))
        
        # Title animation - slide in from top
        title_y = int(300 - (1 - progress) * 100)
        title_alpha = int(255 * progress)
        
        # Draw title with shadow
        if self.fonts.get('title'):
            bbox = draw.textbbox((0, 0), title, font=self.fonts['title'])
            text_width = bbox[2] - bbox[0]
            title_x = self.VIDEO_WIDTH // 2 - text_width // 2
            
            # Shadow
            draw.text((title_x + 3, title_y + 3), title, 
                     fill=(0, 0, 0), font=self.fonts['title'])
            # Main text
            draw.text((title_x, title_y), title, 
                     fill=colors["text"], font=self.fonts['title'])
        
        # Subject badge - fade in
        if progress > 0.3:
            badge_progress = min(1.0, (progress - 0.3) / 0.4)
            badge_y = 400 + int((1 - badge_progress) * 30)
            
            badge_text = f"📚 {subject}"
            if self.fonts.get('subtitle'):
                bbox = draw.textbbox((0, 0), badge_text, font=self.fonts['subtitle'])
                text_width = bbox[2] - bbox[0]
                badge_x = self.VIDEO_WIDTH // 2 - text_width // 2
                
                # Badge background
                draw.rounded_rectangle([badge_x - 20, badge_y - 10, 
                                       badge_x + text_width + 20, badge_y + 50],
                                      radius=10, fill=colors["primary"])
                draw.text((badge_x, badge_y), badge_text, 
                         fill=colors["text"], font=self.fonts['subtitle'])
        
        # Character entrance - slide in from bottom
        if progress > 0.4:
            char_progress = min(1.0, (progress - 0.4) / 0.5)
            char_y = int(self.VIDEO_HEIGHT + 200 - char_progress * 250)
            
            self.character.draw_character(
                draw, 
                (self.VIDEO_WIDTH // 2, char_y),
                gesture="waving",
                expression="happy",
                frame=frame,
                talking=False
            )
        
        # "Let's Learn!" text
        if progress > 0.7:
            learn_text = "Let's Learn!"
            if self.fonts.get('body'):
                bbox = draw.textbbox((0, 0), learn_text, font=self.fonts['body'])
                text_width = bbox[2] - bbox[0]
                draw.text((self.VIDEO_WIDTH // 2 - text_width // 2, 520),
                         learn_text, fill=colors["accent"], font=self.fonts['body'])
        
        return img

    def create_explanation_frame(self, scene: dict, script: dict,
                                colors: dict, frame: int,
                                scene_progress: float,
                                char_position: str = "left") -> Image.Image:
        """
        Create an explanation frame showing actual content being explained.
        Shows the narration text on screen in a readable format.
        """
        img = self.create_background(colors, frame)
        draw = ImageDraw.Draw(img)
        
        # Get scene data
        title = scene.get("scene_title", "")
        key_points = scene.get("key_points", [])
        narration = scene.get("narration", "")
        topic = script.get("topic", "")
        scene_num = scene.get("scene_number", 1)
        total_scenes = script.get("total_scenes", 4)
        
        # Determine if character should be talking
        talking = 0.1 < scene_progress < 0.9
        
        # Draw title bar
        self._draw_title_bar(draw, title, colors, frame)
        
        # Character position and gesture
        char_pos = self.CHARACTER_POSITIONS.get(char_position, self.CHARACTER_POSITIONS["left"])
        content_pos = self.CONTENT_POSITIONS.get(char_position, self.CONTENT_POSITIONS["left"])
        
        # Determine gesture
        gesture = self._determine_gesture(scene, frame, scene_progress)
        expression = "explaining" if talking else "smile"
        
        # Draw character
        self.character.draw_character(
            draw, char_pos,
            gesture=gesture,
            expression=expression,
            frame=frame,
            talking=talking
        )
        
        # Content display logic:
        # - Scene 1 (Intro): Show topic overview
        # - Scene 2: Show diagram/flowchart (only once per video)
        # - Other scenes: Show narration content as text cards
        
        topic_type = self.topic_visualizer.detect_topic_type(topic)
        
        # Show diagram only in scene 2 (middle of video) for visual variety
        show_diagram = (scene_num == 2 and topic_type != "general")
        
        if show_diagram:
            # Draw topic-specific visualization (circuit, DNA, etc.)
            self.topic_visualizer.draw_topic_visual(
                draw, topic,
                content_pos, (750, 550),
                frame, scene
            )
        else:
            # Show the actual narration content as readable text
            self._draw_content_display(draw, narration, title, colors, 
                                      content_pos, scene_progress, frame)
        
        # Draw caption/subtitle at bottom (shorter version of narration)
        if narration:
            self._draw_caption(draw, narration, colors, scene_progress)
        
        return img
    
    def _draw_content_display(self, draw: ImageDraw, narration: str, title: str,
                             colors: dict, position: tuple, progress: float, frame: int):
        """
        Draw the actual narration content on screen in a readable, creative format.
        This shows what the narrator is explaining.
        """
        x, y = position
        
        # Clean the narration text
        import re
        clean_text = narration
        clean_text = re.sub(r'Chapter\s+\w+', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\d+\.\d+\s*', '', clean_text)
        clean_text = re.sub(r'[-_]{2,}', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if not clean_text:
            return
        
        # Split into sentences for display
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            sentences = [clean_text[:200]]
        
        # Show sentences progressively based on progress
        num_to_show = max(1, int(len(sentences) * progress) + 1)
        sentences_to_show = sentences[:min(num_to_show, 4)]  # Max 4 sentences
        
        # Content box dimensions
        box_width = 750
        box_height = 450
        box_x = x - box_width // 2
        box_y = y - box_height // 2 + 50
        
        # Draw main content box with rounded corners
        draw.rounded_rectangle([box_x, box_y, box_x + box_width, box_y + box_height],
                              radius=20, fill=(30, 40, 70, 230), 
                              outline=colors["primary"], width=2)
        
        # Title inside the box
        title_y = box_y + 15
        if self.fonts.get('subtitle') and title:
            # Clean title
            clean_title = re.sub(r'[-_]{2,}', ' ', title).strip()
            bbox = draw.textbbox((0, 0), clean_title, font=self.fonts['subtitle'])
            tw = bbox[2] - bbox[0]
            draw.text((x - tw // 2, title_y), clean_title, 
                     fill=colors["accent"], font=self.fonts['subtitle'])
        
        # Divider line
        draw.line([(box_x + 30, title_y + 50), (box_x + box_width - 30, title_y + 50)],
                 fill=colors["primary"], width=2)
        
        # Draw each sentence as a content item
        content_y = title_y + 70
        item_height = 85
        
        for i, sentence in enumerate(sentences_to_show):
            # Highlight current sentence being spoken
            is_current = (i == len(sentences_to_show) - 1) and progress < 0.95
            
            # Item background
            item_color = (50, 65, 100) if is_current else (40, 50, 80)
            draw.rounded_rectangle([box_x + 20, content_y, 
                                   box_x + box_width - 20, content_y + item_height - 10],
                                  radius=12, fill=item_color)
            
            # Bullet/number
            bullet_color = colors["accent"] if is_current else colors["secondary"]
            bullet_x = box_x + 45
            bullet_y = content_y + (item_height - 10) // 2
            draw.ellipse([bullet_x - 12, bullet_y - 12, bullet_x + 12, bullet_y + 12],
                        fill=bullet_color)
            draw.text((bullet_x - 5, bullet_y - 9), str(i + 1), fill=(255, 255, 255))
            
            # Sentence text - wrap to fit
            text_x = box_x + 75
            max_text_width = box_width - 110
            wrapped = self._wrap_content_text(sentence, max_text_width, draw)
            
            # Center text vertically
            lines = wrapped.split('\n')
            line_height = 26
            text_start_y = content_y + (item_height - 10 - len(lines) * line_height) // 2
            
            for line in lines:
                if self.fonts.get('body'):
                    draw.text((text_x, text_start_y), line, 
                             fill=colors["text"], font=self.fonts['body'])
                else:
                    draw.text((text_x, text_start_y), line, fill=colors["text"])
                text_start_y += line_height
            
            content_y += item_height
    
    def _wrap_content_text(self, text: str, max_width: int, draw: ImageDraw) -> str:
        """Wrap text to fit within max_width."""
        if not text:
            return ""
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if self.fonts.get('body'):
                try:
                    bbox = draw.textbbox((0, 0), test_line, font=self.fonts['body'])
                    text_width = bbox[2] - bbox[0]
                except:
                    text_width = len(test_line) * 10
            else:
                text_width = len(test_line) * 10
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                if len(lines) >= 2:
                    break
        
        if current_line and len(lines) < 2:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines) if lines else text[:60]

    def _draw_title_bar(self, draw: ImageDraw, title: str, colors: dict, frame: int):
        """Draw animated title bar at top"""
        # Background bar with gradient
        bar_height = 90
        for y in range(bar_height):
            alpha = 1 - (y / bar_height) * 0.3
            color = tuple(int(c * alpha) for c in colors["primary"])
            draw.line([(0, y), (self.VIDEO_WIDTH, y)], fill=color)
        
        # Animated accent line
        line_width = int(self.VIDEO_WIDTH * min(1.0, frame / 30))
        draw.line([(0, bar_height - 3), (line_width, bar_height - 3)], 
                 fill=colors["accent"], width=3)
        
        # Title text
        if self.fonts.get('subtitle') and title:
            bbox = draw.textbbox((0, 0), title, font=self.fonts['subtitle'])
            text_width = bbox[2] - bbox[0]
            
            # Shadow
            draw.text((self.VIDEO_WIDTH // 2 - text_width // 2 + 2, 27),
                     title, fill=(0, 0, 0), font=self.fonts['subtitle'])
            # Main
            draw.text((self.VIDEO_WIDTH // 2 - text_width // 2, 25),
                     title, fill=colors["text"], font=self.fonts['subtitle'])
    
    def _draw_caption(self, draw: ImageDraw, text: str, colors: dict, progress: float):
        """Draw caption/subtitle at bottom with word highlighting"""
        # Background bar
        bar_y = self.VIDEO_HEIGHT - 130
        draw.rectangle([0, bar_y, self.VIDEO_WIDTH, self.VIDEO_HEIGHT],
                      fill=(0, 0, 0, 200))
        
        # Truncate text for display
        display_text = text[:180] + "..." if len(text) > 180 else text
        
        # Word wrap
        if self.fonts.get('body'):
            words = display_text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=self.fonts['body'])
                if bbox[2] - bbox[0] < self.VIDEO_WIDTH - 100:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw lines (max 2)
            y = bar_y + 20
            for line in lines[:2]:
                bbox = draw.textbbox((0, 0), line, font=self.fonts['body'])
                text_width = bbox[2] - bbox[0]
                x = self.VIDEO_WIDTH // 2 - text_width // 2
                
                # Shadow
                draw.text((x + 2, y + 2), line, fill=(0, 0, 0), font=self.fonts['body'])
                # Main text
                draw.text((x, y), line, fill=colors["text"], font=self.fonts['body'])
                y += 40
    
    def _determine_gesture(self, scene: dict, frame: int, progress: float) -> str:
        """Determine appropriate gesture for the scene"""
        title = scene.get("scene_title", "").lower()
        
        # Cycle through gestures for variety
        gesture_cycle = ["explaining", "pointing_right", "presenting", "pointing_up"]
        cycle_index = (frame // 90) % len(gesture_cycle)
        
        if "introduction" in title or "intro" in title:
            return "waving" if progress < 0.3 else "presenting"
        elif "summary" in title or "conclusion" in title:
            return "presenting"
        elif "step" in title or "process" in title:
            return "pointing_right"
        elif "important" in title or "key" in title:
            return "pointing_up"
        else:
            return gesture_cycle[cycle_index]
    
    def _determine_scene_type(self, scene: dict) -> str:
        """Determine visualization type for scene"""
        title = scene.get("scene_title", "").lower()
        narration = scene.get("narration", "").lower()
        key_points = scene.get("key_points", [])
        
        if any(kw in narration for kw in ["step", "first", "then", "finally", "process"]):
            return "process" if len(key_points) <= 5 else "flowchart"
        elif any(kw in narration for kw in ["flow", "sequence", "leads to"]):
            return "flowchart"
        elif "summary" in title or "conclusion" in title:
            return "summary"
        
        return "concept"

    def create_summary_frame(self, script: dict, colors: dict, 
                            frame: int, total_frames: int) -> Image.Image:
        """Create summary/outro frame"""
        img = self.create_background(colors, frame)
        draw = ImageDraw.Draw(img)
        
        progress = min(1.0, frame / (total_frames * 0.8))
        
        # Title
        self._draw_title_bar(draw, "📝 Summary", colors, frame)
        
        # Character in presenting pose
        self.character.draw_character(
            draw, (350, 900),
            gesture="presenting",
            expression="happy",
            frame=frame,
            talking=False
        )
        
        # Summary points
        learning_objectives = script.get("learning_objectives", [])
        exam_tips = script.get("exam_tips", [])
        
        # Draw summary box
        summary_points = learning_objectives[:3] + exam_tips[:2]
        if summary_points:
            self.diagram_gen.draw_concept_box(
                draw, "Key Takeaways", summary_points,
                (1200, 200),
                width=650,
                font=self.fonts.get('body'),
                title_font=self.fonts.get('subtitle')
            )
        
        # "Thanks for watching" text
        if progress > 0.6:
            thanks_text = "Thanks for Learning! 🎓"
            if self.fonts.get('subtitle'):
                bbox = draw.textbbox((0, 0), thanks_text, font=self.fonts['subtitle'])
                text_width = bbox[2] - bbox[0]
                draw.text((self.VIDEO_WIDTH // 2 - text_width // 2, 
                          self.VIDEO_HEIGHT - 180),
                         thanks_text, fill=colors["accent"], font=self.fonts['subtitle'])
        
        return img
    
    def create_transition_frame(self, from_scene: dict, to_scene: dict,
                               colors: dict, progress: float, frame: int) -> Image.Image:
        """Create smooth transition between scenes"""
        img = self.create_background(colors, frame)
        draw = ImageDraw.Draw(img)
        
        # Fade effect
        if progress < 0.5:
            # Fade out old scene title
            alpha = int(255 * (1 - progress * 2))
            old_title = from_scene.get("scene_title", "")
            if self.fonts.get('title') and old_title:
                bbox = draw.textbbox((0, 0), old_title, font=self.fonts['title'])
                text_width = bbox[2] - bbox[0]
                draw.text((self.VIDEO_WIDTH // 2 - text_width // 2, 
                          self.VIDEO_HEIGHT // 2 - 50),
                         old_title, fill=(*colors["text"][:3],), 
                         font=self.fonts['title'])
        else:
            # Fade in new scene title
            alpha = int(255 * ((progress - 0.5) * 2))
            new_title = to_scene.get("scene_title", "")
            if self.fonts.get('title') and new_title:
                bbox = draw.textbbox((0, 0), new_title, font=self.fonts['title'])
                text_width = bbox[2] - bbox[0]
                y_offset = int((1 - (progress - 0.5) * 2) * 50)
                draw.text((self.VIDEO_WIDTH // 2 - text_width // 2, 
                          self.VIDEO_HEIGHT // 2 - 50 + y_offset),
                         new_title, fill=colors["text"], 
                         font=self.fonts['title'])
        
        # Character walking animation
        char_x = int(400 + progress * 300)
        self.character.draw_character(
            draw, (char_x, 900),
            gesture="neutral",
            expression="smile",
            frame=frame
        )
        
        return img

    def compile_explainer_video(self, script: dict, output_filename: str = None) -> str:
        """
        Compile a complete YouTube-style explainer video.
        
        Features:
        - Intro with character entrance
        - Scene-by-scene explanation with visuals
        - Smooth transitions
        - Synchronized audio
        - Summary/outro
        """
        try:
            from moviepy import VideoClip, ImageClip, AudioFileClip, concatenate_videoclips
            MOVIEPY_AVAILABLE = True
        except ImportError:
            try:
                from moviepy.editor import VideoClip, ImageClip, AudioFileClip, concatenate_videoclips
                MOVIEPY_AVAILABLE = True
            except ImportError:
                MOVIEPY_AVAILABLE = False
                print("⚠ MoviePy not available - generating frames only")
        
        subject = script.get("subject", "default")
        colors = self.get_colors(subject)
        scenes = script.get("scenes", [])
        topic = script.get("topic", "Educational Video")
        
        print(f"\n{'='*60}")
        print(f"Creating YouTube-Style Explainer Video")
        print(f"Topic: {topic}")
        print(f"Scenes: {len(scenes)}")
        print(f"{'='*60}")
        
        # Step 1: Generate audio for all scenes FIRST
        print("\n📢 Step 1: Generating audio narration...")
        audio_files = {}
        scene_durations = {}
        
        for scene in scenes:
            scene_num = scene.get("scene_number", 0)
            narration = scene.get("narration", "")
            
            if narration:
                filename = f"scene_{scene_num:02d}.mp3"
                audio_path = self.audio_service.generate_audio(narration, filename)
                
                if audio_path:
                    audio_files[scene_num] = audio_path
                    # Get actual audio duration
                    duration = self.audio_service.get_audio_duration(audio_path)
                    scene_durations[scene_num] = max(duration + 1, 5)  # Min 5 seconds
                    print(f"  Scene {scene_num}: {duration:.1f}s audio")
                else:
                    # Estimate duration from text
                    scene_durations[scene_num] = self.audio_service.estimate_speech_duration(narration)
            else:
                scene_durations[scene_num] = 5  # Default 5 seconds
        
        print(f"  Total audio files: {len(audio_files)}")

        # Step 2: Create video clips
        print("\n🎬 Step 2: Creating video frames...")
        clips = []
        
        if not MOVIEPY_AVAILABLE:
            return self._save_frames_fallback(script, colors, output_filename or "video")
        
        # Intro clip (3 seconds, no audio - just visual intro)
        print("  Creating intro...")
        intro_duration = 3
        intro_frames = int(intro_duration * self.FPS)
        
        def make_intro_frame(t):
            frame_num = int(t * self.FPS)
            img = self.create_intro_frame(script, colors, frame_num, intro_frames)
            return np.array(img)
        
        intro_clip = VideoClip(make_intro_frame, duration=intro_duration).with_fps(self.FPS)
        clips.append(intro_clip)
        
        # Scene clips with audio
        char_positions = ["left", "right"]  # Alternate character position
        
        for i, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", i + 1)
            duration = scene_durations.get(scene_num, 10)
            audio_path = audio_files.get(scene_num)
            
            print(f"  Creating scene {scene_num} ({duration:.1f}s)...")
            
            # Alternate character position for variety
            char_pos = char_positions[i % 2]
            
            # Create scene frames - use closure to capture variables
            def make_scene_frame_factory(sc, cp, dur):
                def make_frame(t):
                    frame_num = int(t * self.FPS)
                    progress = t / dur if dur > 0 else 0
                    img = self.create_explanation_frame(
                        sc, script, colors, frame_num, progress, cp
                    )
                    return np.array(img)
                return make_frame
            
            scene_clip = VideoClip(make_scene_frame_factory(scene, char_pos, duration), 
                                   duration=duration).with_fps(self.FPS)
            
            # Add audio if available
            if audio_path and os.path.exists(audio_path):
                try:
                    audio = AudioFileClip(audio_path)
                    scene_clip = scene_clip.with_audio(audio)
                except Exception as e:
                    print(f"    ⚠ Audio error: {e}")
            
            clips.append(scene_clip)
            
            # Add short transition between scenes (except last)
            if i < len(scenes) - 1:
                trans_duration = 0.5
                next_scene = scenes[i + 1]
                
                def make_trans_frame_factory(from_s, to_s, td):
                    def make_frame(t):
                        progress = t / td if td > 0 else 0
                        frame_num = int(t * self.FPS)
                        img = self.create_transition_frame(from_s, to_s, colors, progress, frame_num)
                        return np.array(img)
                    return make_frame
                
                trans_clip = VideoClip(make_trans_frame_factory(scene, next_scene, trans_duration), 
                                       duration=trans_duration).with_fps(self.FPS)
                clips.append(trans_clip)

        # Summary/outro clip (4 seconds)
        print("  Creating summary...")
        summary_duration = 4
        summary_total_frames = int(summary_duration * self.FPS)
        
        def make_summary_frame(t):
            frame_num = int(t * self.FPS)
            img = self.create_summary_frame(script, colors, frame_num, summary_total_frames)
            return np.array(img)
        
        summary_clip = VideoClip(make_summary_frame, duration=summary_duration).with_fps(self.FPS)
        clips.append(summary_clip)
        
        # Step 3: Compile final video
        print("\n🎥 Step 3: Compiling final video...")
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = topic.replace(" ", "_")[:25]
            output_filename = f"explainer_{topic_slug}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Write video file
        final_video.write_videofile(
            str(output_path),
            fps=self.FPS,
            codec='libx264',
            audio_codec='aac',
            preset='medium',
            threads=4,
            logger=None  # Suppress moviepy output
        )
        
        # Cleanup
        final_video.close()
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        
        total_duration = sum(scene_durations.values()) + intro_duration + summary_duration
        print(f"\n✅ Video created successfully!")
        print(f"   📁 File: {output_path}")
        print(f"   ⏱️  Duration: {total_duration:.1f} seconds")
        print(f"   🎬 Scenes: {len(scenes)}")
        
        return str(output_path)
    
    def _save_frames_fallback(self, script: dict, colors: dict, 
                             base_filename: str) -> str:
        """Save frames when MoviePy is not available"""
        frames_dir = self.output_dir / f"{base_filename}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        
        # Intro frames
        intro_frames = self.FPS * 3
        for i in range(intro_frames):
            frame = self.create_intro_frame(script, colors, i, intro_frames)
            frame.save(frames_dir / f"frame_{frame_count:05d}.png")
            frame_count += 1
        
        # Scene frames
        for scene in script.get("scenes", []):
            duration = scene.get("duration_seconds", 10)
            scene_frames = self.FPS * duration
            
            for i in range(scene_frames):
                progress = i / scene_frames
                frame = self.create_explanation_frame(
                    scene, script, colors, i, progress, "left"
                )
                frame.save(frames_dir / f"frame_{frame_count:05d}.png")
                frame_count += 1
        
        # Summary frames
        summary_frames = self.FPS * 4
        for i in range(summary_frames):
            frame = self.create_summary_frame(script, colors, i, summary_frames)
            frame.save(frames_dir / f"frame_{frame_count:05d}.png")
            frame_count += 1
        
        print(f"Saved {frame_count} frames to {frames_dir}")
        print("Use ffmpeg to compile: ffmpeg -framerate 30 -i frame_%05d.png -c:v libx264 output.mp4")
        
        return str(frames_dir)
