"""
Professional Character Animator for YouTube-Style Explainer Videos
Creates large, expressive animated characters with smooth movements
"""

from PIL import Image, ImageDraw, ImageFont
import math
from typing import Tuple, List, Dict, Optional


class CharacterAnimator:
    """
    Creates professional animated presenter characters for explainer videos.
    Features:
    - Large, visible character (takes up significant screen space)
    - Multiple character positions (left, right, center, corner)
    - Smooth gesture animations (pointing, presenting, waving, thinking)
    - Lip sync animation for talking
    - Head turns and body movements
    - Multiple character styles/colors
    """
    
    # Character color presets
    CHARACTER_STYLES = {
        "default": {
            "skin": (255, 213, 170),
            "hair": (60, 40, 30),
            "shirt": (70, 130, 180),
            "pants": (50, 50, 60),
            "accent": (255, 200, 100)
        },
        "professional": {
            "skin": (240, 200, 160),
            "hair": (40, 30, 25),
            "shirt": (50, 80, 120),
            "pants": (40, 40, 50),
            "accent": (200, 160, 80)
        },
        "friendly": {
            "skin": (255, 220, 180),
            "hair": (80, 50, 30),
            "shirt": (100, 180, 100),
            "pants": (60, 60, 70),
            "accent": (255, 180, 100)
        },
        "modern": {
            "skin": (245, 205, 165),
            "hair": (30, 30, 40),
            "shirt": (180, 80, 100),
            "pants": (45, 45, 55),
            "accent": (255, 150, 150)
        }
    }

    def __init__(self, scale: float = 1.5, style: str = "default"):
        """
        Initialize character animator.
        
        Args:
            scale: Character size multiplier (1.5 = 50% larger than before)
            style: Character color style preset
        """
        self.scale = scale
        self.style = self.CHARACTER_STYLES.get(style, self.CHARACTER_STYLES["default"])
        
        # Base dimensions (will be scaled)
        self.base_height = int(500 * scale)
        self.base_width = int(250 * scale)
        
        # Animation state
        self.current_gesture = "neutral"
        self.current_expression = "smile"
        self.talk_frame = 0
        self.blink_timer = 0
        
    def _s(self, value: int) -> int:
        """Scale a value"""
        return int(value * self.scale)
    
    def draw_character(self, 
                       draw: ImageDraw,
                       position: Tuple[int, int],
                       gesture: str = "neutral",
                       expression: str = "smile",
                       frame: int = 0,
                       facing: str = "front",
                       talking: bool = False) -> None:
        """
        Draw the character at the given position.
        
        Args:
            draw: PIL ImageDraw object
            position: (x, y) position for character center-bottom
            gesture: "neutral", "pointing_right", "pointing_up", "waving", 
                    "presenting", "thinking", "explaining", "both_hands_up"
            expression: "smile", "thinking", "explaining", "happy", "surprised"
            frame: Animation frame number for smooth motion
            facing: "front", "left", "right" - direction character faces
            talking: Whether character is currently speaking (for lip sync)
        """
        x, y = position
        
        # Animation offsets
        breath_offset = int(math.sin(frame * 0.08) * 4)
        sway_offset = int(math.sin(frame * 0.05) * 2)
        
        # Blink animation
        self.blink_timer = (self.blink_timer + 1) % 120
        is_blinking = self.blink_timer > 115
        
        # Draw shadow
        self._draw_shadow(draw, x, y)
        
        # Draw body parts from back to front
        self._draw_legs(draw, x, y, frame)
        self._draw_body(draw, x + sway_offset, y, breath_offset)
        self._draw_arms(draw, x + sway_offset, y, gesture, frame)
        self._draw_head(draw, x + sway_offset, y - self._s(320) + breath_offset, 
                       expression, frame, facing, talking, is_blinking)
    
    def _draw_shadow(self, draw: ImageDraw, x: int, y: int):
        """Draw character shadow on ground"""
        shadow_width = self._s(100)
        shadow_height = self._s(20)
        draw.ellipse([x - shadow_width//2, y - shadow_height//2,
                     x + shadow_width//2, y + shadow_height//2],
                    fill=(0, 0, 0, 50))

    def _draw_legs(self, draw: ImageDraw, x: int, y: int, frame: int):
        """Draw character legs with subtle animation"""
        pants_color = self.style["pants"]
        shoe_color = (40, 40, 40)
        
        # Subtle leg movement
        leg_offset = int(math.sin(frame * 0.1) * 2)
        
        # Left leg
        leg_points = [
            (x - self._s(35), y - self._s(130)),
            (x - self._s(8), y - self._s(130)),
            (x - self._s(12) + leg_offset, y - self._s(10)),
            (x - self._s(42) + leg_offset, y - self._s(10))
        ]
        draw.polygon(leg_points, fill=pants_color)
        
        # Right leg
        leg_points = [
            (x + self._s(8), y - self._s(130)),
            (x + self._s(35), y - self._s(130)),
            (x + self._s(42) - leg_offset, y - self._s(10)),
            (x + self._s(12) - leg_offset, y - self._s(10))
        ]
        draw.polygon(leg_points, fill=pants_color)
        
        # Shoes
        draw.ellipse([x - self._s(50) + leg_offset, y - self._s(20), 
                     x - self._s(5) + leg_offset, y + self._s(5)], fill=shoe_color)
        draw.ellipse([x + self._s(5) - leg_offset, y - self._s(20), 
                     x + self._s(50) - leg_offset, y + self._s(5)], fill=shoe_color)
    
    def _draw_body(self, draw: ImageDraw, x: int, y: int, offset: int):
        """Draw character torso"""
        shirt_color = self.style["shirt"]
        
        # Main torso
        torso_points = [
            (x - self._s(50), y - self._s(250) + offset),
            (x + self._s(50), y - self._s(250) + offset),
            (x + self._s(45), y - self._s(130)),
            (x - self._s(45), y - self._s(130))
        ]
        draw.polygon(torso_points, fill=shirt_color)
        
        # Collar/neckline
        collar_points = [
            (x - self._s(25), y - self._s(250) + offset),
            (x, y - self._s(220) + offset),
            (x + self._s(25), y - self._s(250) + offset)
        ]
        draw.polygon(collar_points, fill=(255, 255, 255))
        
        # Shirt details - buttons or design
        for i in range(3):
            btn_y = y - self._s(210) + i * self._s(30) + offset
            draw.ellipse([x - self._s(5), btn_y - self._s(4),
                         x + self._s(5), btn_y + self._s(4)],
                        fill=(200, 200, 200))

    def _draw_arms(self, draw: ImageDraw, x: int, y: int, gesture: str, frame: int):
        """Draw arms with gesture animations"""
        arm_color = self.style["shirt"]
        hand_color = self.style["skin"]
        
        # Animation offsets
        wave_offset = int(math.sin(frame * 0.25) * 20) if gesture == "waving" else 0
        point_offset = int(math.sin(frame * 0.12) * 8)
        present_offset = int(math.sin(frame * 0.1) * 5)
        
        arm_width = self._s(22)
        
        if gesture == "neutral":
            # Arms relaxed at sides
            self._draw_arm_segment(draw, x - self._s(50), y - self._s(240),
                                  x - self._s(70), y - self._s(150), arm_color, arm_width)
            self._draw_hand(draw, x - self._s(75), y - self._s(145), hand_color, "relaxed")
            
            self._draw_arm_segment(draw, x + self._s(50), y - self._s(240),
                                  x + self._s(70), y - self._s(150), arm_color, arm_width)
            self._draw_hand(draw, x + self._s(75), y - self._s(145), hand_color, "relaxed")
            
        elif gesture == "pointing_right":
            # Left arm relaxed
            self._draw_arm_segment(draw, x - self._s(50), y - self._s(240),
                                  x - self._s(70), y - self._s(150), arm_color, arm_width)
            self._draw_hand(draw, x - self._s(75), y - self._s(145), hand_color, "relaxed")
            
            # Right arm pointing with animation
            point_x = x + self._s(140) + point_offset
            point_y = y - self._s(220)
            self._draw_arm_segment(draw, x + self._s(50), y - self._s(240),
                                  point_x, point_y, arm_color, arm_width)
            self._draw_pointing_hand(draw, point_x + self._s(10), point_y, hand_color, "right")
            
        elif gesture == "pointing_up":
            # Left arm relaxed
            self._draw_arm_segment(draw, x - self._s(50), y - self._s(240),
                                  x - self._s(70), y - self._s(150), arm_color, arm_width)
            self._draw_hand(draw, x - self._s(75), y - self._s(145), hand_color, "relaxed")
            
            # Right arm pointing up
            point_y = y - self._s(350) + point_offset
            self._draw_arm_segment(draw, x + self._s(50), y - self._s(240),
                                  x + self._s(80), point_y, arm_color, arm_width)
            self._draw_pointing_hand(draw, x + self._s(80), point_y - self._s(20), hand_color, "up")
            
        elif gesture == "presenting" or gesture == "explaining":
            # Both arms out in presenting pose
            self._draw_arm_segment(draw, x - self._s(50), y - self._s(240),
                                  x - self._s(130) - present_offset, y - self._s(200), 
                                  arm_color, arm_width)
            self._draw_open_hand(draw, x - self._s(145) - present_offset, y - self._s(200), 
                                hand_color, "left")
            
            self._draw_arm_segment(draw, x + self._s(50), y - self._s(240),
                                  x + self._s(130) + present_offset, y - self._s(200), 
                                  arm_color, arm_width)
            self._draw_open_hand(draw, x + self._s(145) + present_offset, y - self._s(200), 
                                hand_color, "right")

        elif gesture == "waving":
            # Left arm relaxed
            self._draw_arm_segment(draw, x - self._s(50), y - self._s(240),
                                  x - self._s(70), y - self._s(150), arm_color, arm_width)
            self._draw_hand(draw, x - self._s(75), y - self._s(145), hand_color, "relaxed")
            
            # Right arm waving
            wave_y = y - self._s(340) + wave_offset
            self._draw_arm_segment(draw, x + self._s(50), y - self._s(240),
                                  x + self._s(90), wave_y, arm_color, arm_width)
            self._draw_open_hand(draw, x + self._s(95), wave_y - self._s(15), hand_color, "wave")
            
        elif gesture == "thinking":
            # Left arm relaxed
            self._draw_arm_segment(draw, x - self._s(50), y - self._s(240),
                                  x - self._s(70), y - self._s(150), arm_color, arm_width)
            self._draw_hand(draw, x - self._s(75), y - self._s(145), hand_color, "relaxed")
            
            # Right arm to chin
            self._draw_arm_segment(draw, x + self._s(50), y - self._s(240),
                                  x + self._s(25), y - self._s(310), arm_color, arm_width)
            self._draw_hand(draw, x + self._s(20), y - self._s(320), hand_color, "fist")
            
        elif gesture == "both_hands_up":
            # Both arms raised
            self._draw_arm_segment(draw, x - self._s(50), y - self._s(240),
                                  x - self._s(80), y - self._s(340), arm_color, arm_width)
            self._draw_open_hand(draw, x - self._s(85), y - self._s(355), hand_color, "up")
            
            self._draw_arm_segment(draw, x + self._s(50), y - self._s(240),
                                  x + self._s(80), y - self._s(340), arm_color, arm_width)
            self._draw_open_hand(draw, x + self._s(85), y - self._s(355), hand_color, "up")
    
    def _draw_arm_segment(self, draw: ImageDraw, x1: int, y1: int, 
                         x2: int, y2: int, color: tuple, width: int):
        """Draw an arm segment"""
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        # Add rounded ends
        draw.ellipse([x1 - width//2, y1 - width//2, x1 + width//2, y1 + width//2], fill=color)
        draw.ellipse([x2 - width//2, y2 - width//2, x2 + width//2, y2 + width//2], fill=color)
    
    def _draw_hand(self, draw: ImageDraw, x: int, y: int, color: tuple, style: str):
        """Draw a cartoon-style hand"""
        size = self._s(22)
        if style == "relaxed":
            # Simple rounded hand
            draw.ellipse([x - size, y - size, x + size, y + size], fill=color)
        elif style == "fist":
            # Fist shape - slightly oval
            draw.ellipse([x - size, y - size*0.7, x + size, y + size*0.7], fill=color)

    def _draw_pointing_hand(self, draw: ImageDraw, x: int, y: int, color: tuple, direction: str):
        """Draw a cartoon-style pointing hand"""
        palm_size = self._s(18)
        
        # Draw palm as oval
        draw.ellipse([x - palm_size, y - palm_size, x + palm_size, y + palm_size], fill=color)
        
        # Draw pointing finger as elongated oval (looks better than line)
        finger_length = self._s(30)
        finger_width = self._s(12)
        
        if direction == "right":
            # Finger pointing right
            draw.ellipse([x + palm_size - self._s(5), y - finger_width//2,
                         x + palm_size + finger_length, y + finger_width//2], fill=color)
        elif direction == "up":
            # Finger pointing up
            draw.ellipse([x - finger_width//2, y - palm_size - finger_length,
                         x + finger_width//2, y - palm_size + self._s(5)], fill=color)
        elif direction == "left":
            # Finger pointing left
            draw.ellipse([x - palm_size - finger_length, y - finger_width//2,
                         x - palm_size + self._s(5), y + finger_width//2], fill=color)
    
    def _draw_open_hand(self, draw: ImageDraw, x: int, y: int, color: tuple, side: str):
        """Draw a cartoon-style open hand (rounded, simple)"""
        # Use a simple rounded hand shape instead of individual fingers
        # This looks much better in 2D animation style
        
        hand_width = self._s(28)
        hand_height = self._s(32)
        
        if side in ["up", "wave"]:
            # Hand pointing up - oval shape
            draw.ellipse([x - hand_width//2, y - hand_height, 
                         x + hand_width//2, y + self._s(5)], fill=color)
            # Thumb on side
            draw.ellipse([x + hand_width//2 - self._s(5), y - hand_height//2,
                         x + hand_width//2 + self._s(10), y - hand_height//2 + self._s(15)], 
                        fill=color)
        elif side == "left":
            # Hand pointing left
            draw.ellipse([x - hand_height, y - hand_width//2,
                         x + self._s(5), y + hand_width//2], fill=color)
            # Thumb
            draw.ellipse([x - hand_height//2, y - hand_width//2 - self._s(8),
                         x - hand_height//2 + self._s(15), y - hand_width//2 + self._s(5)],
                        fill=color)
        elif side == "right":
            # Hand pointing right
            draw.ellipse([x - self._s(5), y - hand_width//2,
                         x + hand_height, y + hand_width//2], fill=color)
            # Thumb
            draw.ellipse([x + hand_height//2 - self._s(8), y - hand_width//2 - self._s(8),
                         x + hand_height//2 + self._s(8), y - hand_width//2 + self._s(5)],
                        fill=color)

    def _draw_head(self, draw: ImageDraw, x: int, y: int, expression: str, 
                   frame: int, facing: str, talking: bool, is_blinking: bool):
        """Draw character head with expressions and lip sync"""
        skin_color = self.style["skin"]
        hair_color = self.style["hair"]
        
        # Head dimensions
        head_width = self._s(90)
        head_height = self._s(100)
        
        # Facing offset for 3/4 view
        face_offset = 0
        if facing == "left":
            face_offset = -self._s(10)
        elif facing == "right":
            face_offset = self._s(10)
        
        # Head shape
        draw.ellipse([x - head_width//2 + face_offset, y - head_height//2,
                     x + head_width//2 + face_offset, y + head_height//2],
                    fill=skin_color)
        
        # Neck
        neck_width = self._s(35)
        draw.rectangle([x - neck_width//2, y + head_height//2 - self._s(10),
                       x + neck_width//2, y + head_height//2 + self._s(30)],
                      fill=skin_color)
        
        # Hair
        self._draw_hair(draw, x + face_offset, y, head_width, head_height, hair_color)
        
        # Face features
        self._draw_eyes(draw, x + face_offset, y - self._s(10), expression, frame, is_blinking)
        self._draw_eyebrows(draw, x + face_offset, y - self._s(30), expression)
        self._draw_nose(draw, x + face_offset, y + self._s(10))
        self._draw_mouth(draw, x + face_offset, y + self._s(30), expression, frame, talking)
        
        # Ears
        ear_y = y
        draw.ellipse([x - head_width//2 - self._s(8) + face_offset, ear_y - self._s(15),
                     x - head_width//2 + self._s(8) + face_offset, ear_y + self._s(15)],
                    fill=skin_color)
        draw.ellipse([x + head_width//2 - self._s(8) + face_offset, ear_y - self._s(15),
                     x + head_width//2 + self._s(8) + face_offset, ear_y + self._s(15)],
                    fill=skin_color)
    
    def _draw_hair(self, draw: ImageDraw, x: int, y: int, 
                   head_width: int, head_height: int, color: tuple):
        """Draw character hair"""
        # Main hair shape
        hair_points = [
            (x - head_width//2 - self._s(5), y - self._s(5)),
            (x - head_width//2 + self._s(15), y - head_height//2 - self._s(20)),
            (x - self._s(20), y - head_height//2 - self._s(35)),
            (x, y - head_height//2 - self._s(40)),
            (x + self._s(20), y - head_height//2 - self._s(35)),
            (x + head_width//2 - self._s(15), y - head_height//2 - self._s(20)),
            (x + head_width//2 + self._s(5), y - self._s(5)),
            (x + head_width//2, y - self._s(25)),
            (x - head_width//2, y - self._s(25))
        ]
        draw.polygon(hair_points, fill=color)

    def _draw_eyes(self, draw: ImageDraw, x: int, y: int, 
                   expression: str, frame: int, is_blinking: bool):
        """Draw eyes with animation"""
        eye_spacing = self._s(25)
        eye_width = self._s(16)
        eye_height = self._s(18) if not is_blinking else self._s(3)
        
        # Eye whites
        for side in [-1, 1]:
            ex = x + side * eye_spacing
            draw.ellipse([ex - eye_width//2, y - eye_height//2,
                         ex + eye_width//2, y + eye_height//2],
                        fill=(255, 255, 255))
        
        if not is_blinking:
            # Pupils with subtle movement
            pupil_offset_x = int(math.sin(frame * 0.03) * 3)
            pupil_offset_y = int(math.cos(frame * 0.04) * 2)
            pupil_size = self._s(8)
            
            for side in [-1, 1]:
                ex = x + side * eye_spacing + pupil_offset_x
                ey = y + pupil_offset_y
                draw.ellipse([ex - pupil_size//2, ey - pupil_size//2,
                             ex + pupil_size//2, ey + pupil_size//2],
                            fill=(40, 40, 50))
                
                # Eye shine
                shine_size = self._s(3)
                draw.ellipse([ex - pupil_size//4, ey - pupil_size//4,
                             ex - pupil_size//4 + shine_size, ey - pupil_size//4 + shine_size],
                            fill=(255, 255, 255))
    
    def _draw_eyebrows(self, draw: ImageDraw, x: int, y: int, expression: str):
        """Draw eyebrows based on expression"""
        brow_width = self._s(20)
        brow_spacing = self._s(25)
        brow_thickness = self._s(4)
        hair_color = self.style["hair"]
        
        if expression == "thinking":
            # One raised eyebrow
            draw.line([(x - brow_spacing - brow_width//2, y),
                      (x - brow_spacing + brow_width//2, y - self._s(5))],
                     fill=hair_color, width=brow_thickness)
            draw.line([(x + brow_spacing - brow_width//2, y + self._s(3)),
                      (x + brow_spacing + brow_width//2, y - self._s(3))],
                     fill=hair_color, width=brow_thickness)
        elif expression == "surprised":
            # Raised eyebrows
            for side in [-1, 1]:
                bx = x + side * brow_spacing
                draw.arc([bx - brow_width//2, y - self._s(8),
                         bx + brow_width//2, y + self._s(8)],
                        0, 180, fill=hair_color, width=brow_thickness)
        else:
            # Normal eyebrows
            for side in [-1, 1]:
                bx = x + side * brow_spacing
                draw.line([(bx - brow_width//2, y),
                          (bx + brow_width//2, y)],
                         fill=hair_color, width=brow_thickness)
    
    def _draw_nose(self, draw: ImageDraw, x: int, y: int):
        """Draw nose"""
        nose_color = tuple(max(0, c - 20) for c in self.style["skin"])
        draw.polygon([(x, y - self._s(12)),
                     (x - self._s(8), y + self._s(8)),
                     (x + self._s(8), y + self._s(8))],
                    fill=nose_color)

    def _draw_mouth(self, draw: ImageDraw, x: int, y: int, 
                    expression: str, frame: int, talking: bool):
        """Draw mouth with lip sync animation"""
        mouth_color = (180, 100, 100)
        inner_mouth = (100, 50, 50)
        teeth_color = (255, 255, 255)
        
        if talking:
            # Animated talking mouth
            talk_phase = (frame % 12) / 12.0
            mouth_open = abs(math.sin(talk_phase * math.pi * 2)) * self._s(15) + self._s(5)
            
            # Open mouth
            draw.ellipse([x - self._s(15), y - mouth_open//2,
                         x + self._s(15), y + mouth_open//2],
                        fill=mouth_color)
            
            # Inner mouth (dark)
            if mouth_open > self._s(8):
                draw.ellipse([x - self._s(12), y - mouth_open//3,
                             x + self._s(12), y + mouth_open//3],
                            fill=inner_mouth)
                
                # Teeth (top)
                draw.rectangle([x - self._s(10), y - mouth_open//3,
                               x + self._s(10), y - mouth_open//3 + self._s(4)],
                              fill=teeth_color)
        else:
            # Static expressions
            if expression in ["smile", "happy"]:
                # Smiling mouth
                draw.arc([x - self._s(18), y - self._s(12),
                         x + self._s(18), y + self._s(18)],
                        0, 180, fill=mouth_color, width=self._s(5))
            elif expression == "explaining":
                # Slightly open mouth
                draw.ellipse([x - self._s(12), y - self._s(5),
                             x + self._s(12), y + self._s(10)],
                            fill=mouth_color)
                draw.ellipse([x - self._s(10), y - self._s(2),
                             x + self._s(10), y + self._s(7)],
                            fill=inner_mouth)
            elif expression == "thinking":
                # Slight frown
                draw.arc([x - self._s(12), y,
                         x + self._s(12), y + self._s(12)],
                        180, 360, fill=mouth_color, width=self._s(4))
            elif expression == "surprised":
                # O-shaped mouth
                draw.ellipse([x - self._s(10), y - self._s(10),
                             x + self._s(10), y + self._s(10)],
                            fill=mouth_color)
                draw.ellipse([x - self._s(7), y - self._s(7),
                             x + self._s(7), y + self._s(7)],
                            fill=inner_mouth)
            else:
                # Neutral
                draw.line([(x - self._s(15), y), (x + self._s(15), y)],
                         fill=mouth_color, width=self._s(4))


class TopicVisualizer:
    """
    Creates topic-specific visual content for explainer videos.
    Generates diagrams, circuits, flowcharts, and animations based on the topic.
    """
    
    # Topic-specific visual templates
    TOPIC_VISUALS = {
        "electric_current": ["circuit", "electron_flow", "battery", "resistor"],
        "photosynthesis": ["plant_cell", "chloroplast", "light_reaction", "calvin_cycle"],
        "dna": ["double_helix", "base_pairs", "replication_fork", "nucleotides"],
        "newton": ["force_arrows", "motion_diagram", "free_body", "acceleration"],
        "chemical_bonding": ["electron_shells", "ionic_bond", "covalent_bond", "molecular_structure"],
        "cell": ["cell_membrane", "nucleus", "organelles", "mitochondria"],
        "wave": ["wave_diagram", "amplitude", "wavelength", "frequency"],
        "atom": ["atomic_structure", "electron_orbits", "nucleus", "protons_neutrons"],
    }
    
    def __init__(self):
        self.colors = {
            "primary": (102, 126, 234),
            "secondary": (118, 75, 162),
            "accent": (240, 147, 251),
            "success": (16, 185, 129),
            "warning": (245, 158, 11),
            "error": (239, 68, 68),
            "text": (255, 255, 255),
            "text_dark": (30, 30, 30),
            "wire": (100, 100, 100),
            "positive": (239, 68, 68),
            "negative": (59, 130, 246),
            "electron": (255, 220, 100),
        }
    
    def detect_topic_type(self, topic: str) -> str:
        """Detect the type of topic for appropriate visualization"""
        topic_lower = topic.lower()
        
        if any(kw in topic_lower for kw in ["electric", "current", "circuit", "voltage", "resistance"]):
            return "electric_current"
        elif any(kw in topic_lower for kw in ["photosynthesis", "chlorophyll", "plant"]):
            return "photosynthesis"
        elif any(kw in topic_lower for kw in ["dna", "rna", "gene", "chromosome", "replication"]):
            return "dna"
        elif any(kw in topic_lower for kw in ["newton", "force", "motion", "inertia", "momentum"]):
            return "newton"
        elif any(kw in topic_lower for kw in ["bond", "ionic", "covalent", "molecule"]):
            return "chemical_bonding"
        elif any(kw in topic_lower for kw in ["cell", "membrane", "nucleus", "organelle"]):
            return "cell"
        elif any(kw in topic_lower for kw in ["wave", "frequency", "amplitude", "sound", "light"]):
            return "wave"
        elif any(kw in topic_lower for kw in ["atom", "electron", "proton", "neutron", "orbital"]):
            return "atom"
        else:
            return "general"

    def draw_topic_visual(self, draw: ImageDraw, topic: str, 
                         position: Tuple[int, int], size: Tuple[int, int],
                         frame: int, scene_data: dict = None, font=None) -> None:
        """
        Draw topic-specific visualization.
        
        Args:
            draw: PIL ImageDraw object
            topic: The topic being explained
            position: (x, y) center position for the visual
            size: (width, height) of the visual area
            frame: Animation frame number
            scene_data: Additional scene data (key_points, etc.)
        """
        topic_type = self.detect_topic_type(topic)
        x, y = position
        width, height = size
        
        if topic_type == "electric_current":
            self._draw_circuit_diagram(draw, x, y, width, height, frame)
        elif topic_type == "photosynthesis":
            self._draw_photosynthesis(draw, x, y, width, height, frame)
        elif topic_type == "dna":
            self._draw_dna_helix(draw, x, y, width, height, frame)
        elif topic_type == "newton":
            self._draw_force_diagram(draw, x, y, width, height, frame)
        elif topic_type == "chemical_bonding":
            self._draw_molecular_structure(draw, x, y, width, height, frame)
        elif topic_type == "cell":
            self._draw_cell_diagram(draw, x, y, width, height, frame)
        elif topic_type == "wave":
            self._draw_wave_diagram(draw, x, y, width, height, frame)
        elif topic_type == "atom":
            self._draw_atomic_structure(draw, x, y, width, height, frame)
        else:
            # Generic concept visualization
            if scene_data and scene_data.get("key_points"):
                self._draw_concept_visual(draw, x, y, width, height, 
                                         scene_data.get("key_points", []), font)
    
    def _draw_circuit_diagram(self, draw: ImageDraw, x: int, y: int, 
                              width: int, height: int, frame: int):
        """Draw an animated electric circuit"""
        wire_color = self.colors["wire"]
        
        # Circuit rectangle
        cx, cy = x, y
        cw, ch = width * 0.7, height * 0.6
        
        # Draw wires
        wire_width = 4
        # Top wire
        draw.line([(cx - cw//2, cy - ch//2), (cx + cw//2, cy - ch//2)], 
                 fill=wire_color, width=wire_width)
        # Right wire
        draw.line([(cx + cw//2, cy - ch//2), (cx + cw//2, cy + ch//2)], 
                 fill=wire_color, width=wire_width)
        # Bottom wire
        draw.line([(cx - cw//2, cy + ch//2), (cx + cw//2, cy + ch//2)], 
                 fill=wire_color, width=wire_width)
        # Left wire
        draw.line([(cx - cw//2, cy - ch//2), (cx - cw//2, cy + ch//2)], 
                 fill=wire_color, width=wire_width)
        
        # Battery (left side)
        bat_x = cx - cw//2
        bat_y = cy
        # Long line (positive)
        draw.line([(bat_x - 15, bat_y - 25), (bat_x - 15, bat_y + 25)], 
                 fill=self.colors["positive"], width=6)
        # Short line (negative)
        draw.line([(bat_x + 15, bat_y - 15), (bat_x + 15, bat_y + 15)], 
                 fill=self.colors["negative"], width=6)
        # + and - labels
        draw.text((bat_x - 25, bat_y - 40), "+", fill=self.colors["positive"])
        draw.text((bat_x + 20, bat_y - 40), "-", fill=self.colors["negative"])

        # Resistor (top)
        res_x = cx
        res_y = cy - ch//2
        zigzag_points = []
        for i in range(7):
            zx = res_x - 40 + i * 13
            zy = res_y + (10 if i % 2 == 0 else -10)
            zigzag_points.append((zx, zy))
        for i in range(len(zigzag_points) - 1):
            draw.line([zigzag_points[i], zigzag_points[i+1]], 
                     fill=self.colors["warning"], width=3)
        
        # Bulb (right side)
        bulb_x = cx + cw//2
        bulb_y = cy
        # Bulb glow animation
        glow_intensity = int(abs(math.sin(frame * 0.15)) * 50) + 30
        glow_color = (255, 255, glow_intensity + 150)
        draw.ellipse([bulb_x - 25, bulb_y - 30, bulb_x + 25, bulb_y + 30], 
                    fill=glow_color, outline=(200, 200, 200), width=2)
        # Filament
        draw.arc([bulb_x - 10, bulb_y - 15, bulb_x + 10, bulb_y + 15], 
                0, 180, fill=(255, 200, 100), width=2)
        
        # Animated electrons
        num_electrons = 8
        for i in range(num_electrons):
            # Calculate position along circuit
            progress = ((frame * 2 + i * 45) % 360) / 360.0
            ex, ey = self._get_circuit_position(cx, cy, cw, ch, progress)
            
            # Draw electron
            draw.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], 
                        fill=self.colors["electron"])
            draw.text((ex - 3, ey - 5), "e⁻", fill=(0, 0, 0))
        
        # Labels
        draw.text((cx - 30, cy + ch//2 + 20), "Electric Circuit", 
                 fill=self.colors["text"])
        draw.text((bat_x - 40, cy + 50), "Battery", fill=self.colors["text"])
        draw.text((res_x - 30, res_y - 40), "Resistor", fill=self.colors["text"])
        draw.text((bulb_x - 20, bulb_y + 40), "Bulb", fill=self.colors["text"])
    
    def _get_circuit_position(self, cx: int, cy: int, cw: int, ch: int, 
                              progress: float) -> Tuple[int, int]:
        """Get position along circuit path"""
        perimeter = 2 * cw + 2 * ch
        distance = progress * perimeter
        
        if distance < cw:
            # Top edge (left to right)
            return (int(cx - cw//2 + distance), int(cy - ch//2))
        elif distance < cw + ch:
            # Right edge (top to bottom)
            return (int(cx + cw//2), int(cy - ch//2 + (distance - cw)))
        elif distance < 2*cw + ch:
            # Bottom edge (right to left)
            return (int(cx + cw//2 - (distance - cw - ch)), int(cy + ch//2))
        else:
            # Left edge (bottom to top)
            return (int(cx - cw//2), int(cy + ch//2 - (distance - 2*cw - ch)))

    def _draw_wave_diagram(self, draw: ImageDraw, x: int, y: int,
                          width: int, height: int, frame: int):
        """Draw animated wave diagram"""
        wave_color = self.colors["primary"]
        
        # Axis
        draw.line([(x - width//2, y), (x + width//2, y)], 
                 fill=(150, 150, 150), width=2)
        draw.line([(x - width//2 + 20, y - height//2), (x - width//2 + 20, y + height//2)], 
                 fill=(150, 150, 150), width=2)
        
        # Animated sine wave
        points = []
        amplitude = height * 0.35
        wavelength = width * 0.3
        phase = frame * 0.1
        
        for i in range(int(width)):
            wx = x - width//2 + i
            wy = y - amplitude * math.sin(2 * math.pi * i / wavelength + phase)
            points.append((wx, int(wy)))
        
        # Draw wave
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=wave_color, width=3)
        
        # Amplitude arrow
        peak_x = x - width//4
        draw.line([(peak_x, y), (peak_x, int(y - amplitude))], 
                 fill=self.colors["accent"], width=2)
        draw.polygon([(peak_x, int(y - amplitude - 10)), 
                     (peak_x - 5, int(y - amplitude)), 
                     (peak_x + 5, int(y - amplitude))], 
                    fill=self.colors["accent"])
        draw.text((peak_x + 10, int(y - amplitude//2)), "Amplitude", 
                 fill=self.colors["text"])
        
        # Wavelength indicator
        draw.line([(x - width//4, y + height//3), (x + width//4 - 30, y + height//3)], 
                 fill=self.colors["success"], width=2)
        draw.text((x - 40, y + height//3 + 10), "Wavelength (λ)", 
                 fill=self.colors["text"])
    
    def _draw_atomic_structure(self, draw: ImageDraw, x: int, y: int,
                               width: int, height: int, frame: int):
        """Draw animated atomic structure"""
        # Nucleus
        nucleus_radius = min(width, height) * 0.1
        draw.ellipse([x - nucleus_radius, y - nucleus_radius,
                     x + nucleus_radius, y + nucleus_radius],
                    fill=self.colors["positive"])
        
        # Protons and neutrons in nucleus
        for i in range(3):
            angle = i * 2.1
            px = x + int(nucleus_radius * 0.4 * math.cos(angle))
            py = y + int(nucleus_radius * 0.4 * math.sin(angle))
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], 
                        fill=(255, 100, 100))  # Proton
        for i in range(3):
            angle = i * 2.1 + 1
            nx = x + int(nucleus_radius * 0.4 * math.cos(angle))
            ny = y + int(nucleus_radius * 0.4 * math.sin(angle))
            draw.ellipse([nx - 8, ny - 8, nx + 8, ny + 8], 
                        fill=(150, 150, 150))  # Neutron

        # Electron orbits
        orbit_radii = [min(width, height) * 0.25, min(width, height) * 0.4]
        electrons_per_orbit = [2, 4]
        
        for orbit_idx, (radius, num_electrons) in enumerate(zip(orbit_radii, electrons_per_orbit)):
            # Draw orbit path
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                        outline=(100, 100, 150), width=1)
            
            # Animated electrons
            for e in range(num_electrons):
                angle = (frame * 0.05 * (orbit_idx + 1) + e * (2 * math.pi / num_electrons))
                ex = x + int(radius * math.cos(angle))
                ey = y + int(radius * math.sin(angle))
                
                # Electron with glow
                draw.ellipse([ex - 10, ey - 10, ex + 10, ey + 10],
                            fill=self.colors["negative"])
                draw.text((ex - 4, ey - 6), "e⁻", fill=(255, 255, 255))
        
        # Labels
        draw.text((x - 25, y + orbit_radii[-1] + 20), "Atomic Structure", 
                 fill=self.colors["text"])
    
    def _draw_dna_helix(self, draw: ImageDraw, x: int, y: int,
                        width: int, height: int, frame: int):
        """Draw animated DNA double helix"""
        helix_height = height * 0.8
        helix_width = width * 0.3
        
        # Animation phase
        phase = frame * 0.05
        
        # Draw the two strands
        strand1_points = []
        strand2_points = []
        
        for i in range(50):
            t = i / 50.0
            hy = y - helix_height//2 + t * helix_height
            
            # Strand 1
            hx1 = x + int(helix_width * math.sin(t * 4 * math.pi + phase))
            strand1_points.append((hx1, int(hy)))
            
            # Strand 2 (opposite phase)
            hx2 = x + int(helix_width * math.sin(t * 4 * math.pi + phase + math.pi))
            strand2_points.append((hx2, int(hy)))
        
        # Draw strands
        for i in range(len(strand1_points) - 1):
            draw.line([strand1_points[i], strand1_points[i+1]], 
                     fill=self.colors["primary"], width=4)
            draw.line([strand2_points[i], strand2_points[i+1]], 
                     fill=self.colors["secondary"], width=4)
        
        # Draw base pairs (rungs)
        base_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        for i in range(0, 50, 5):
            if i < len(strand1_points) and i < len(strand2_points):
                color = base_colors[i % 4]
                draw.line([strand1_points[i], strand2_points[i]], fill=color, width=3)
        
        draw.text((x - 40, y + helix_height//2 + 10), "DNA Double Helix", 
                 fill=self.colors["text"])

    def _draw_force_diagram(self, draw: ImageDraw, x: int, y: int,
                           width: int, height: int, frame: int):
        """Draw Newton's force diagram"""
        # Object (box)
        box_size = min(width, height) * 0.2
        box_x = x + int(math.sin(frame * 0.05) * 20)  # Animated movement
        
        draw.rectangle([box_x - box_size//2, y - box_size//2,
                       box_x + box_size//2, y + box_size//2],
                      fill=self.colors["warning"], outline=(200, 150, 50), width=2)
        draw.text((box_x - 10, y - 8), "m", fill=self.colors["text_dark"])
        
        # Force arrows
        arrow_length = min(width, height) * 0.25
        
        # Applied force (right)
        self._draw_arrow(draw, box_x + box_size//2, y, 
                        box_x + box_size//2 + arrow_length, y,
                        self.colors["positive"], "F")
        
        # Friction (left)
        self._draw_arrow(draw, box_x - box_size//2, y,
                        box_x - box_size//2 - arrow_length * 0.6, y,
                        self.colors["negative"], "f")
        
        # Weight (down)
        self._draw_arrow(draw, box_x, y + box_size//2,
                        box_x, y + box_size//2 + arrow_length * 0.8,
                        self.colors["success"], "W")
        
        # Normal force (up)
        self._draw_arrow(draw, box_x, y - box_size//2,
                        box_x, y - box_size//2 - arrow_length * 0.8,
                        self.colors["accent"], "N")
        
        # Ground
        draw.line([(x - width//2, y + box_size//2 + 5), 
                  (x + width//2, y + box_size//2 + 5)],
                 fill=(150, 150, 150), width=3)
        
        # Equation
        draw.text((x - 60, y + height//3), "F = ma", fill=self.colors["text"])
    
    def _draw_arrow(self, draw: ImageDraw, x1: int, y1: int, 
                   x2: int, y2: int, color: tuple, label: str = ""):
        """Draw an arrow with optional label"""
        # Line
        draw.line([(x1, y1), (x2, y2)], fill=color, width=4)
        
        # Arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_size = 15
        
        left_x = x2 - arrow_size * math.cos(angle - math.pi/6)
        left_y = y2 - arrow_size * math.sin(angle - math.pi/6)
        right_x = x2 - arrow_size * math.cos(angle + math.pi/6)
        right_y = y2 - arrow_size * math.sin(angle + math.pi/6)
        
        draw.polygon([(x2, y2), (int(left_x), int(left_y)), 
                     (int(right_x), int(right_y))], fill=color)
        
        # Label
        if label:
            label_x = x2 + int(15 * math.cos(angle))
            label_y = y2 + int(15 * math.sin(angle)) - 10
            draw.text((label_x, label_y), label, fill=color)

    def _draw_photosynthesis(self, draw: ImageDraw, x: int, y: int,
                             width: int, height: int, frame: int):
        """Draw photosynthesis diagram"""
        # Leaf shape
        leaf_width = width * 0.5
        leaf_height = height * 0.4
        
        # Leaf outline
        leaf_points = [
            (x, y - leaf_height//2),
            (x + leaf_width//2, y - leaf_height//4),
            (x + leaf_width//2, y + leaf_height//4),
            (x, y + leaf_height//2),
            (x - leaf_width//2, y + leaf_height//4),
            (x - leaf_width//2, y - leaf_height//4),
        ]
        draw.polygon(leaf_points, fill=(50, 180, 50), outline=(30, 120, 30), width=2)
        
        # Leaf vein
        draw.line([(x, y - leaf_height//2), (x, y + leaf_height//2)], 
                 fill=(30, 120, 30), width=2)
        
        # Sun (animated rays)
        sun_x = x - width//3
        sun_y = y - height//3
        sun_radius = 25
        draw.ellipse([sun_x - sun_radius, sun_y - sun_radius,
                     sun_x + sun_radius, sun_y + sun_radius],
                    fill=(255, 220, 50))
        
        # Animated sun rays
        for i in range(8):
            angle = i * math.pi / 4 + frame * 0.02
            ray_length = 20 + int(math.sin(frame * 0.1 + i) * 5)
            rx1 = sun_x + int((sun_radius + 5) * math.cos(angle))
            ry1 = sun_y + int((sun_radius + 5) * math.sin(angle))
            rx2 = sun_x + int((sun_radius + ray_length) * math.cos(angle))
            ry2 = sun_y + int((sun_radius + ray_length) * math.sin(angle))
            draw.line([(rx1, ry1), (rx2, ry2)], fill=(255, 200, 50), width=3)
        
        # Light arrow to leaf
        draw.line([(sun_x + sun_radius + 30, sun_y + 20), (x - leaf_width//3, y - leaf_height//4)],
                 fill=(255, 255, 100), width=2)
        draw.text((sun_x + 40, sun_y + 30), "Light", fill=(255, 255, 100))
        
        # CO2 input
        draw.text((x - leaf_width//2 - 60, y), "CO₂ →", fill=self.colors["text"])
        
        # H2O input
        draw.text((x - 20, y + leaf_height//2 + 20), "H₂O ↑", fill=self.colors["primary"])
        
        # O2 output
        draw.text((x + leaf_width//2 + 10, y), "→ O₂", fill=self.colors["success"])
        
        # Glucose output
        draw.text((x - 30, y - leaf_height//2 - 30), "Glucose", fill=self.colors["warning"])
        
        # Equation
        draw.text((x - 100, y + height//3), "6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂", 
                 fill=self.colors["text"])

    def _draw_molecular_structure(self, draw: ImageDraw, x: int, y: int,
                                  width: int, height: int, frame: int):
        """Draw molecular/chemical bonding diagram"""
        # Central atom
        atom_radius = 35
        draw.ellipse([x - atom_radius, y - atom_radius,
                     x + atom_radius, y + atom_radius],
                    fill=self.colors["primary"])
        draw.text((x - 8, y - 10), "C", fill=(255, 255, 255))
        
        # Bonded atoms (tetrahedral arrangement)
        bond_length = min(width, height) * 0.25
        bond_angles = [0, math.pi/2, math.pi, 3*math.pi/2]
        atom_labels = ["H", "H", "H", "H"]
        
        for i, (angle, label) in enumerate(zip(bond_angles, atom_labels)):
            # Animated bond vibration
            vibration = int(math.sin(frame * 0.1 + i) * 3)
            
            bx = x + int((bond_length + vibration) * math.cos(angle))
            by = y + int((bond_length + vibration) * math.sin(angle))
            
            # Bond line
            draw.line([(x, y), (bx, by)], fill=(200, 200, 200), width=4)
            
            # Bonded atom
            small_radius = 25
            draw.ellipse([bx - small_radius, by - small_radius,
                         bx + small_radius, by + small_radius],
                        fill=self.colors["accent"])
            draw.text((bx - 6, by - 8), label, fill=(0, 0, 0))
        
        # Electron pairs (dots)
        for i, angle in enumerate(bond_angles):
            mid_x = x + int(bond_length * 0.4 * math.cos(angle))
            mid_y = y + int(bond_length * 0.4 * math.sin(angle))
            draw.ellipse([mid_x - 4, mid_y - 4, mid_x + 4, mid_y + 4],
                        fill=self.colors["electron"])
        
        draw.text((x - 50, y + bond_length + 40), "Covalent Bonding", 
                 fill=self.colors["text"])
    
    def _draw_cell_diagram(self, draw: ImageDraw, x: int, y: int,
                          width: int, height: int, frame: int):
        """Draw cell structure diagram"""
        cell_width = width * 0.6
        cell_height = height * 0.5
        
        # Cell membrane
        draw.ellipse([x - cell_width//2, y - cell_height//2,
                     x + cell_width//2, y + cell_height//2],
                    fill=(200, 230, 200), outline=(100, 150, 100), width=4)
        
        # Nucleus
        nucleus_x = x - cell_width//6
        nucleus_radius = min(cell_width, cell_height) * 0.2
        draw.ellipse([nucleus_x - nucleus_radius, y - nucleus_radius,
                     nucleus_x + nucleus_radius, y + nucleus_radius],
                    fill=(150, 100, 150), outline=(100, 50, 100), width=2)
        draw.text((nucleus_x - 25, y - 8), "Nucleus", fill=(255, 255, 255))
        
        # Mitochondria
        mito_x = x + cell_width//4
        mito_y = y - cell_height//6
        draw.ellipse([mito_x - 30, mito_y - 15, mito_x + 30, mito_y + 15],
                    fill=(200, 150, 150), outline=(150, 100, 100), width=2)
        
        # ER (wavy lines)
        for i in range(3):
            ey = y + cell_height//6 + i * 15
            points = [(x + j * 10, ey + int(math.sin(j + frame * 0.1) * 5)) 
                     for j in range(-5, 6)]
            for j in range(len(points) - 1):
                draw.line([points[j], points[j+1]], fill=(180, 180, 100), width=2)
        
        draw.text((x - 30, y + cell_height//2 + 15), "Animal Cell", 
                 fill=self.colors["text"])

    def _clean_text(self, text: str) -> str:
        """Clean NCERT content text."""
        import re
        if not text:
            return ""
        text = re.sub(r'Chapter\s+\w+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'CHAPTER\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\.\d+\s*', '', text)
        text = re.sub(r'[-_]{2,}', '', text)
        text = re.sub(r'\s*-\s*-\s*-\s*', ' ', text)
        text = re.sub(r'[A-Z]{3,}\s+[A-Z]+\s+\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _draw_concept_visual(self, draw: ImageDraw, x: int, y: int,
                            width: int, height: int, key_points: List[str], font=None):
        """Draw creative concept visualization with proper alignment"""
        import re
        
        # Clean the key points
        clean_points = []
        for p in key_points:
            cleaned = self._clean_text(p)
            if cleaned and len(cleaned) > 15:
                clean_points.append(cleaned)
        
        if not clean_points:
            clean_points = ["Key concept information will be displayed here"]
        
        clean_points = clean_points[:4]  # Max 4 points
        
        # Background panel with rounded corners effect
        panel_x1 = x - width//2
        panel_y1 = y - height//2
        panel_x2 = x + width//2
        panel_y2 = y + height//2
        
        # Main panel
        draw.rounded_rectangle([panel_x1, panel_y1, panel_x2, panel_y2],
                              radius=20, fill=(35, 45, 75), outline=self.colors["primary"], width=2)
        
        # Title bar
        draw.rounded_rectangle([panel_x1, panel_y1, panel_x2, panel_y1 + 50],
                              radius=20, fill=self.colors["primary"])
        
        # Title text
        title_text = "Key Points"
        if font:
            bbox = draw.textbbox((0, 0), title_text, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x - tw//2, panel_y1 + 12), title_text, fill=(255, 255, 255), font=font)
        
        # Draw each point as a card
        card_height = (height - 80) // len(clean_points) - 10
        card_start_y = panel_y1 + 60
        
        for i, point in enumerate(clean_points):
            card_y = card_start_y + i * (card_height + 8)
            
            # Card background
            draw.rounded_rectangle([panel_x1 + 15, card_y, panel_x2 - 15, card_y + card_height],
                                  radius=12, fill=(50, 60, 95))
            
            # Number badge
            badge_x = panel_x1 + 40
            badge_y = card_y + card_height // 2
            draw.ellipse([badge_x - 14, badge_y - 14, badge_x + 14, badge_y + 14],
                        fill=self.colors["accent"])
            draw.text((badge_x - 5, badge_y - 9), str(i + 1), fill=(0, 0, 0))
            
            # Text - wrap to fit
            text_x = panel_x1 + 70
            text_width = width - 100
            wrapped = self._wrap_text_simple(point, text_width, font, draw)
            lines = wrapped.split('\n')
            
            line_height = 26
            total_height = len(lines) * line_height
            text_y = card_y + (card_height - total_height) // 2
            
            for line in lines:
                if font:
                    draw.text((text_x, text_y), line, fill=self.colors["text"], font=font)
                else:
                    draw.text((text_x, text_y), line, fill=self.colors["text"])
                text_y += line_height
    
    def _wrap_text_simple(self, text: str, max_width: int, font, draw) -> str:
        """Simple text wrapping."""
        if not text:
            return ""
        
        # Clean the text first
        text = self._clean_text(text)
        
        if len(text) < 40:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font:
                try:
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    text_width = bbox[2] - bbox[0]
                except:
                    text_width = len(test_line) * 9
            else:
                text_width = len(test_line) * 9
            
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
        
        return '\n'.join(lines) if lines else text[:50]


class DiagramGenerator:
    """
    Generates educational diagrams for explainer videos.
    Supports: flowcharts, process diagrams, concept maps, comparison tables
    """
    
    def __init__(self):
        self.colors = {
            "primary": (102, 126, 234),
            "secondary": (118, 75, 162),
            "accent": (240, 147, 251),
            "success": (16, 185, 129),
            "warning": (245, 158, 11),
            "text": (255, 255, 255),
            "box_fill": (40, 50, 80),
            "arrow": (200, 200, 200)
        }
    
    def draw_flowchart(self, draw: ImageDraw, steps: List[str],
                      position: Tuple[int, int], width: int = 500,
                      font=None, animated_step: int = -1) -> None:
        """Draw a vertical flowchart with full text (word wrapped)"""
        x, y = position
        box_height = 70  # Taller boxes for more text
        box_width = width
        spacing = 20
        arrow_length = 15
        
        for i, step in enumerate(steps[:5]):  # Max 5 steps
            box_y = y + i * (box_height + spacing + arrow_length)
            
            # Highlight animated step
            fill_color = self.colors["primary"] if i == animated_step else self.colors["box_fill"]
            outline_color = self.colors["accent"] if i == animated_step else self.colors["primary"]
            
            # Draw box
            self._draw_rounded_rect(draw, x - box_width//2, box_y,
                                   x + box_width//2, box_y + box_height,
                                   radius=10, fill=fill_color, outline=outline_color)
            
            # Step number
            draw.ellipse([x - box_width//2 + 8, box_y + 22,
                         x - box_width//2 + 32, box_y + 46],
                        fill=self.colors["accent"])
            draw.text((x - box_width//2 + 15, box_y + 25), str(i + 1), 
                     fill=(0, 0, 0))
            
            # Word wrap text to fit in box
            text = self._wrap_text(step, box_width - 60, font, draw)
            text_x = x - box_width//2 + 42
            text_y = box_y + 12 if len(text.split('\n')) > 1 else box_y + 22
            
            if font:
                draw.text((text_x, text_y), text,
                         fill=self.colors["text"], font=font)
            else:
                draw.text((text_x, text_y), text,
                         fill=self.colors["text"])
            
            # Arrow to next
            if i < len(steps) - 1 and i < 4:
                arrow_y = box_y + box_height + 3
                self._draw_arrow(draw, x, arrow_y, x, arrow_y + spacing + arrow_length - 5)

    def draw_process_diagram(self, draw: ImageDraw, stages: List[str],
                            position: Tuple[int, int], width: int = 900,
                            font=None, animated_stage: int = -1) -> None:
        """Draw horizontal process diagram with full labels"""
        x, y = position
        num_stages = min(len(stages), 4)  # Max 4 for readability
        stage_width = width // max(num_stages, 1)
        circle_radius = 45
        
        for i, stage in enumerate(stages[:4]):
            stage_x = x - width//2 + stage_width//2 + i * stage_width
            
            # Highlight animated stage
            fill_color = self.colors["accent"] if i == animated_stage else self.colors["primary"]
            
            # Circle
            draw.ellipse([stage_x - circle_radius, y - circle_radius,
                         stage_x + circle_radius, y + circle_radius],
                        fill=fill_color, outline=self.colors["text"], width=2)
            
            # Number
            draw.text((stage_x - 8, y - 12), str(i + 1), fill=self.colors["text"])
            
            # Word wrap label below circle
            label = self._wrap_text(stage, stage_width - 20, font, draw, max_lines=2)
            if font:
                lines = label.split('\n')
                line_y = y + circle_radius + 12
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                    draw.text((stage_x - text_width//2, line_y),
                             line, fill=self.colors["text"], font=font)
                    line_y += 22
            else:
                draw.text((stage_x - len(stage) * 3, y + circle_radius + 15),
                         stage, fill=self.colors["text"])
            
            # Arrow
            if i < num_stages - 1:
                arrow_start = stage_x + circle_radius + 10
                arrow_end = stage_x + stage_width - circle_radius - 10
                self._draw_arrow(draw, arrow_start, y, arrow_end, y)
    
    def _clean_content_text(self, text: str) -> str:
        """Clean up NCERT content text - remove chapter markers, dashes, etc."""
        import re
        if not text:
            return ""
        
        # Remove chapter markers like "Chapter Eight", "CHAPTER 8", etc.
        text = re.sub(r'Chapter\s+\w+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'CHAPTER\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove section numbers like "8.1", "8.2"
        text = re.sub(r'\d+\.\d+\s*', '', text)
        
        # Remove dashes and underscores used as separators
        text = re.sub(r'[-_]{2,}', '', text)
        text = re.sub(r'\s*-\s*-\s*-\s*', ' ', text)
        
        # Remove "ELECTROMAGNETIC WAVES 8" type headers
        text = re.sub(r'[A-Z]{3,}\s+[A-Z]+\s+\d+', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def draw_concept_box(self, draw: ImageDraw, title: str, points: List[str],
                        position: Tuple[int, int], width: int = 700,
                        font=None, title_font=None, highlight_point: int = -1) -> int:
        """Draw a creative concept box with proper text alignment."""
        x, y = position
        padding = 25
        item_height = 70  # Height per item for proper spacing
        title_height = 60
        
        # Clean the points
        clean_points = [self._clean_content_text(p) for p in points if p]
        clean_points = [p for p in clean_points if len(p) > 10][:4]  # Max 4 clean points
        
        if not clean_points:
            clean_points = ["Key concept information"]
        
        # Calculate total height
        height = title_height + padding + len(clean_points) * item_height + padding
        
        # Main box with gradient effect (darker at bottom)
        self._draw_rounded_rect(draw, x - width//2, y, x + width//2, y + height,
                               radius=20, fill=(35, 45, 75),
                               outline=self.colors["primary"])
        
        # Title bar with accent color
        self._draw_rounded_rect(draw, x - width//2, y, x + width//2, y + title_height,
                               radius=20, fill=self.colors["primary"])
        
        # Title text centered
        clean_title = self._clean_content_text(title) or title
        if title_font:
            bbox = draw.textbbox((0, 0), clean_title, font=title_font)
            text_width = bbox[2] - bbox[0]
            draw.text((x - text_width//2, y + 15), clean_title,
                     fill=self.colors["text"], font=title_font)
        else:
            draw.text((x - len(clean_title) * 6, y + 18), clean_title, 
                     fill=self.colors["text"])
        
        # Draw each point as a card-style item
        for i, point in enumerate(clean_points):
            item_y = y + title_height + padding + i * item_height
            
            # Item background (subtle card effect)
            card_color = (45, 55, 90) if i != highlight_point else (60, 80, 130)
            self._draw_rounded_rect(draw, x - width//2 + 15, item_y,
                                   x + width//2 - 15, item_y + item_height - 10,
                                   radius=12, fill=card_color)
            
            # Number badge on left
            badge_color = self.colors["accent"] if i == highlight_point else self.colors["secondary"]
            badge_x = x - width//2 + 35
            badge_y = item_y + (item_height - 10) // 2
            draw.ellipse([badge_x - 15, badge_y - 15, badge_x + 15, badge_y + 15],
                        fill=badge_color)
            draw.text((badge_x - 6, badge_y - 10), str(i + 1), fill=(255, 255, 255))
            
            # Text content - properly aligned
            text_x = x - width//2 + 70
            text_max_width = width - 100
            
            # Wrap text to 2 lines max
            wrapped = self._wrap_text(point, text_max_width, font, draw, max_lines=2)
            lines = wrapped.split('\n')
            
            # Center text vertically in the card
            total_text_height = len(lines) * 28
            text_start_y = item_y + (item_height - 10 - total_text_height) // 2 + 5
            
            for line_idx, line in enumerate(lines):
                line_y = text_start_y + line_idx * 28
                if font:
                    draw.text((text_x, line_y), line, fill=self.colors["text"], font=font)
                else:
                    draw.text((text_x, line_y), line, fill=self.colors["text"])
        
        return height

    def _wrap_text(self, text: str, max_width: int, font, draw: ImageDraw, 
                   max_lines: int = 2) -> str:
        """Wrap text to fit within max_width pixels."""
        if not text:
            return ""
        
        words = text.split()
        if not words:
            return ""
        
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            
            # Get text width
            if font:
                try:
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    text_width = bbox[2] - bbox[0]
                except:
                    text_width = len(test_line) * 8
            else:
                text_width = len(test_line) * 8
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                
                if len(lines) >= max_lines:
                    break
        
        if current_line and len(lines) < max_lines:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _draw_rounded_rect(self, draw: ImageDraw, x1: int, y1: int, 
                          x2: int, y2: int, radius: int = 10,
                          fill=None, outline=None) -> None:
        """Draw rounded rectangle"""
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
        
        draw.ellipse([x1, y1, x1 + radius*2, y1 + radius*2], fill=fill)
        draw.ellipse([x2 - radius*2, y1, x2, y1 + radius*2], fill=fill)
        draw.ellipse([x1, y2 - radius*2, x1 + radius*2, y2], fill=fill)
        draw.ellipse([x2 - radius*2, y2 - radius*2, x2, y2], fill=fill)
        
        if outline:
            draw.arc([x1, y1, x1 + radius*2, y1 + radius*2], 180, 270, fill=outline, width=2)
            draw.arc([x2 - radius*2, y1, x2, y1 + radius*2], 270, 360, fill=outline, width=2)
            draw.arc([x1, y2 - radius*2, x1 + radius*2, y2], 90, 180, fill=outline, width=2)
            draw.arc([x2 - radius*2, y2 - radius*2, x2, y2], 0, 90, fill=outline, width=2)
            draw.line([(x1 + radius, y1), (x2 - radius, y1)], fill=outline, width=2)
            draw.line([(x1 + radius, y2), (x2 - radius, y2)], fill=outline, width=2)
            draw.line([(x1, y1 + radius), (x1, y2 - radius)], fill=outline, width=2)
            draw.line([(x2, y1 + radius), (x2, y2 - radius)], fill=outline, width=2)
    
    def _draw_arrow(self, draw: ImageDraw, x1: int, y1: int, 
                   x2: int, y2: int, color=None) -> None:
        """Draw arrow"""
        if color is None:
            color = self.colors["arrow"]
        
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
        
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_size = 12
        
        left_x = x2 - arrow_size * math.cos(angle - math.pi/6)
        left_y = y2 - arrow_size * math.sin(angle - math.pi/6)
        right_x = x2 - arrow_size * math.cos(angle + math.pi/6)
        right_y = y2 - arrow_size * math.sin(angle + math.pi/6)
        
        draw.polygon([(x2, y2), (int(left_x), int(left_y)), 
                     (int(right_x), int(right_y))], fill=color)
