"""
Script Generator for Educational Explainer Videos
Generates detailed, structured scripts optimized for YouTube-style videos

Uses Gemini/OpenAI API to create educational video scripts with:
- Engaging introductions
- Clear explanations with visual cues
- Topic-specific content analysis
- Proper narration for TTS
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, List

# Try to import AI libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ScriptGenerator:
    """
    Generates educational video scripts optimized for explainer videos.
    Creates structured scripts with scenes, narration, and visual cues.
    """
    
    def __init__(self, content_path: str = None):
        """
        Initialize the script generator.
        
        Args:
            content_path: Optional path to content folder for reference material.
                         If not provided, scripts are generated using AI only.
        """
        self.gemini_model = None
        self.openai_client = None
        self._init_ai_clients()
        
        # Content path (optional - for reference material like NCERT)
        self.content_path = Path(content_path) if content_path else None
        
        # Check if content path exists
        if self.content_path and not self.content_path.exists():
            print(f"Note: Content path '{content_path}' not found. Using AI-only generation.")
            self.content_path = None
    
    def _init_ai_clients(self):
        """Initialize AI API clients"""
        # Support multiple Gemini API keys (comma-separated)
        api_keys_str = os.getenv("GEMINI_API_KEYS", "")
        gemini_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        
        # Fall back to single key if multiple keys not set
        if not gemini_keys:
            single_key = os.getenv("GEMINI_API_KEY", "")
            if single_key:
                gemini_keys = [single_key]
        
        gemini_key = gemini_keys[0] if gemini_keys else None
        if gemini_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
                print(f"✓ Script Generator: Gemini API ready ({len(gemini_keys)} key(s) available)")
            except Exception as e:
                print(f"Gemini init failed: {e}")
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                print("✓ Script Generator: OpenAI API ready")
            except Exception as e:
                print(f"OpenAI init failed: {e}")

    def get_reference_content(self, topic: str, subject: str = None) -> str:
        """
        Retrieve relevant reference content for a topic.
        Works with any content folder structure (not just NCERT).
        """
        if not self.content_path:
            return ""
        
        content_parts = []
        
        # Search in content folder for relevant files
        search_paths = [
            self.content_path / "extracted",
            self.content_path,
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for ext in ["*.txt", "*.md"]:
                for file_path in search_path.rglob(ext):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            if topic.lower() in text.lower():
                                clean_text = self._clean_content(text)
                                if clean_text:
                                    content_parts.append(clean_text[:2500])
                    except:
                        continue
        
        if content_parts:
            return "\n\n".join(content_parts[:3])
        
        return ""
    
    # Alias for backward compatibility
    def get_ncert_content(self, topic: str, subject: str = None) -> str:
        """Alias for get_reference_content (backward compatibility)."""
        return self.get_reference_content(topic, subject)
    
    def _clean_content(self, text: str) -> str:
        """Clean content by removing metadata and formatting issues."""
        # Remove common metadata patterns
        patterns_to_remove = [
            r'Page\s*\d+', r'Reprint\s*\d+', r'ISBN[\s\d-]+',
            r'NCERT.*?Publication', r'Copyright.*?\d{4}',
            r'Phone\s*:.*?\n', r'www\..*?\n', r'http\S+',
            r'printed on.*?\n', r'price.*?\n', r'rubber stamp.*?\n',
            r'Chief Editor.*?\n', r'Production.*?\n',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    # Alias for backward compatibility
    _clean_ncert_content = _clean_content
    
    def estimate_video_duration(self, topic: str, content_length: int) -> dict:
        """Estimate video duration based on topic and content."""
        base_duration = 60
        
        # Add time based on content
        words_estimate = content_length // 5
        narration_time = (words_estimate / 150) * 60
        
        # Complexity multiplier
        complexity = 1.0
        topic_lower = topic.lower()
        
        if any(kw in topic_lower for kw in ['quantum', 'organic', 'calculus', 'genetics', 'relativity']):
            complexity = 1.5
        elif any(kw in topic_lower for kw in ['cell', 'atom', 'equation', 'theorem', 'circuit']):
            complexity = 1.3
        
        total_duration = max(base_duration, narration_time * complexity)
        total_duration = min(total_duration, 480)  # Cap at 8 minutes
        
        return {
            "total_seconds": int(total_duration),
            "total_minutes": round(total_duration / 60, 1),
            "formatted": f"{int(total_duration // 60)}:{int(total_duration % 60):02d}"
        }

    def generate_script(self, topic: str, subject: str = None, 
                       target_duration: int = None) -> dict:
        """
        Generate a complete video script for the given topic.
        
        Returns structured script with scenes, narration, and visual cues.
        """
        # Get NCERT content
        ncert_content = self.get_ncert_content(topic, subject)
        
        # Estimate duration
        if not target_duration:
            duration_info = self.estimate_video_duration(topic, len(ncert_content))
            target_duration = duration_info["total_seconds"]
        else:
            duration_info = {
                "total_seconds": target_duration,
                "total_minutes": round(target_duration / 60, 1),
                "formatted": f"{int(target_duration // 60)}:{int(target_duration % 60):02d}"
            }
        
        # Generate script using AI
        script = self._generate_with_ai(topic, subject, ncert_content, target_duration)
        
        if script:
            script["duration_info"] = duration_info
            script["topic"] = topic
            script["subject"] = subject or self._detect_subject(topic)
            return script
        
        # Fallback to template
        return self._generate_fallback_script(topic, subject, ncert_content, target_duration)
    
    def _detect_subject(self, topic: str) -> str:
        """Detect subject from topic keywords."""
        topic_lower = topic.lower()
        
        physics_keywords = ['force', 'motion', 'electric', 'magnetic', 'wave', 'light', 
                          'energy', 'momentum', 'gravity', 'newton', 'current', 'voltage']
        chemistry_keywords = ['atom', 'molecule', 'bond', 'reaction', 'acid', 'base',
                            'element', 'compound', 'organic', 'periodic']
        biology_keywords = ['cell', 'dna', 'gene', 'plant', 'animal', 'photosynthesis',
                          'respiration', 'evolution', 'ecology', 'organ']
        maths_keywords = ['equation', 'theorem', 'function', 'calculus', 'algebra',
                        'geometry', 'trigonometry', 'probability', 'matrix']
        
        if any(kw in topic_lower for kw in physics_keywords):
            return "Physics"
        elif any(kw in topic_lower for kw in chemistry_keywords):
            return "Chemistry"
        elif any(kw in topic_lower for kw in biology_keywords):
            return "Biology"
        elif any(kw in topic_lower for kw in maths_keywords):
            return "Mathematics"
        
        return "Science"

    def _generate_with_ai(self, topic: str, subject: str, 
                         ncert_content: str, duration: int) -> Optional[dict]:
        """Generate script using AI API."""
        
        prompt = f"""You are an expert educational content creator for Indian students (CBSE/NCERT curriculum).

Create a detailed video script for a 2D animated explainer video.

TOPIC: {topic}
SUBJECT: {subject or 'Science'}
TARGET DURATION: {duration} seconds ({round(duration/60, 1)} minutes)

REFERENCE CONTENT:
{ncert_content[:3500] if ncert_content else 'Use your knowledge about this topic.'}

Generate a JSON script with this EXACT structure:
{{
    "title": "Clear, engaging title",
    "description": "Brief description",
    "total_scenes": 4-6,
    "scenes": [
        {{
            "scene_number": 1,
            "scene_title": "Introduction",
            "duration_seconds": 15,
            "narration": "Welcome! Today we'll explore {topic}. This is a fascinating concept that...",
            "key_points": ["Point 1", "Point 2"],
            "visual_type": "intro"
        }},
        {{
            "scene_number": 2,
            "scene_title": "What is {topic}?",
            "duration_seconds": 25,
            "narration": "Let me explain what {topic} actually means...",
            "key_points": ["Definition", "Key characteristic"],
            "visual_type": "concept"
        }},
        {{
            "scene_number": 3,
            "scene_title": "How it Works",
            "duration_seconds": 30,
            "narration": "Now let's understand how this works step by step...",
            "key_points": ["Step 1", "Step 2", "Step 3"],
            "visual_type": "process"
        }},
        {{
            "scene_number": 4,
            "scene_title": "Key Takeaways",
            "duration_seconds": 20,
            "narration": "To summarize what we learned today...",
            "key_points": ["Summary point 1", "Summary point 2"],
            "visual_type": "summary"
        }}
    ],
    "learning_objectives": ["Objective 1", "Objective 2"],
    "exam_tips": ["Tip 1", "Tip 2"]
}}

CRITICAL REQUIREMENTS:
1. Narration must be NATURAL SPEECH - write as if speaking to a student
2. DO NOT use abbreviations that will be spelled out (write "DNA" not "D.N.A")
3. Start narration with engaging hooks, not "In this scene..."
4. Each scene should have 2-4 key points for visual display
5. Total narration should match the target duration (150 words per minute)
6. Make content educational but engaging - like a friendly teacher
7. Include real-world examples and analogies

Return ONLY valid JSON."""

        # Try Gemini
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                json_text = response.text
                json_match = re.search(r'\{[\s\S]*\}', json_text)
                if json_match:
                    script = json.loads(json_match.group())
                    return self._validate_and_fix_script(script, topic, subject, duration)
            except Exception as e:
                print(f"Gemini script generation failed: {e}")
        
        # Try OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert educational video script writer. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                script = json.loads(response.choices[0].message.content)
                return self._validate_and_fix_script(script, topic, subject, duration)
            except Exception as e:
                print(f"OpenAI script generation failed: {e}")
        
        return None

    def _validate_and_fix_script(self, script: dict, topic: str, 
                                 subject: str, duration: int) -> dict:
        """Validate and fix script structure."""
        # Ensure required fields
        if "title" not in script:
            script["title"] = f"{topic} - Explained"
        
        if "scenes" not in script or not script["scenes"]:
            return None
        
        # Fix each scene
        for i, scene in enumerate(script["scenes"]):
            scene["scene_number"] = i + 1
            
            if "narration" not in scene or not scene["narration"]:
                scene["narration"] = f"Let's explore {scene.get('scene_title', 'this concept')}."
            
            if "key_points" not in scene:
                scene["key_points"] = []
            
            if "duration_seconds" not in scene:
                # Estimate from narration length
                words = len(scene["narration"].split())
                scene["duration_seconds"] = max(10, int(words / 2.5))
        
        script["total_scenes"] = len(script["scenes"])
        
        return script
    
    def _generate_fallback_script(self, topic: str, subject: str, 
                                  content: str, duration: int) -> dict:
        """Generate template-based script when AI is unavailable."""
        
        subject = subject or self._detect_subject(topic)
        num_scenes = max(4, min(6, duration // 25))
        scene_duration = duration // num_scenes
        
        # Extract key information from content
        key_facts = self._extract_key_facts(content, topic) if content else []
        
        scenes = [
            {
                "scene_number": 1,
                "scene_title": "Introduction",
                "duration_seconds": min(15, scene_duration),
                "narration": f"Welcome to this educational video! Today, we're going to learn about {topic}. This is an important concept in {subject} that you'll find in your NCERT textbook. Let's dive in and understand it together!",
                "key_points": [f"Understanding {topic}", f"{subject} fundamentals"],
                "visual_type": "intro"
            },
            {
                "scene_number": 2,
                "scene_title": f"What is {topic}?",
                "duration_seconds": scene_duration,
                "narration": self._generate_definition_narration(topic, subject, key_facts),
                "key_points": key_facts[:2] if key_facts else ["Definition", "Basic concept"],
                "visual_type": "concept"
            },
            {
                "scene_number": 3,
                "scene_title": "How it Works",
                "duration_seconds": scene_duration + 10,
                "narration": self._generate_explanation_narration(topic, subject, key_facts),
                "key_points": key_facts[2:5] if len(key_facts) > 2 else ["Step 1: Understand the basics", "Step 2: Apply the concept", "Step 3: Practice problems"],
                "visual_type": "process"
            },
            {
                "scene_number": 4,
                "scene_title": "Key Takeaways",
                "duration_seconds": min(20, scene_duration),
                "narration": f"Let's quickly recap what we learned about {topic}. Remember the key points we discussed, and make sure to practice related questions from your textbook. This topic is important for your exams!",
                "key_points": ["Review main concepts", "Practice questions"],
                "visual_type": "summary"
            }
        ]
        
        return {
            "title": f"{topic} - NCERT Explained",
            "description": f"Learn about {topic} from {subject} curriculum",
            "total_scenes": len(scenes),
            "scenes": scenes,
            "learning_objectives": [f"Understand {topic}", "Apply concepts to problems"],
            "exam_tips": ["Focus on understanding", "Practice numerical problems"],
            "generated_by": "fallback_template"
        }

    def _extract_key_facts(self, content: str, topic: str) -> List[str]:
        """Extract key facts from NCERT content."""
        if not content:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        key_facts = []
        
        topic_words = [w.lower() for w in topic.split() if len(w) > 2]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 150:
                continue
            
            # Check if sentence contains topic words
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in topic_words):
                # Clean up the sentence
                sentence = re.sub(r'\s+', ' ', sentence)
                key_facts.append(sentence)
        
        return key_facts[:6]
    
    def _generate_definition_narration(self, topic: str, subject: str, 
                                       key_facts: List[str]) -> str:
        """Generate narration for definition scene."""
        base = f"So, what exactly is {topic}? "
        
        if key_facts:
            base += key_facts[0] + " "
            if len(key_facts) > 1:
                base += "Additionally, " + key_facts[1].lower() + " "
        else:
            base += f"This is a fundamental concept in {subject} that helps us understand how things work in the natural world. "
        
        base += f"Understanding {topic} is essential for mastering {subject}."
        
        return base
    
    def _generate_explanation_narration(self, topic: str, subject: str,
                                        key_facts: List[str]) -> str:
        """Generate narration for explanation scene."""
        base = f"Now let's understand how {topic} actually works. "
        
        if len(key_facts) > 2:
            base += "First, " + key_facts[2].lower() + " "
            if len(key_facts) > 3:
                base += "Then, " + key_facts[3].lower() + " "
            if len(key_facts) > 4:
                base += "Finally, " + key_facts[4].lower() + " "
        else:
            base += f"The key to understanding {topic} is to break it down into simple steps. "
            base += "Think of it like building blocks - each concept builds on the previous one. "
        
        base += "This is how the process works in real life!"
        
        return base
