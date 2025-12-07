"""
Audio Service for Explainer Videos
Handles text-to-speech using ElevenLabs API with proper text preprocessing
"""

import os
import re
import requests
from pathlib import Path
from typing import Optional, List, Dict
import tempfile


class AudioService:
    """
    Generates audio narration for explainer videos.
    Features:
    - ElevenLabs API (primary - high quality)
    - Proper text preprocessing to avoid spelling out words
    - gTTS fallback (free, lower quality)
    - Audio duration detection
    """
    
    def __init__(self):
        # Use separate ElevenLabs key for video generation (ELEVENLABS_VIDEO_API_KEY)
        # Falls back to main ELEVENLABS_API_KEY if video-specific key not set
        self.elevenlabs_key = os.getenv("ELEVENLABS_VIDEO_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
        self.output_dir = Path("generated_videos/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ElevenLabs settings - using a natural voice
        self.elevenlabs_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
        self.elevenlabs_model = "eleven_monolingual_v1"
        
        # Check available TTS options
        self.gtts_available = self._check_gtts()
        self.pyttsx3_available = self._check_pyttsx3()
        
        if self.elevenlabs_key:
            key_type = "video-specific" if os.getenv("ELEVENLABS_VIDEO_API_KEY") else "shared"
            print(f"✓ Audio Service: ElevenLabs API configured ({key_type} key)")
        elif self.gtts_available:
            print("✓ Audio Service: Using gTTS fallback")
        elif self.pyttsx3_available:
            print("✓ Audio Service: Using pyttsx3 offline fallback")
        else:
            print("⚠ Audio Service: No TTS available")
    
    def _check_gtts(self) -> bool:
        try:
            from gtts import gTTS
            return True
        except ImportError:
            return False
    
    def _check_pyttsx3(self) -> bool:
        try:
            import pyttsx3
            return True
        except ImportError:
            return False

    def preprocess_text_for_speech(self, text: str) -> str:
        """
        Preprocess text to ensure natural speech output.
        Fixes issues like spelling out words, handles abbreviations, etc.
        """
        if not text:
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'[*#_`~]', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
        
        # Remove scene/chapter markers like "1 INTRODUCTION" or "Chapter 12"
        text = re.sub(r'\b\d+\s+[A-Z]{2,}\b', '', text)  # "1 INTRODUCTION"
        text = re.sub(r'\bChapter\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bScene\s+\d+\b', '', text, flags=re.IGNORECASE)
        
        # Fix words that might be spelled out (like "I N T R O")
        # Pattern: single letters with spaces between them
        def fix_spelled_words(match):
            letters = match.group(0).replace(' ', '')
            return letters
        
        text = re.sub(r'\b([A-Z]\s+){2,}[A-Z]\b', fix_spelled_words, text)
        
        # Common abbreviations - expand for natural speech
        abbreviations = {
            r'\bNCERT\b': 'N.C.E.R.T.',
            r'\bDNA\b': 'D.N.A.',
            r'\bRNA\b': 'R.N.A.',
            r'\bATP\b': 'A.T.P.',
            r'\bpH\b': 'p.H.',
            r'\bAC\b': 'A.C.',
            r'\bDC\b': 'D.C.',
            r'\bEMF\b': 'E.M.F.',
            r'\bJEE\b': 'J.E.E.',
            r'\bNEET\b': 'N.E.E.T.',
            r'\bCBSE\b': 'C.B.S.E.',
            r'\bCO2\b': 'C.O.2',
            r'\bH2O\b': 'H.2.O',
            r'\bO2\b': 'O.2',
            r'\bNaCl\b': 'sodium chloride',
            r'\bHCl\b': 'H.C.L.',
            r'\bH2SO4\b': 'H.2.S.O.4',
        }
        
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Fix common text issues
        replacements = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'etcetera',
            'vs.': 'versus',
            '°C': 'degrees Celsius',
            '°F': 'degrees Fahrenheit',
            '→': 'leads to',
            '←': 'comes from',
            '↑': 'increases',
            '↓': 'decreases',
            '≈': 'approximately equals',
            '≠': 'is not equal to',
            '≤': 'is less than or equal to',
            '≥': 'is greater than or equal to',
            '∞': 'infinity',
            'π': 'pi',
            '²': ' squared',
            '³': ' cubed',
            '⁻': ' to the power of minus',
            '₂': ' 2',
            '₃': ' 3',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Clean up numbers with units
        text = re.sub(r'(\d+)\s*m/s', r'\1 meters per second', text)
        text = re.sub(r'(\d+)\s*km/h', r'\1 kilometers per hour', text)
        text = re.sub(r'(\d+)\s*kg', r'\1 kilograms', text)
        text = re.sub(r'(\d+)\s*cm', r'\1 centimeters', text)
        text = re.sub(r'(\d+)\s*mm', r'\1 millimeters', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Ensure proper sentence structure
        # Add periods if missing at end
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text

    def generate_audio(self, text: str, filename: str = None) -> Optional[str]:
        """
        Generate audio from text.
        
        Args:
            text: Text to convert to speech
            filename: Output filename (optional)
        
        Returns:
            Path to generated audio file, or None if failed
        """
        if not text or len(text.strip()) < 5:
            return None
        
        # Preprocess text for natural speech
        clean_text = self.preprocess_text_for_speech(text)
        
        if not clean_text:
            return None
        
        if not filename:
            import hashlib
            text_hash = hashlib.md5(clean_text.encode()).hexdigest()[:10]
            filename = f"narration_{text_hash}.mp3"
        
        output_path = self.output_dir / filename
        
        # Try ElevenLabs first
        if self.elevenlabs_key:
            result = self._generate_elevenlabs(clean_text, output_path)
            if result:
                return result
        
        # Fallback to gTTS
        if self.gtts_available:
            result = self._generate_gtts(clean_text, output_path)
            if result:
                return result
        
        # Fallback to pyttsx3
        if self.pyttsx3_available:
            result = self._generate_pyttsx3(clean_text, output_path)
            if result:
                return result
        
        return None
    
    def _generate_elevenlabs(self, text: str, output_path: Path) -> Optional[str]:
        """Generate audio using ElevenLabs API"""
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_key
            }
            
            data = {
                "text": text,
                "model_id": self.elevenlabs_model,
                "voice_settings": {
                    "stability": 0.6,  # Slightly more stable for educational content
                    "similarity_boost": 0.8,
                    "style": 0.4,  # Natural style
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=120)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✓ ElevenLabs audio: {output_path.name}")
                return str(output_path)
            else:
                print(f"ElevenLabs error: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"ElevenLabs failed: {e}")
            return None
    
    def _generate_gtts(self, text: str, output_path: Path) -> Optional[str]:
        """Generate audio using Google TTS"""
        try:
            from gtts import gTTS
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(str(output_path))
            
            print(f"✓ gTTS audio: {output_path.name}")
            return str(output_path)
            
        except Exception as e:
            print(f"gTTS failed: {e}")
            return None
    
    def _generate_pyttsx3(self, text: str, output_path: Path) -> Optional[str]:
        """Generate audio using pyttsx3"""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            
            print(f"✓ pyttsx3 audio: {output_path.name}")
            return str(output_path)
            
        except Exception as e:
            print(f"pyttsx3 failed: {e}")
            return None

    def generate_scene_audio(self, scenes: List[Dict]) -> Dict[int, str]:
        """
        Generate audio for all scenes in a script.
        
        Args:
            scenes: List of scene dictionaries with 'narration' key
        
        Returns:
            Dictionary mapping scene numbers to audio file paths
        """
        audio_files = {}
        
        for scene in scenes:
            scene_num = scene.get("scene_number", 0)
            narration = scene.get("narration", "")
            
            if narration:
                filename = f"scene_{scene_num:02d}.mp3"
                audio_path = self.generate_audio(narration, filename)
                
                if audio_path:
                    audio_files[scene_num] = audio_path
                    print(f"  Scene {scene_num}: Audio generated ({len(narration)} chars)")
        
        return audio_files
    
    def generate_full_narration(self, script: Dict) -> Optional[str]:
        """
        Generate a single audio file for the entire script.
        Better for sync and smoother transitions.
        """
        scenes = script.get("scenes", [])
        if not scenes:
            return None
        
        # Combine all narrations with natural pauses
        full_text_parts = []
        
        for scene in scenes:
            narration = scene.get("narration", "")
            if narration:
                # Add scene narration
                full_text_parts.append(narration)
                # Add a pause marker (will be handled by TTS)
                full_text_parts.append("...")
        
        full_text = " ".join(full_text_parts)
        
        # Generate single audio file
        filename = "full_narration.mp3"
        return self.generate_audio(full_text, filename)
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file in seconds"""
        if not audio_path or not os.path.exists(audio_path):
            return 0.0
        
        try:
            from mutagen.mp3 import MP3
            audio = MP3(audio_path)
            return audio.info.length
        except:
            pass
        
        try:
            from moviepy import AudioFileClip
            clip = AudioFileClip(audio_path)
            duration = clip.duration
            clip.close()
            return duration
        except:
            pass
        
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(audio_path)
            duration = clip.duration
            clip.close()
            return duration
        except:
            pass
        
        # Estimate based on file size (rough approximation)
        try:
            file_size = os.path.getsize(audio_path)
            # MP3 at 128kbps ≈ 16KB per second
            return file_size / 16000
        except:
            pass
        
        return 10.0  # Default fallback
    
    def estimate_speech_duration(self, text: str) -> float:
        """
        Estimate how long it will take to speak the text.
        Average speaking rate: ~150 words per minute
        """
        if not text:
            return 0.0
        
        words = len(text.split())
        # 150 words per minute = 2.5 words per second
        duration = words / 2.5
        
        # Add time for pauses
        sentences = text.count('.') + text.count('!') + text.count('?')
        duration += sentences * 0.3  # 0.3 seconds pause per sentence
        
        return max(3.0, duration)  # Minimum 3 seconds
