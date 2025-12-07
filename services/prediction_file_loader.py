"""
Prediction File Loader
Loads pre-generated prediction papers from JSON files in the previous_papers/NEET/ folder.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

BASE_DIR = Path(__file__).parent.parent.absolute()
NEET_PAPERS_DIR = BASE_DIR / 'previous_papers' / 'NEET'


class PredictionFileLoader:
    """Loads pre-generated prediction papers from JSON files."""
    
    def __init__(self, papers_dir: Path = None):
        """
        Initialize the loader.
        
        Args:
            papers_dir: Directory containing prediction JSON files
        """
        self.papers_dir = papers_dir or NEET_PAPERS_DIR
    
    def check_prediction_file_exists(self, year: int, subject: str = None) -> bool:
        """
        Check if a pre-generated prediction file exists.
        
        Args:
            year: Year of prediction (e.g., 2026)
            subject: Subject name (Physics, Chemistry, Biology) or None for complete paper
        
        Returns:
            True if file exists, False otherwise
        """
        if subject:
            # Subject-specific file
            filename = f"{year}_predicted_{subject.lower()}.json"
        else:
            # Complete paper file
            filename = f"{year}_predicted_complete.json"
        
        filepath = self.papers_dir / filename
        return filepath.exists()
    
    def load_prediction_file(self, year: int, subject: str) -> Optional[Dict[str, Any]]:
        """
        Load a pre-generated prediction file.
        
        Args:
            year: Year of prediction (e.g., 2026)
            subject: Subject name (Physics, Chemistry, Biology)
        
        Returns:
            Dictionary with paper_info and questions, or None if file doesn't exist
        """
        filename = f"{year}_predicted_{subject.lower()}.json"
        filepath = self.papers_dir / filename
        
        if not filepath.exists():
            print(f"Prediction file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✓ Loaded pre-generated prediction from: {filename}")
            return data
        
        except Exception as e:
            print(f"Error loading prediction file {filename}: {e}")
            return None
    
    def load_complete_prediction(self, year: int) -> Optional[Dict[str, Any]]:
        """
        Load a complete NEET prediction paper (all subjects).
        
        Args:
            year: Year of prediction (e.g., 2026)
        
        Returns:
            Dictionary with complete paper or None if files don't exist
        """
        # Try to load complete file first
        complete_filename = f"{year}_predicted_complete.json"
        complete_filepath = self.papers_dir / complete_filename
        
        if complete_filepath.exists():
            try:
                with open(complete_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✓ Loaded complete prediction from: {complete_filename}")
                return data
            except Exception as e:
                print(f"Error loading complete prediction: {e}")
        
        # Otherwise, try to combine individual subject files
        subjects = ['Physics', 'Chemistry', 'Biology']
        all_questions = []
        subject_papers = {}
        
        for subject in subjects:
            subject_data = self.load_prediction_file(year, subject)
            if not subject_data:
                print(f"Missing prediction file for {subject} {year}")
                return None
            
            questions = subject_data.get('questions', [])
            subject_papers[subject] = {
                'questions': questions,
                'confidence': subject_data.get('paper_info', {}).get('confidence', 0.75)
            }
            all_questions.extend(questions)
        
        # Get pattern for the year
        from services.question_predictor import get_neet_pattern
        pattern = get_neet_pattern(year)
        
        # Calculate overall confidence
        overall_confidence = sum(p['confidence'] for p in subject_papers.values()) / 3
        
        # Build complete paper structure
        physics_config = pattern['subjects']['Physics']
        chemistry_config = pattern['subjects']['Chemistry']
        biology_config = pattern['subjects']['Biology']
        
        complete_paper = {
            'paper_info': {
                'exam_type': 'NEET_PREDICTED',
                'year': year,
                'total_questions': pattern['total_questions'],
                'to_attempt': pattern['to_attempt'],
                'duration_minutes': pattern['duration_minutes'],
                'total_marks': pattern['total_marks'],
                'prediction_confidence': overall_confidence,
                'based_on_years': 5,
                'pattern_type': 'all_compulsory' if year >= 2026 else 'choice',
                'source': 'pre_generated_files',
                'subjects': {
                    'Physics': {
                        'questions': f"{physics_config['question_range'][0]}-{physics_config['question_range'][1]}",
                        'to_attempt': physics_config['to_attempt'],
                        'confidence': subject_papers['Physics']['confidence']
                    },
                    'Chemistry': {
                        'questions': f"{chemistry_config['question_range'][0]}-{chemistry_config['question_range'][1]}",
                        'to_attempt': chemistry_config['to_attempt'],
                        'confidence': subject_papers['Chemistry']['confidence']
                    },
                    'Biology': {
                        'questions': f"{biology_config['question_range'][0]}-{biology_config['question_range'][1]}",
                        'to_attempt': biology_config['to_attempt'],
                        'confidence': subject_papers['Biology']['confidence']
                    }
                }
            },
            'questions': all_questions
        }
        
        print(f"✓ Combined {len(all_questions)} questions from individual subject files")
        return complete_paper
    
    def list_available_predictions(self) -> Dict[int, list]:
        """
        List all available pre-generated predictions.
        
        Returns:
            Dictionary mapping year to list of available subjects
        """
        available = {}
        
        if not self.papers_dir.exists():
            return available
        
        # Scan for prediction files
        for filepath in self.papers_dir.glob("*_predicted_*.json"):
            filename = filepath.stem
            parts = filename.split('_')
            
            if len(parts) >= 3:
                try:
                    year = int(parts[0])
                    subject = parts[2].capitalize()
                    
                    if year not in available:
                        available[year] = []
                    
                    if subject not in available[year]:
                        available[year].append(subject)
                
                except ValueError:
                    continue
        
        return available
    
    def get_prediction_info(self, year: int, subject: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a prediction file without loading all questions.
        
        Args:
            year: Year of prediction
            subject: Subject name
        
        Returns:
            Dictionary with paper_info only, or None if file doesn't exist
        """
        filename = f"{year}_predicted_{subject.lower()}.json"
        filepath = self.papers_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                'paper_info': data.get('paper_info', {}),
                'question_count': len(data.get('questions', [])),
                'source_file': filename
            }
        
        except Exception as e:
            print(f"Error reading prediction info: {e}")
            return None


def get_prediction_loader():
    """Get singleton instance of PredictionFileLoader."""
    if not hasattr(get_prediction_loader, '_instance'):
        get_prediction_loader._instance = PredictionFileLoader()
    return get_prediction_loader._instance
