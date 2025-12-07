"""
AI-Powered Question Paper Predictor
Predicts future NEET questions based on previous year patterns and NCERT content.
"""

import sqlite3
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict
import json

BASE_DIR = Path(__file__).parent.parent.absolute()
DB_PATH = BASE_DIR / 'guruai.db'

# NEET Exam Pattern (Official - Exact Format)
# Updated for NEET 2026 onwards - all questions compulsory, no sections
def get_neet_pattern(year=2026):
    """Get NEET pattern based on year."""
    if year >= 2026:
        # New pattern (2026 onwards)
        return {
            'total_questions': 180,
            'to_attempt': 180,  # All compulsory
            'duration_minutes': 180,  # 3 hours
            'total_marks': 720,
            'subjects': {
                'Physics': {
                    'question_range': (1, 45),
                    'total_questions': 45,
                    'to_attempt': 45,  # All compulsory
                    'sections': None,  # No sections
                    'class_11_weightage': 0.45,
                    'class_12_weightage': 0.55,
                    'difficulty': {
                        'easy': 0.20,
                        'medium': 0.50,
                        'hard': 0.30
                    }
                },
                'Chemistry': {
                    'question_range': (46, 90),
                    'total_questions': 45,
                    'to_attempt': 45,  # All compulsory
                    'sections': None,  # No sections
                    'class_11_weightage': 0.45,
                    'class_12_weightage': 0.55,
                    'difficulty': {
                        'easy': 0.25,
                        'medium': 0.50,
                        'hard': 0.25
                    }
                },
                'Biology': {
                    'question_range': (91, 180),
                    'total_questions': 90,
                    'to_attempt': 90,  # All compulsory
                    'sections': None,  # No sections
                    'class_11_weightage': 0.50,
                    'class_12_weightage': 0.50,
                    'difficulty': {
                        'easy': 0.30,
                        'medium': 0.50,
                        'hard': 0.20
                    }
                }
            }
        }
    else:
        # Legacy pattern (2025 and earlier)
        return {
            'total_questions': 200,
            'to_attempt': 180,
            'duration_minutes': 200,  # 3 hours 20 minutes
            'total_marks': 720,
            'subjects': {
                'Physics': {
                    'question_range': (1, 50),
                    'total_questions': 50,
                    'to_attempt': 45,
                    'sections': {
                        'Section A': {'questions': 35, 'compulsory': True},
                        'Section B': {'questions': 15, 'attempt': 10}
                    },
                    'class_11_weightage': 0.45,
                    'class_12_weightage': 0.55,
                    'difficulty': {
                        'easy': 0.20,
                        'medium': 0.50,
                        'hard': 0.30
                    }
                },
                'Chemistry': {
                    'question_range': (51, 100),
                    'total_questions': 50,
                    'to_attempt': 45,
                    'sections': {
                        'Section A': {'questions': 35, 'compulsory': True},
                        'Section B': {'questions': 15, 'attempt': 10}
                    },
                    'class_11_weightage': 0.45,
                    'class_12_weightage': 0.55,
                    'difficulty': {
                        'easy': 0.25,
                        'medium': 0.50,
                        'hard': 0.25
                    }
                },
                'Biology': {
                    'question_range': (101, 200),
                    'total_questions': 100,
                    'to_attempt': 90,
                    'sections': {
                        'Botany Section A': {'questions': 35, 'compulsory': True},
                        'Botany Section B': {'questions': 15, 'attempt': 10},
                        'Zoology Section A': {'questions': 35, 'compulsory': True},
                        'Zoology Section B': {'questions': 15, 'attempt': 10}
                    },
                    'class_11_weightage': 0.50,
                    'class_12_weightage': 0.50,
                    'difficulty': {
                        'easy': 0.30,
                        'medium': 0.50,
                        'hard': 0.20
                    }
                }
            }
        }

# Default to 2026 pattern
NEET_PATTERN = get_neet_pattern(2026)

# High-weightage chapters (based on historical analysis)
HIGH_WEIGHTAGE_CHAPTERS = {
    'Physics': [
        'Current Electricity',
        'Electrostatics',
        'Magnetism',
        'Optics',
        'Modern Physics',
        'Thermodynamics',
        'Mechanics',
        'Waves'
    ],
    'Chemistry': [
        'Chemical Bonding',
        'Thermodynamics',
        'Equilibrium',
        'Electrochemistry',
        'Organic Chemistry',
        'Coordination Compounds',
        'p-Block Elements',
        'Biomolecules'
    ],
    'Biology': [
        'Genetics',
        'Ecology',
        'Human Physiology',
        'Plant Physiology',
        'Cell Biology',
        'Reproduction',
        'Evolution',
        'Biotechnology'
    ]
}


class QuestionPredictor:
    """Predicts future NEET questions based on patterns."""
    
    def __init__(self, db_path: str = None, user_id: str = None):
        """
        Initialize predictor.
        
        Args:
            db_path: Path to database
            user_id: User ID for tier-based access checks
        """
        self.db_path = db_path or DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.user_id = user_id
        
        # Import services for tier-based access (lazy import to avoid circular dependencies)
        self._feature_gate_service = None
        self._usage_tracker = None
    
    def analyze_previous_patterns(self, subject: str, years: int = 5) -> Dict[str, Any]:
        """
        Analyze patterns from previous years.
        
        Args:
            subject: Subject name
            years: Number of years to analyze
        
        Returns:
            Dictionary with pattern analysis
        
        Raises:
            PermissionError: If user doesn't have access to chapter analysis
        """
        # Check access to chapter analysis
        if self.user_id:
            allowed, message = self._check_prediction_access('chapter_analysis')
            if not allowed:
                raise PermissionError(message)
        # Get questions from previous years
        self.cursor.execute("""
            SELECT topic, topic as chapter, difficulty, year
            FROM previous_year_questions
            WHERE subject = ? AND exam_type = 'NEET'
            ORDER BY year DESC
            LIMIT ?
        """, (subject, years * 50))
        
        questions = self.cursor.fetchall()
        
        if not questions:
            return self._get_default_pattern(subject)
        
        # Analyze topics
        topics = [q[0] for q in questions if q[0]]
        topic_freq = Counter(topics)
        
        # Analyze chapters
        chapters = [q[1] for q in questions if q[1]]
        chapter_freq = Counter(chapters)
        
        # Analyze difficulty
        difficulties = [q[2] for q in questions if q[2]]
        difficulty_dist = Counter(difficulties)
        
        return {
            'top_topics': topic_freq.most_common(20),
            'top_chapters': chapter_freq.most_common(15),
            'difficulty_distribution': dict(difficulty_dist),
            'total_analyzed': len(questions)
        }
    
    def _get_default_pattern(self, subject: str) -> Dict[str, Any]:
        """Get default pattern if no data available."""
        return {
            'top_topics': [],
            'top_chapters': [(ch, 5) for ch in HIGH_WEIGHTAGE_CHAPTERS.get(subject, [])],
            'difficulty_distribution': NEET_PATTERN[subject]['difficulty'],
            'total_analyzed': 0
        }
    
    def get_ncert_coverage(self, subject: str) -> List[Dict[str, Any]]:
        """
        Get NCERT content coverage for prediction.
        
        Args:
            subject: Subject name
        
        Returns:
            List of NCERT topics with metadata
        """
        # This would query your NCERT content database
        # For now, return structure
        return []
    
    def predict_complete_neet_paper(
        self,
        year: int = 2026,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Predict a complete NEET paper.
        
        For 2026+: 180 questions (45 Physics, 45 Chemistry, 90 Biology)
        For 2025 and earlier: 200 questions (50 Physics, 50 Chemistry, 100 Biology)
        
        Args:
            year: Year to predict for
            use_ai: Whether to use AI for question generation
        
        Returns:
            Complete predicted NEET paper
        
        Raises:
            PermissionError: If user doesn't have access to this feature
        """
        # Check access to complete paper prediction
        allowed, message = self._check_prediction_access('complete_paper_prediction')
        if not allowed:
            raise PermissionError(message)
        
        # Check and increment prediction limit
        allowed, message = self._check_and_increment_prediction_limit()
        if not allowed:
            raise PermissionError(message)
        
        # Get pattern for the year
        pattern = get_neet_pattern(year)
        total_q = pattern['total_questions']
        
        print(f"\n{'='*60}")
        print(f"Predicting Complete NEET {year} Paper ({total_q} Questions)")
        print('='*60)
        
        # Check if pre-generated prediction files exist
        from services.prediction_file_loader import get_prediction_loader
        loader = get_prediction_loader()
        
        pre_generated = loader.load_complete_prediction(year)
        if pre_generated:
            print(f"✓ Using pre-generated prediction files for NEET {year}")
            return pre_generated
        
        all_questions = []
        subject_papers = {}
        
        # Generate for each subject
        for subject in ['Physics', 'Chemistry', 'Biology']:
            print(f"\nGenerating {subject}...")
            patterns = self.analyze_previous_patterns(subject)
            exam_pattern = NEET_PATTERN['subjects'][subject]
            
            questions = self._generate_predicted_questions(
                subject=subject,
                patterns=patterns,
                exam_pattern=exam_pattern,
                use_ai=use_ai
            )
            
            subject_papers[subject] = {
                'questions': questions,
                'confidence': self._calculate_confidence(patterns)
            }
            
            all_questions.extend(questions)
            print(f"✓ Generated {len(questions)} {subject} questions")
        
        # Calculate overall confidence
        overall_confidence = sum(p['confidence'] for p in subject_papers.values()) / 3
        
        # Get pattern info for response
        pattern = get_neet_pattern(year)
        physics_config = pattern['subjects']['Physics']
        chemistry_config = pattern['subjects']['Chemistry']
        biology_config = pattern['subjects']['Biology']
        
        paper = {
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
        
        print(f"\n{'='*60}")
        print(f"✓ Complete NEET {year} Paper Generated!")
        print(f"Total Questions: {len(all_questions)}")
        print(f"Pattern: {'All Compulsory' if year >= 2026 else 'With Choice'}")
        print(f"Overall Confidence: {overall_confidence*100:.0f}%")
        print('='*60)
        
        return paper
    
    def predict_question_paper(
        self,
        subject: str,
        year: int = 2026,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Predict a future NEET question paper for a single subject.
        
        Args:
            subject: Subject name (Physics, Chemistry, Biology)
            year: Year to predict for
            use_ai: Whether to use AI for question generation
        
        Returns:
            Predicted question paper
        
        Raises:
            PermissionError: If user doesn't have access to this feature
        """
        # Check access to prediction insights (basic prediction feature)
        allowed, message = self._check_prediction_access('prediction_insights')
        if not allowed:
            raise PermissionError(message)
        
        # Check and increment prediction limit
        allowed, message = self._check_and_increment_prediction_limit()
        if not allowed:
            raise PermissionError(message)
        
        print(f"\nPredicting NEET {year} {subject} paper...")
        
        # Check if pre-generated prediction file exists
        from services.prediction_file_loader import get_prediction_loader
        loader = get_prediction_loader()
        
        pre_generated = loader.load_prediction_file(year, subject)
        if pre_generated:
            print(f"✓ Using pre-generated prediction file for {subject} {year}")
            
            # Get exam pattern for metadata
            exam_pattern = NEET_PATTERN['subjects'][subject]
            
            # Enhance with additional metadata
            paper = {
                'paper_info': {
                    'exam_type': 'NEET_PREDICTED',
                    'year': year,
                    'subject': subject,
                    'question_count': len(pre_generated.get('questions', [])),
                    'to_attempt': exam_pattern['to_attempt'],
                    'duration_minutes': 60,
                    'total_marks': exam_pattern['to_attempt'] * 4,
                    'prediction_confidence': pre_generated.get('paper_info', {}).get('confidence', 0.80),
                    'based_on_years': 5,
                    'source': 'pre_generated_file',
                    'metadata': {
                        'high_weightage_chapters': HIGH_WEIGHTAGE_CHAPTERS[subject]
                    }
                },
                'questions': pre_generated.get('questions', [])
            }
            
            return paper
        
        # Analyze patterns
        patterns = self.analyze_previous_patterns(subject)
        
        # Get exam pattern
        exam_pattern = NEET_PATTERN['subjects'][subject]
        
        # Generate questions based on pattern
        questions = self._generate_predicted_questions(
            subject=subject,
            patterns=patterns,
            exam_pattern=exam_pattern,
            use_ai=use_ai
        )
        
        paper = {
            'paper_info': {
                'exam_type': 'NEET_PREDICTED',
                'year': year,
                'subject': subject,
                'question_count': len(questions),
                'to_attempt': exam_pattern['to_attempt'],
                'duration_minutes': 60,
                'total_marks': exam_pattern['to_attempt'] * 4,
                'prediction_confidence': self._calculate_confidence(patterns),
                'based_on_years': 5,
                'metadata': {
                    'patterns_analyzed': patterns,
                    'high_weightage_chapters': HIGH_WEIGHTAGE_CHAPTERS[subject]
                }
            },
            'questions': questions
        }
        
        return paper
    
    def _generate_predicted_questions(
        self,
        subject: str,
        patterns: Dict[str, Any],
        exam_pattern: Dict[str, Any],
        use_ai: bool
    ) -> List[Dict[str, Any]]:
        """Generate predicted questions based on patterns using real questions from database."""
        questions = []
        
        # Get difficulty distribution
        difficulty_dist = exam_pattern['difficulty']
        total_questions = exam_pattern['total_questions']
        
        # Calculate questions per difficulty
        easy_count = int(total_questions * difficulty_dist['easy'])
        medium_count = int(total_questions * difficulty_dist['medium'])
        hard_count = total_questions - easy_count - medium_count
        
        # Get top chapters from patterns
        top_chapters = [ch for ch, _ in patterns['top_chapters'][:15]]
        if not top_chapters:
            top_chapters = HIGH_WEIGHTAGE_CHAPTERS[subject]
        
        # Try to fetch real questions from database
        q_num = 1
        
        # Fetch more questions than needed so we can filter out incomplete ones and image-based questions
        # Many questions have incomplete options or image references in the database
        # Fetch 10x to account for heavy filtering
        self.cursor.execute("""
            SELECT id, question_text, option_a, option_b, option_c, option_d,
                   correct_answer, explanation, topic as chapter, topic, difficulty
            FROM previous_year_questions
            WHERE subject = ? AND exam_type = 'NEET'
            ORDER BY RANDOM()
            LIMIT ?
        """, (subject, total_questions * 10))  # Fetch 10x to account for filtering
        
        real_questions = self.cursor.fetchall()
        
        print(f"Fetched {len(real_questions)} questions from database for filtering")
        
        # Distribute questions across difficulty levels
        difficulty_labels = []
        difficulty_labels.extend(['easy'] * easy_count)
        difficulty_labels.extend(['medium'] * medium_count)
        difficulty_labels.extend(['hard'] * hard_count)
        
        questions_added = 0
        
        # Try to use real questions first
        if real_questions:
            for q_data in real_questions:
                # Stop if we have enough questions
                if questions_added >= total_questions:
                    break
                    
                q_id, q_text, opt_a, opt_b, opt_c, opt_d, correct, solution, chapter, topic, diff = q_data
                
                # Skip questions with image/diagram references
                image_indicators = [
                    'diagram', 'figure', 'image', 'graph', 'chart',
                    'shown below', 'given below', 'shown above', 'given above',
                    'refer to', 'see the', 'observe the', 'look at',
                    '.png', '.jpg', '.jpeg', '.gif', 'PAGE', '---PAGE',
                    'A/Question (PDF)', 'refer the diagram'
                ]
                
                # Check question text and options for image references
                text_to_check = (q_text or '').lower()
                has_image = any(indicator.lower() in text_to_check for indicator in image_indicators)
                
                if has_image:
                    continue  # Skip this question
                
                # Build options list from individual columns
                options = []
                for opt in [opt_a, opt_b, opt_c, opt_d]:
                    if opt and opt.strip():
                        opt_text = opt.strip()
                        # Also check options for image references
                        if any(indicator.lower() in opt_text.lower() for indicator in image_indicators):
                            has_image = True
                            break
                        options.append(opt_text)
                
                # Skip if any option has image reference
                if has_image:
                    continue
                
                # Skip questions with less than 3 valid options (incomplete data)
                if len(options) < 3:
                    continue
                
                # Ensure we have exactly 4 options
                while len(options) < 4:
                    options.append(f'Option {chr(65 + len(options))}')
                
                # Assign difficulty label
                assigned_difficulty = difficulty_labels[questions_added] if questions_added < len(difficulty_labels) else 'medium'
                
                # Determine correct answer
                # If correct answer exists and matches an option, use it
                # Otherwise, mark as "Not specified"
                correct_answer_display = 'Not specified'
                if correct and correct.strip():
                    # Check if correct answer matches any option
                    for idx, opt in enumerate(options[:4]):
                        if opt.strip() == correct.strip():
                            correct_answer_display = correct
                            break
                
                # Create a better default solution if none exists
                if not solution or solution.strip() == '':
                    chapter_info = chapter or topic or subject
                    solution = (
                        f"This question is from {chapter_info}. "
                        f"To solve this question:\n"
                        f"1. Review the key concepts from {chapter_info} in your NCERT textbook\n"
                        f"2. Identify the relevant formulas or principles\n"
                        f"3. Apply the concepts systematically\n"
                        f"4. Verify your answer by checking units and reasonableness\n\n"
                        f"For detailed explanation, refer to NCERT {subject} textbook."
                    )
                
                question = {
                    'question_number': q_num,
                    'question_id': q_id,
                    'question_text': q_text or f'[PREDICTED] Question from {chapter or topic or subject}',
                    'question_type': 'MCQ',
                    'options': options[:4],  # Take only first 4 options
                    'correct_answer': correct_answer_display,
                    'solution': solution,
                    'difficulty': assigned_difficulty,
                    'topic': topic or chapter or subject,
                    'chapter': chapter or topic or subject,
                    'marks': 4,
                    'negative_marks': -1,
                    'prediction_score': self._calculate_question_score(chapter or topic or subject, patterns),
                    'is_predicted': True
                }
                
                questions.append(question)
                q_num += 1
                questions_added += 1
        
        # Fill remaining with predicted questions if we don't have enough real ones
        if questions_added < total_questions:
            remaining = total_questions - questions_added
            print(f"Note: Only {questions_added} complete questions found in database. Generating {remaining} predicted questions...")
            
            # Create more realistic predicted questions with better templates
            question_templates = {
                'easy': [
                    'Which of the following is correct regarding {topic}?',
                    'What is the primary function of {topic}?',
                    'Identify the correct statement about {topic}:',
                    'Which property is characteristic of {topic}?'
                ],
                'medium': [
                    'In the context of {topic}, which statement is most accurate?',
                    'Consider the following statements about {topic}. Which is correct?',
                    'What is the relationship between {topic} and its key properties?',
                    'Which mechanism best explains {topic}?'
                ],
                'hard': [
                    'Analyze the following scenario related to {topic}. What conclusion can be drawn?',
                    'Given the principles of {topic}, which outcome is most likely?',
                    'Compare and contrast the aspects of {topic}. Which statement is valid?',
                    'Apply the concepts of {topic} to solve the following problem:'
                ]
            }
            
            for _ in range(remaining):
                assigned_difficulty = difficulty_labels[questions_added] if questions_added < len(difficulty_labels) else 'medium'
                chapter = random.choice(top_chapters) if top_chapters else subject
                
                # Select a template based on difficulty
                templates = question_templates.get(assigned_difficulty, question_templates['medium'])
                question_text = random.choice(templates).format(topic=chapter)
                
                question = {
                    'question_number': q_num,
                    'question_text': f'[PREDICTED] {question_text}',
                    'question_type': 'MCQ',
                    'options': [
                        f'Statement related to {chapter} - Option A',
                        f'Statement related to {chapter} - Option B',
                        f'Statement related to {chapter} - Option C',
                        f'Statement related to {chapter} - Option D'
                    ],
                    'correct_answer': 'Refer to NCERT textbook',
                    'solution': f'This is a predicted question based on NEET patterns for {chapter}. To solve:\n1. Review {chapter} concepts from NCERT {subject} textbook\n2. Understand the key principles and formulas\n3. Apply the concepts systematically\n4. Practice similar questions from previous years\n\nDifficulty: {assigned_difficulty.capitalize()}',
                    'difficulty': assigned_difficulty,
                    'topic': chapter,
                    'chapter': chapter,
                    'marks': 4,
                    'negative_marks': -1,
                    'prediction_score': self._calculate_question_score(chapter, patterns),
                    'is_predicted': True,
                    'note': 'This is a placeholder question. For actual practice, refer to NCERT textbook and previous year papers.'
                }
                
                questions.append(question)
                q_num += 1
                questions_added += 1
        
        return questions
    
    def _calculate_question_score(self, chapter: str, patterns: Dict[str, Any]) -> float:
        """Calculate prediction confidence score for a question."""
        # Check if chapter appears in top chapters
        top_chapters = dict(patterns['top_chapters'])
        
        if chapter in top_chapters:
            frequency = top_chapters[chapter]
            max_frequency = max(top_chapters.values()) if top_chapters else 1
            return min(frequency / max_frequency, 1.0)
        
        return 0.5  # Default score
    
    def _calculate_confidence(self, patterns: Dict[str, Any]) -> float:
        """Calculate overall prediction confidence."""
        if patterns['total_analyzed'] == 0:
            return 0.5
        
        # Confidence based on data availability
        if patterns['total_analyzed'] >= 200:
            return 0.9
        elif patterns['total_analyzed'] >= 100:
            return 0.8
        elif patterns['total_analyzed'] >= 50:
            return 0.7
        else:
            return 0.6
    
    def generate_smart_paper(
        self,
        subject: str,
        focus_chapters: List[str] = None,
        difficulty_level: str = 'mixed'
    ) -> Dict[str, Any]:
        """
        Generate a smart practice paper based on weak areas.
        
        Args:
            subject: Subject name
            focus_chapters: Specific chapters to focus on
            difficulty_level: 'easy', 'medium', 'hard', or 'mixed'
        
        Returns:
            Smart practice paper
        
        Raises:
            PermissionError: If user doesn't have access to this feature
        """
        # Check access to smart paper generation
        allowed, message = self._check_prediction_access('smart_paper_generation')
        if not allowed:
            raise PermissionError(message)
        
        # Check and increment prediction limit
        allowed, message = self._check_and_increment_prediction_limit()
        if not allowed:
            raise PermissionError(message)
        
        patterns = self.analyze_previous_patterns(subject)
        
        if focus_chapters:
            # Filter patterns to focus chapters
            filtered_chapters = [(ch, freq) for ch, freq in patterns['top_chapters'] if ch in focus_chapters]
            patterns['top_chapters'] = filtered_chapters or patterns['top_chapters']
        
        # Adjust difficulty distribution
        if difficulty_level != 'mixed':
            exam_pattern = NEET_PATTERN['subjects'][subject].copy()
            exam_pattern['difficulty'] = {
                'easy': 1.0 if difficulty_level == 'easy' else 0.0,
                'medium': 1.0 if difficulty_level == 'medium' else 0.0,
                'hard': 1.0 if difficulty_level == 'hard' else 0.0
            }
        else:
            exam_pattern = NEET_PATTERN['subjects'][subject]
        
        # Try to generate AI questions first if Gemini is enabled
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        use_ai = os.getenv('USE_GEMINI', 'false').lower() == 'true'
        
        questions = []
        if use_ai:
            logger.info(f"Attempting to generate AI questions for {subject} using Gemini")
            
            # Check if Gemini is actually available
            try:
                from services.gemini_ai import get_gemini_ai
                gemini = get_gemini_ai()
                logger.info("✓ Gemini AI service initialized successfully")
            except Exception as e:
                logger.error(f"✗ Failed to initialize Gemini AI: {e}")
                logger.warning("Falling back to database questions only")
                use_ai = False
            
            # Determine chapters to use
            chapters_to_use = focus_chapters if focus_chapters else [ch for ch, _ in patterns['top_chapters'][:5]]
            
            if not chapters_to_use:
                # Fallback to high weightage chapters
                chapters_to_use = HIGH_WEIGHTAGE_CHAPTERS.get(subject, [])[:5]
            
            # Only proceed with AI generation if still enabled
            if use_ai:
                # Generate AI questions for selected chapters
                # Limit to 3 questions per API call to avoid token limits
                questions_per_chapter = max(2, min(3, exam_pattern['total_questions'] // len(chapters_to_use)))
                
                logger.info(f"Generating {questions_per_chapter} questions per chapter for: {chapters_to_use}")
                
                import time
                for idx, chapter in enumerate(chapters_to_use):
                    # Stop if we have enough questions
                    if len(questions) >= exam_pattern['total_questions']:
                        break
                    
                    # Add delay between chapters to avoid rate limiting (except for first chapter)
                    if idx > 0:
                        time.sleep(3)  # 3 second delay between chapters
                    
                    # Determine difficulty for this chapter
                    if difficulty_level == 'mixed':
                        # For mixed, just use medium difficulty to simplify
                        ai_questions = self._generate_ai_questions_from_content(
                            subject=subject,
                            chapter=chapter,
                            difficulty='medium',
                            count=questions_per_chapter
                        )
                        if ai_questions:  # Only extend if we got questions
                            questions.extend(ai_questions)
                            logger.info(f"✓ Added {len(ai_questions)} AI questions for {chapter}")
                        else:
                            logger.warning(f"✗ No AI questions generated for {chapter}")
                    else:
                        ai_questions = self._generate_ai_questions_from_content(
                            subject=subject,
                            chapter=chapter,
                            difficulty=difficulty_level,
                            count=questions_per_chapter
                        )
                        if ai_questions:  # Only extend if we got questions
                            questions.extend(ai_questions)
                            logger.info(f"✓ Added {len(ai_questions)} AI questions for {chapter}")
                        else:
                            logger.warning(f"✗ No AI questions generated for {chapter}")
                
                logger.info(f"Generated {len(questions)} AI questions total")
                
                # Trim to exact count needed
                questions = questions[:exam_pattern['total_questions']]
                
                # If we didn't get enough AI questions, fill with database questions
                if len(questions) < exam_pattern['total_questions']:
                    remaining = exam_pattern['total_questions'] - len(questions)
                    logger.warning(f"Only got {len(questions)} AI questions, filling {remaining} from database")
                    
                    # Create a modified exam pattern for the remaining questions
                    remaining_pattern = exam_pattern.copy()
                    remaining_pattern['total_questions'] = remaining
                    
                    # Adjust difficulty distribution to maintain proportions
                    if remaining < exam_pattern['total_questions']:
                        # Keep same proportions
                        remaining_pattern['difficulty'] = exam_pattern['difficulty'].copy()
                    
                    db_questions = self._generate_predicted_questions(
                        subject=subject,
                        patterns=patterns,
                        exam_pattern=remaining_pattern,
                        use_ai=False
                    )
                    questions.extend(db_questions)
                    logger.info(f"Added {len(db_questions)} questions from database/templates")
            else:
                logger.info("Skipping AI generation, will use database questions only")
        else:
            # Use database questions
            logger.info(f"Gemini disabled, using database questions for {subject}")
            questions = self._generate_predicted_questions(
                subject=subject,
                patterns=patterns,
                exam_pattern=exam_pattern,
                use_ai=False
            )
        
        # Renumber questions
        for idx, q in enumerate(questions, 1):
            q['question_number'] = idx
        
        # Calculate duration and marks based on question count
        question_count = len(questions)
        duration_minutes = question_count * 3  # 3 minutes per question
        total_marks = question_count * 4  # 4 marks per question (NEET pattern)
        
        return {
            'paper_info': {
                'exam_type': 'SMART_PRACTICE',
                'subject': subject,
                'question_count': question_count,
                'duration_minutes': duration_minutes,
                'total_marks': total_marks,
                'focus_chapters': focus_chapters or 'All',
                'difficulty_level': difficulty_level,
                'prediction_confidence': self._calculate_confidence(patterns),
                'metadata': {
                    'focus_chapters': focus_chapters or [],
                    'difficulty_level': difficulty_level,
                    'weak_areas': [ch for ch, _ in patterns['top_chapters'][:5]] if patterns.get('top_chapters') else []
                }
            },
            'questions': questions
        }
    
    def get_prediction_insights(self, subject: str) -> Dict[str, Any]:
        """
        Get insights for prediction.
        
        Args:
            subject: Subject name
        
        Returns:
            Prediction insights
        
        Raises:
            PermissionError: If user doesn't have access to this feature
        """
        # Check access to prediction insights
        allowed, message = self._check_prediction_access('prediction_insights')
        if not allowed:
            raise PermissionError(message)
        
        # Note: Getting insights doesn't count against prediction limit
        # Only actual paper generation does
        
        patterns = self.analyze_previous_patterns(subject)
        
        return {
            'subject': subject,
            'high_probability_chapters': [ch for ch, _ in patterns['top_chapters'][:10]],
            'recommended_focus': HIGH_WEIGHTAGE_CHAPTERS[subject],
            'difficulty_trend': patterns['difficulty_distribution'],
            'data_confidence': self._calculate_confidence(patterns),
            'total_questions_analyzed': patterns['total_analyzed']
        }
    
    @property
    def feature_gate_service(self):
        """Lazy load feature gate service."""
        if self._feature_gate_service is None:
            from services.feature_gate_service import get_feature_gate_service
            self._feature_gate_service = get_feature_gate_service()
        return self._feature_gate_service
    
    @property
    def usage_tracker(self):
        """Lazy load usage tracker."""
        if self._usage_tracker is None:
            from services.usage_tracker import get_usage_tracker
            self._usage_tracker = get_usage_tracker()
        return self._usage_tracker
    
    def _check_prediction_access(self, prediction_type: str) -> tuple[bool, str]:
        """
        Check if user has access to a prediction feature.
        
        Args:
            prediction_type: Type of prediction to check
            
        Returns:
            Tuple of (allowed: bool, message: str)
        """
        if not self.user_id:
            # No user ID provided, allow access (for backward compatibility)
            return (True, "Access granted")
        
        # Check feature access
        access_result = self.feature_gate_service.check_prediction_access(
            self.user_id,
            prediction_type
        )
        
        if not access_result.allowed:
            return (False, access_result.message)
        
        return (True, "Access granted")
    
    def _check_and_increment_prediction_limit(self) -> tuple[bool, str]:
        """
        Check if user has predictions remaining and increment counter.
        
        Returns:
            Tuple of (allowed: bool, message: str)
        """
        if not self.user_id:
            # No user ID provided, allow access (for backward compatibility)
            return (True, "Access granted")
        
        # Check prediction limit
        allowed, predictions_remaining = self.feature_gate_service.check_prediction_limit(self.user_id)
        
        if not allowed:
            return (False, "Monthly prediction limit reached. Upgrade your plan for more predictions.")
        
        # Increment prediction counter
        success = self.usage_tracker.track_prediction_usage(self.user_id)
        
        if not success:
            return (False, "Failed to track prediction usage")
        
        return (True, f"Prediction tracked. Remaining: {predictions_remaining - 1 if predictions_remaining > 0 else 'unlimited'}")
    
    def _generate_ai_questions_from_content(
        self,
        subject: str,
        chapter: str,
        difficulty: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate questions using Gemini AI from NCERT content.
        
        Args:
            subject: Subject name
            chapter: Chapter name
            difficulty: Difficulty level (easy, medium, hard)
            count: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            from services.gemini_ai import get_gemini_ai
            import os
            
            # Check if Gemini is enabled
            if not os.getenv('USE_GEMINI', 'false').lower() == 'true':
                return []
            
            gemini = get_gemini_ai()
            
            # Create prompt for question generation
            # Generate only 2 questions at a time to avoid token limits
            actual_count = min(count, 2)
            
            prompt = f"""Generate {actual_count} NEET MCQ for {subject} - {chapter} ({difficulty} level).

CRITICAL: Return valid JSON only. Use straight quotes ("), not smart quotes.

Format:
[{{"question_text":"Q?","options":["A","B","C","D"],"correct_answer":"A","explanation":"Why"}}]

Rules:
- Text only, no diagrams
- 4 options each
- Use straight quotes only
- Escape special characters
- Keep explanations under 40 words

Generate {actual_count} questions:"""

            # Generate questions with retry logic
            max_retries = 3
            result = None
            
            for attempt in range(max_retries):
                try:
                    result = gemini.generate(prompt, temperature=0.7, max_tokens=2048)
                    
                    if result['success'] and result['text']:
                        logger.info(f"✓ Attempt {attempt + 1} succeeded for {chapter}")
                        break  # Success, exit retry loop
                    else:
                        error_msg = result.get('error', 'Empty response')
                        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {chapter}: {error_msg}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(2)  # Wait 2 seconds before retry
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} exception for {chapter}: {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)
            
            # Check final result
            if not result or not result['success'] or not result['text']:
                logger.warning(f"All {max_retries} attempts failed for {chapter}, skipping AI generation for this chapter")
                return []  # Return empty, will be filled from database
            
            # Parse JSON response
            import re
            response_text = result['text'].strip()
            
            logger.info(f"Gemini response length: {len(response_text)} characters")
            
            # Remove markdown code blocks if present
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'^```\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()
            
            # Try to fix common JSON issues
            # Remove trailing commas before closing brackets
            response_text = re.sub(r',\s*}', '}', response_text)
            response_text = re.sub(r',\s*]', ']', response_text)
            # Fix smart quotes that might break JSON
            response_text = response_text.replace('"', '"').replace('"', '"')
            response_text = response_text.replace(''', "'").replace(''', "'")
            
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON array found in response. Response: {response_text[:500]}")
                return []
            
            json_str = json_match.group()
            
            # Try to fix incomplete JSON (common with token limits)
            # If JSON ends abruptly, try to close it
            if not json_str.rstrip().endswith(']'):
                # Find the last complete object
                last_complete = json_str.rfind('}')
                if last_complete > 0:
                    json_str = json_str[:last_complete + 1] + ']'
                    logger.warning("Fixed incomplete JSON by truncating to last complete object")
            
            try:
                questions_data = json.loads(json_str)
                if not isinstance(questions_data, list):
                    logger.error(f"Response is not a list: {type(questions_data)}")
                    return []
                logger.info(f"Successfully parsed {len(questions_data)} questions from Gemini")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Problematic JSON (first 500 chars): {json_str[:500]}")
                # Try to salvage what we can
                return []
            
            # Format questions
            questions = []
            for idx, q_data in enumerate(questions_data[:actual_count]):
                # Validate question data
                if not isinstance(q_data, dict):
                    logger.warning(f"Skipping invalid question data at index {idx}")
                    continue
                
                question_text = q_data.get('question_text', '')
                options = q_data.get('options', [])
                
                # Ensure we have valid data
                if not question_text or len(options) < 4:
                    logger.warning(f"Skipping incomplete question at index {idx}")
                    continue
                
                question = {
                    'question_number': idx + 1,
                    'question_id': f'ai_gen_{subject}_{chapter}_{idx}',
                    'question_text': question_text,
                    'question_type': 'MCQ',
                    'options': options[:4],  # Take only first 4 options
                    'correct_answer': q_data.get('correct_answer', options[0] if options else 'Not specified'),
                    'solution': q_data.get('explanation', 'Refer to NCERT textbook for detailed explanation.'),
                    'difficulty': difficulty,
                    'topic': chapter,
                    'chapter': chapter,
                    'marks': 4,
                    'negative_marks': -1,
                    'prediction_score': 0.85,  # High score for AI-generated questions
                    'is_predicted': True,
                    'is_ai_generated': True
                }
                questions.append(question)
            
            logger.info(f"Formatted {len(questions)} valid questions")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate AI questions: {e}", exc_info=True)
            return []
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


def main():
    """Test the predictor."""
    predictor = QuestionPredictor()
    
    # Predict for each subject
    for subject in ['Physics', 'Chemistry', 'Biology']:
        print(f"\n{'='*60}")
        print(f"Predicting NEET 2026 {subject}")
        print('='*60)
        
        # Get insights
        insights = predictor.get_prediction_insights(subject)
        print(f"\nHigh Probability Chapters:")
        for ch in insights['high_probability_chapters'][:5]:
            print(f"  - {ch}")
        
        print(f"\nData Confidence: {insights['data_confidence']*100:.0f}%")
        print(f"Questions Analyzed: {insights['total_questions_analyzed']}")
        
        # Generate predicted paper
        paper = predictor.predict_question_paper(subject, year=2026)
        print(f"\nGenerated {len(paper['questions'])} predicted questions")
        print(f"Prediction Confidence: {paper['paper_info']['prediction_confidence']*100:.0f}%")


if __name__ == '__main__':
    main()
