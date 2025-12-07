"""
Question Paper Generator for GuruAI - Generates custom practice tests and question papers.

This module creates practice question papers based on previous year JEE/NEET questions
and NCERT content. It supports topic-based selection, difficulty distribution, and
full-length test generation matching actual exam structures.

Features:
- Topic-based question selection
- Difficulty distribution control
- No duplicate questions in papers
- Answer keys with detailed solutions
- Full-length test generation (JEE/NEET structure)
- Custom paper generation with flexible parameters

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 3.4
"""

import logging
import random
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime

from sqlalchemy.orm import Session
from models.question import Question

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Exam structure definitions
EXAM_STRUCTURES = {
    'JEE_MAIN': {
        'total_questions': 90,
        'sections': {
            'Physics': {'count': 30, 'marks_per_question': 4},
            'Chemistry': {'count': 30, 'marks_per_question': 4},
            'Mathematics': {'count': 30, 'marks_per_question': 4}
        },
        'difficulty_distribution': {
            'easy': 0.30,
            'medium': 0.50,
            'hard': 0.20
        },
        'total_marks': 360,
        'duration_minutes': 180
    },
    'JEE_ADVANCED': {
        'total_questions': 54,
        'sections': {
            'Physics': {'count': 18, 'marks_per_question': 4},
            'Chemistry': {'count': 18, 'marks_per_question': 4},
            'Mathematics': {'count': 18, 'marks_per_question': 4}
        },
        'difficulty_distribution': {
            'easy': 0.20,
            'medium': 0.40,
            'hard': 0.40
        },
        'total_marks': 216,
        'duration_minutes': 180
    },
    'NEET': {
        'total_questions': 200,
        'sections': {
            'Physics': {'count': 50, 'marks_per_question': 4},
            'Chemistry': {'count': 50, 'marks_per_question': 4},
            'Biology': {'count': 100, 'marks_per_question': 4}
        },
        'difficulty_distribution': {
            'easy': 0.35,
            'medium': 0.45,
            'hard': 0.20
        },
        'total_marks': 720,
        'duration_minutes': 180
    }
}


@dataclass
class QuestionPaperConfig:
    """Configuration for generating a question paper."""
    topics: List[str] = None  # List of topics to include
    subjects: List[str] = None  # List of subjects to include
    chapters: List[str] = None  # List of chapters to include
    difficulty_distribution: Dict[str, float] = None  # {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
    question_count: int = 30  # Total number of questions
    exam_type: Optional[str] = None  # 'JEE_MAIN', 'JEE_ADVANCED', 'NEET', or None for custom
    include_solutions: bool = True  # Whether to include solutions in answer key
    randomize_order: bool = True  # Whether to randomize question order
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.topics is None:
            self.topics = []
        if self.subjects is None:
            self.subjects = []
        if self.chapters is None:
            self.chapters = []
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {'easy': 0.30, 'medium': 0.50, 'hard': 0.20}


@dataclass
class QuestionPaper:
    """Represents a generated question paper."""
    paper_id: str
    title: str
    questions: List[Dict[str, Any]]
    answer_key: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'questions': self.questions,
            'answer_key': self.answer_key,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class QuestionGenerator:
    """
    Generates custom question papers based on previous year questions and NCERT content.
    
    Supports topic-based selection, difficulty distribution, and full-length test
    generation matching actual JEE/NEET exam structures.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize the Question Generator.
        
        Args:
            db_session: SQLAlchemy database session for accessing questions
        """
        self.db = db_session
        logger.info("QuestionGenerator initialized successfully")
    
    def generate_paper(self, config: QuestionPaperConfig) -> QuestionPaper:
        """
        Generate a question paper based on the provided configuration.
        
        Args:
            config: Configuration for the question paper
        
        Returns:
            QuestionPaper object containing questions and answer key
        
        Raises:
            ValueError: If configuration is invalid or insufficient questions available
        """
        try:
            logger.info(f"Generating question paper with config: {config}")
            
            # If exam type is specified, use predefined structure
            if config.exam_type and config.exam_type in EXAM_STRUCTURES:
                return self._generate_full_length_test(config.exam_type)
            
            # Otherwise, generate custom paper
            return self._generate_custom_paper(config)
            
        except Exception as e:
            logger.error(f"Error generating question paper: {e}", exc_info=True)
            raise
    
    def _generate_custom_paper(self, config: QuestionPaperConfig) -> QuestionPaper:
        """
        Generate a custom question paper based on user-specified criteria.
        
        Args:
            config: Configuration for the question paper
        
        Returns:
            QuestionPaper object
        """
        # Calculate target counts for each difficulty level
        difficulty_counts = self._calculate_difficulty_counts(
            config.question_count,
            config.difficulty_distribution
        )
        
        logger.info(f"Target difficulty distribution: {difficulty_counts}")
        
        # Retrieve candidate questions
        candidate_questions = self._retrieve_candidate_questions(
            topics=config.topics,
            subjects=config.subjects,
            chapters=config.chapters
        )
        
        if not candidate_questions:
            raise ValueError("No questions found matching the specified criteria")
        
        logger.info(f"Found {len(candidate_questions)} candidate questions")
        
        # Select questions based on difficulty distribution
        selected_questions = self._select_questions_by_difficulty(
            candidate_questions,
            difficulty_counts,
            config.question_count
        )
        
        if len(selected_questions) < config.question_count:
            logger.warning(
                f"Only {len(selected_questions)} questions available, "
                f"requested {config.question_count}"
            )
        
        # Ensure no duplicates
        selected_questions = self._ensure_no_duplicates(selected_questions)
        
        # Randomize order if requested
        if config.randomize_order:
            random.shuffle(selected_questions)
        
        # Create answer key
        answer_key = self._create_answer_key(selected_questions, config.include_solutions)
        
        # Prepare questions for paper (without solutions)
        paper_questions = [q.to_dict(include_solution=False) for q in selected_questions]
        
        # Create metadata
        metadata = self._create_metadata(config, selected_questions)
        
        # Generate paper ID and title
        paper_id = str(uuid.uuid4())
        title = self._generate_title(config)
        
        paper = QuestionPaper(
            paper_id=paper_id,
            title=title,
            questions=paper_questions,
            answer_key=answer_key,
            metadata=metadata,
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Successfully generated custom paper with {len(paper_questions)} questions")
        return paper
    
    def _generate_full_length_test(self, exam_type: str) -> QuestionPaper:
        """
        Generate a full-length test matching actual exam structure.
        
        Args:
            exam_type: Type of exam ('JEE_MAIN', 'JEE_ADVANCED', 'NEET')
        
        Returns:
            QuestionPaper object matching exam structure
        """
        structure = EXAM_STRUCTURES[exam_type]
        logger.info(f"Generating full-length {exam_type} test")
        
        all_questions = []
        
        # Generate questions for each section (subject)
        for subject, section_config in structure['sections'].items():
            section_count = section_config['count']
            
            # Calculate difficulty distribution for this section
            difficulty_counts = self._calculate_difficulty_counts(
                section_count,
                structure['difficulty_distribution']
            )
            
            logger.info(f"Generating {section_count} {subject} questions")
            
            # Retrieve candidate questions for this subject
            candidate_questions = self._retrieve_candidate_questions(
                subjects=[subject],
                topics=[],
                chapters=[]
            )
            
            if not candidate_questions:
                raise ValueError(f"No questions found for subject: {subject}")
            
            # Select questions for this section
            section_questions = self._select_questions_by_difficulty(
                candidate_questions,
                difficulty_counts,
                section_count
            )
            
            if len(section_questions) < section_count:
                logger.warning(
                    f"Only {len(section_questions)} {subject} questions available, "
                    f"needed {section_count}"
                )
            
            all_questions.extend(section_questions)
        
        # Ensure no duplicates across sections
        all_questions = self._ensure_no_duplicates(all_questions)
        
        # Create answer key
        answer_key = self._create_answer_key(all_questions, include_solutions=True)
        
        # Prepare questions for paper
        paper_questions = [q.to_dict(include_solution=False) for q in all_questions]
        
        # Create metadata
        metadata = {
            'exam_type': exam_type,
            'total_questions': len(paper_questions),
            'total_marks': structure['total_marks'],
            'duration_minutes': structure['duration_minutes'],
            'sections': structure['sections'],
            'difficulty_distribution': structure['difficulty_distribution'],
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Generate paper
        paper_id = str(uuid.uuid4())
        title = f"{exam_type.replace('_', ' ')} - Full Length Test"
        
        paper = QuestionPaper(
            paper_id=paper_id,
            title=title,
            questions=paper_questions,
            answer_key=answer_key,
            metadata=metadata,
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Successfully generated {exam_type} test with {len(paper_questions)} questions")
        return paper
    
    def _calculate_difficulty_counts(
        self,
        total_count: int,
        distribution: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Calculate the number of questions needed for each difficulty level.
        
        Args:
            total_count: Total number of questions
            distribution: Difficulty distribution (percentages)
        
        Returns:
            Dictionary mapping difficulty to count
        """
        counts = {}
        remaining = total_count
        
        # Calculate counts for each difficulty
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in distribution:
                count = int(total_count * distribution[difficulty])
                counts[difficulty] = count
                remaining -= count
        
        # Distribute remaining questions to medium difficulty
        if remaining > 0:
            counts['medium'] = counts.get('medium', 0) + remaining
        
        return counts
    
    def _retrieve_candidate_questions(
        self,
        topics: List[str],
        subjects: List[str],
        chapters: List[str]
    ) -> List[Question]:
        """
        Retrieve candidate questions matching the specified criteria.
        
        Args:
            topics: List of topics to filter by
            subjects: List of subjects to filter by
            chapters: List of chapters to filter by
        
        Returns:
            List of Question objects
        """
        query = self.db.query(Question)
        
        # Apply filters
        if subjects:
            query = query.filter(Question.subject.in_(subjects))
        
        if chapters:
            query = query.filter(Question.chapter.in_(chapters))
        
        if topics:
            query = query.filter(Question.topic.in_(topics))
        
        return query.all()
    
    def _select_questions_by_difficulty(
        self,
        candidates: List[Question],
        difficulty_counts: Dict[str, int],
        total_needed: int
    ) -> List[Question]:
        """
        Select questions from candidates based on difficulty distribution.
        
        Args:
            candidates: List of candidate questions
            difficulty_counts: Target count for each difficulty
            total_needed: Total number of questions needed
        
        Returns:
            List of selected Question objects
        """
        # Group candidates by difficulty
        by_difficulty = defaultdict(list)
        for q in candidates:
            by_difficulty[q.difficulty].append(q)
        
        selected = []
        
        # Select questions for each difficulty level
        for difficulty, target_count in difficulty_counts.items():
            available = by_difficulty.get(difficulty, [])
            
            if not available:
                logger.warning(f"No {difficulty} questions available")
                continue
            
            # Randomly select questions
            count = min(target_count, len(available))
            selected_for_difficulty = random.sample(available, count)
            selected.extend(selected_for_difficulty)
            
            logger.info(f"Selected {count} {difficulty} questions")
        
        # If we don't have enough questions, try to add more while respecting distribution
        if len(selected) < total_needed:
            remaining_needed = total_needed - len(selected)
            selected_ids = {q.question_id for q in selected}
            remaining_candidates = [q for q in candidates if q.question_id not in selected_ids]
            
            if remaining_candidates:
                # Try to maintain distribution by selecting from difficulties that are under-represented
                additional = []
                
                # Calculate which difficulties need more questions
                for difficulty in ['easy', 'medium', 'hard']:
                    if difficulty in difficulty_counts:
                        current_count = sum(1 for q in selected if q.difficulty == difficulty)
                        target_count = difficulty_counts[difficulty]
                        
                        if current_count < target_count:
                            # Need more of this difficulty
                            available = [q for q in remaining_candidates if q.difficulty == difficulty]
                            needed = min(target_count - current_count, len(available))
                            
                            if needed > 0:
                                additional.extend(random.sample(available, needed))
                                remaining_needed -= needed
                                
                                if remaining_needed <= 0:
                                    break
                
                # If still need more, add from any remaining
                if remaining_needed > 0:
                    additional_ids = {q.question_id for q in additional}
                    still_available = [q for q in remaining_candidates 
                                     if q.question_id not in additional_ids]
                    
                    if still_available:
                        extra_count = min(remaining_needed, len(still_available))
                        additional.extend(random.sample(still_available, extra_count))
                        logger.warning(
                            f"Added {extra_count} questions from any difficulty to meet target count. "
                            f"Exact difficulty distribution may not be maintained."
                        )
                
                selected.extend(additional)
                logger.info(f"Added {len(additional)} additional questions to meet target")
        
        return selected
    
    def _ensure_no_duplicates(self, questions: List[Question]) -> List[Question]:
        """
        Ensure no duplicate questions in the list.
        
        Args:
            questions: List of Question objects
        
        Returns:
            List of unique Question objects
        """
        seen_ids = set()
        unique_questions = []
        
        for q in questions:
            if q.question_id not in seen_ids:
                seen_ids.add(q.question_id)
                unique_questions.append(q)
        
        if len(unique_questions) < len(questions):
            logger.warning(
                f"Removed {len(questions) - len(unique_questions)} duplicate questions"
            )
        
        return unique_questions
    
    def _create_answer_key(
        self,
        questions: List[Question],
        include_solutions: bool
    ) -> Dict[str, Any]:
        """
        Create an answer key for the question paper.
        
        Args:
            questions: List of Question objects
            include_solutions: Whether to include detailed solutions
        
        Returns:
            Dictionary containing answer key
        """
        answers = []
        
        for i, q in enumerate(questions, 1):
            answer_entry = {
                'question_number': i,
                'question_id': q.question_id,
                'correct_answer': q.correct_answer,
                'marks': q.marks
            }
            
            if include_solutions:
                answer_entry['solution'] = q.solution
                answer_entry['ncert_reference'] = q.ncert_reference
            
            answers.append(answer_entry)
        
        return {
            'answers': answers,
            'total_questions': len(questions),
            'total_marks': sum(q.marks for q in questions)
        }
    
    def _create_metadata(
        self,
        config: QuestionPaperConfig,
        questions: List[Question]
    ) -> Dict[str, Any]:
        """
        Create metadata for the question paper.
        
        Args:
            config: Paper configuration
            questions: Selected questions
        
        Returns:
            Dictionary containing metadata
        """
        # Count questions by subject
        by_subject = defaultdict(int)
        by_difficulty = defaultdict(int)
        by_chapter = defaultdict(int)
        
        for q in questions:
            by_subject[q.subject] += 1
            by_difficulty[q.difficulty] += 1
            by_chapter[q.chapter] += 1
        
        return {
            'total_questions': len(questions),
            'total_marks': sum(q.marks for q in questions),
            'subjects': dict(by_subject),
            'difficulty_distribution': dict(by_difficulty),
            'chapters': dict(by_chapter),
            'topics': config.topics if config.topics else 'All',
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _generate_title(self, config: QuestionPaperConfig) -> str:
        """
        Generate a title for the question paper.
        
        Args:
            config: Paper configuration
        
        Returns:
            Title string
        """
        if config.exam_type:
            return f"{config.exam_type.replace('_', ' ')} Practice Test"
        
        parts = []
        
        if config.subjects:
            parts.append(", ".join(config.subjects))
        
        if config.topics:
            if len(config.topics) <= 3:
                parts.append(f"({', '.join(config.topics)})")
            else:
                parts.append(f"({len(config.topics)} topics)")
        
        if not parts:
            parts.append("Mixed Topics")
        
        parts.append("Practice Test")
        
        return " - ".join(parts)
    
    def get_available_topics(self, subject: Optional[str] = None) -> List[str]:
        """
        Get list of available topics in the database.
        
        Args:
            subject: Optional subject filter
        
        Returns:
            List of unique topic names
        """
        query = self.db.query(Question.topic).distinct()
        
        if subject:
            query = query.filter(Question.subject == subject)
        
        topics = [row[0] for row in query.all()]
        return sorted(topics)
    
    def get_available_chapters(self, subject: Optional[str] = None) -> List[str]:
        """
        Get list of available chapters in the database.
        
        Args:
            subject: Optional subject filter
        
        Returns:
            List of unique chapter names
        """
        query = self.db.query(Question.chapter).distinct()
        
        if subject:
            query = query.filter(Question.subject == subject)
        
        chapters = [row[0] for row in query.all()]
        return sorted(chapters)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available questions.
        
        Returns:
            Dictionary with statistics
        """
        from sqlalchemy import func
        
        total_questions = self.db.query(Question).count()
        
        # Count by subject
        by_subject = dict(
            self.db.query(Question.subject, func.count(Question.question_id))
            .group_by(Question.subject)
            .all()
        )
        
        # Count by difficulty
        by_difficulty = dict(
            self.db.query(Question.difficulty, func.count(Question.question_id))
            .group_by(Question.difficulty)
            .all()
        )
        
        # Count by exam
        by_exam = dict(
            self.db.query(Question.exam, func.count(Question.question_id))
            .group_by(Question.exam)
            .all()
        )
        
        return {
            'total_questions': total_questions,
            'by_subject': by_subject,
            'by_difficulty': by_difficulty,
            'by_exam': by_exam
        }


def main():
    """
    Test the Question Generator with sample data.
    """
    print("="*60)
    print("Question Generator Test")
    print("="*60)
    
    try:
        from models.database import init_db, SessionLocal
        from config import Config
        
        # Initialize database
        init_db()
        db = SessionLocal()
        
        # Initialize generator
        generator = QuestionGenerator(db)
        
        # Get statistics
        print("\nDatabase Statistics:")
        stats = generator.get_statistics()
        print(f"  Total Questions: {stats['total_questions']}")
        print(f"  By Subject: {stats['by_subject']}")
        print(f"  By Difficulty: {stats['by_difficulty']}")
        print(f"  By Exam: {stats['by_exam']}")
        
        if stats['total_questions'] == 0:
            print("\n⚠ No questions in database. Please run setup_previous_papers.py first.")
            return
        
        # Test 1: Generate custom paper
        print(f"\n{'='*60}")
        print("Test 1: Custom Question Paper")
        print(f"{'='*60}")
        
        config = QuestionPaperConfig(
            subjects=['Physics'],
            question_count=10,
            difficulty_distribution={'easy': 0.3, 'medium': 0.5, 'hard': 0.2},
            include_solutions=True,
            randomize_order=True
        )
        
        paper = generator.generate_paper(config)
        print(f"\nGenerated Paper: {paper.title}")
        print(f"  Paper ID: {paper.paper_id}")
        print(f"  Total Questions: {len(paper.questions)}")
        print(f"  Total Marks: {paper.metadata['total_marks']}")
        print(f"  Difficulty Distribution: {paper.metadata['difficulty_distribution']}")
        
        # Display first 3 questions
        print(f"\nFirst 3 Questions:")
        for i, q in enumerate(paper.questions[:3], 1):
            print(f"\n  Q{i}. {q['question_text'][:100]}...")
            print(f"      Subject: {q['subject']}, Difficulty: {q['difficulty']}")
            print(f"      Chapter: {q['chapter']}")
        
        # Test 2: Generate full-length test (if enough questions)
        if stats['total_questions'] >= 30:
            print(f"\n{'='*60}")
            print("Test 2: Full-Length JEE Main Test")
            print(f"{'='*60}")
            
            config = QuestionPaperConfig(exam_type='JEE_MAIN')
            
            try:
                paper = generator.generate_paper(config)
                print(f"\nGenerated Paper: {paper.title}")
                print(f"  Total Questions: {len(paper.questions)}")
                print(f"  Total Marks: {paper.metadata['total_marks']}")
                print(f"  Duration: {paper.metadata['duration_minutes']} minutes")
                print(f"  Sections: {paper.metadata['sections']}")
            except ValueError as e:
                print(f"\n⚠ Could not generate full-length test: {e}")
        
        # Test 3: Verify no duplicates
        print(f"\n{'='*60}")
        print("Test 3: Duplicate Check")
        print(f"{'='*60}")
        
        question_ids = [q['question_id'] for q in paper.questions]
        unique_ids = set(question_ids)
        
        if len(question_ids) == len(unique_ids):
            print("✓ No duplicate questions found")
        else:
            print(f"✗ Found {len(question_ids) - len(unique_ids)} duplicate questions")
        
        # Test 4: Answer key completeness
        print(f"\n{'='*60}")
        print("Test 4: Answer Key Completeness")
        print(f"{'='*60}")
        
        answer_key = paper.answer_key
        print(f"  Total Answers: {answer_key['total_questions']}")
        print(f"  Total Marks: {answer_key['total_marks']}")
        
        # Check if all questions have answers
        if len(answer_key['answers']) == len(paper.questions):
            print("✓ All questions have answers in the key")
        else:
            print(f"✗ Answer key incomplete: {len(answer_key['answers'])} answers for {len(paper.questions)} questions")
        
        # Display first answer
        if answer_key['answers']:
            first_answer = answer_key['answers'][0]
            print(f"\nSample Answer:")
            print(f"  Question {first_answer['question_number']}")
            print(f"  Correct Answer: {first_answer['correct_answer']}")
            if 'solution' in first_answer:
                print(f"  Solution: {first_answer['solution'][:100]}...")
        
        print(f"\n{'='*60}")
        print("Test Complete")
        print(f"{'='*60}")
        
        db.close()
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
