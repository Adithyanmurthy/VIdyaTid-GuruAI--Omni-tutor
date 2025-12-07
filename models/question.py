"""
Question model for GuruAI application.
Stores previous year JEE/NEET questions with metadata and solutions.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, JSON, Index, DateTime
from models.database import Base


class Question(Base):
    """
    Question model for storing previous year exam questions.
    
    Attributes:
        question_id: Unique identifier (UUID)
        source: Source of the question (e.g., "JEE Main 2023", "NEET 2022")
        year: Year the question appeared
        exam: Exam name (JEE Main, JEE Advanced, NEET)
        subject: Subject (Physics, Chemistry, Mathematics, Biology)
        chapter: Chapter name
        topic: Specific topic within the chapter
        difficulty: Difficulty level (easy, medium, hard)
        question_text: The question text
        question_type: Type of question (MCQ, Numerical, Integer, etc.)
        options: List of options for MCQs (JSON)
        correct_answer: The correct answer
        solution: Detailed solution text
        ncert_reference: Reference to NCERT chapter and page
        marks: Marks allocated to the question
        created_at: Timestamp when question was added
        updated_at: Timestamp when question was last updated
    """
    __tablename__ = 'questions'
    
    question_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    exam = Column(String(50), nullable=False, index=True)
    subject = Column(String(50), nullable=False, index=True)
    chapter = Column(String(200), nullable=False, index=True)
    topic = Column(String(200), nullable=False, index=True)
    difficulty = Column(String(20), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50), default='MCQ', nullable=False)
    options = Column(JSON, nullable=True)  # For MCQs: ["option1", "option2", ...]
    correct_answer = Column(String(500), nullable=False)
    solution = Column(Text, nullable=False)
    ncert_reference = Column(String(500), nullable=False)
    marks = Column(Integer, default=4, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Composite indexes for efficient queries
    __table_args__ = (
        Index('idx_exam_year', 'exam', 'year'),
        Index('idx_subject_chapter', 'subject', 'chapter'),
        Index('idx_subject_difficulty', 'subject', 'difficulty'),
        Index('idx_exam_subject_year', 'exam', 'subject', 'year'),
    )
    
    def __init__(self, source, year, exam, subject, chapter, topic, difficulty,
                 question_text, correct_answer, solution, ncert_reference,
                 question_type='MCQ', options=None, marks=4):
        """
        Initialize a new Question.
        
        Args:
            source: Source identifier (e.g., "JEE Main 2023")
            year: Year of the exam
            exam: Exam name (JEE Main, JEE Advanced, NEET)
            subject: Subject name
            chapter: Chapter name
            topic: Topic name
            difficulty: Difficulty level (easy, medium, hard)
            question_text: The question text
            correct_answer: The correct answer
            solution: Detailed solution
            ncert_reference: NCERT reference
            question_type: Type of question (default: MCQ)
            options: List of options for MCQs
            marks: Marks for the question (default: 4)
        """
        self.question_id = str(uuid.uuid4())
        self.source = source
        self.year = year
        self.exam = exam
        self.subject = subject
        self.chapter = chapter
        self.topic = topic
        self.difficulty = difficulty
        self.question_text = question_text
        self.question_type = question_type
        self.options = options or []
        self.correct_answer = correct_answer
        self.solution = solution
        self.ncert_reference = ncert_reference
        self.marks = marks
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self, include_solution=False):
        """
        Convert question to dictionary.
        
        Args:
            include_solution: Whether to include the solution (default: False)
            
        Returns:
            Dictionary representation of question
        """
        data = {
            'question_id': self.question_id,
            'source': self.source,
            'year': self.year,
            'exam': self.exam,
            'subject': self.subject,
            'chapter': self.chapter,
            'topic': self.topic,
            'difficulty': self.difficulty,
            'question_text': self.question_text,
            'question_type': self.question_type,
            'options': self.options,
            'marks': self.marks,
            'ncert_reference': self.ncert_reference
        }
        
        if include_solution:
            data['correct_answer'] = self.correct_answer
            data['solution'] = self.solution
        
        return data
    
    @staticmethod
    def get_by_filters(db_session, exam=None, year=None, subject=None, 
                      chapter=None, topic=None, difficulty=None, limit=None):
        """
        Query questions with various filters.
        
        Args:
            db_session: SQLAlchemy database session
            exam: Filter by exam name
            year: Filter by year
            subject: Filter by subject
            chapter: Filter by chapter
            topic: Filter by topic
            difficulty: Filter by difficulty
            limit: Maximum number of results
            
        Returns:
            List of Question objects matching the filters
        """
        query = db_session.query(Question)
        
        if exam:
            query = query.filter(Question.exam == exam)
        if year:
            query = query.filter(Question.year == year)
        if subject:
            query = query.filter(Question.subject == subject)
        if chapter:
            query = query.filter(Question.chapter == chapter)
        if topic:
            query = query.filter(Question.topic == topic)
        if difficulty:
            query = query.filter(Question.difficulty == difficulty)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    @staticmethod
    def get_random_questions(db_session, count, subject=None, difficulty=None, 
                           exclude_ids=None):
        """
        Get random questions matching criteria.
        
        Args:
            db_session: SQLAlchemy database session
            count: Number of questions to retrieve
            subject: Filter by subject
            difficulty: Filter by difficulty
            exclude_ids: List of question IDs to exclude
            
        Returns:
            List of random Question objects
        """
        from sqlalchemy.sql.expression import func
        
        query = db_session.query(Question)
        
        if subject:
            query = query.filter(Question.subject == subject)
        if difficulty:
            query = query.filter(Question.difficulty == difficulty)
        if exclude_ids:
            query = query.filter(~Question.question_id.in_(exclude_ids))
        
        # Order randomly and limit
        query = query.order_by(func.random()).limit(count)
        
        return query.all()
    
    @staticmethod
    def get_statistics(db_session, exam=None, subject=None):
        """
        Get statistics about questions in the database.
        
        Args:
            db_session: SQLAlchemy database session
            exam: Filter by exam
            subject: Filter by subject
            
        Returns:
            Dictionary with statistics
        """
        from sqlalchemy import func
        
        query = db_session.query(Question)
        
        if exam:
            query = query.filter(Question.exam == exam)
        if subject:
            query = query.filter(Question.subject == subject)
        
        total_count = query.count()
        
        # Count by difficulty
        difficulty_counts = db_session.query(
            Question.difficulty,
            func.count(Question.question_id)
        ).filter(
            *([Question.exam == exam] if exam else []),
            *([Question.subject == subject] if subject else [])
        ).group_by(Question.difficulty).all()
        
        # Count by year
        year_counts = db_session.query(
            Question.year,
            func.count(Question.question_id)
        ).filter(
            *([Question.exam == exam] if exam else []),
            *([Question.subject == subject] if subject else [])
        ).group_by(Question.year).order_by(Question.year.desc()).all()
        
        return {
            'total_questions': total_count,
            'by_difficulty': {diff: count for diff, count in difficulty_counts},
            'by_year': {year: count for year, count in year_counts}
        }
    
    def __repr__(self):
        return (f"<Question(question_id='{self.question_id}', "
                f"source='{self.source}', subject='{self.subject}', "
                f"difficulty='{self.difficulty}')>")
