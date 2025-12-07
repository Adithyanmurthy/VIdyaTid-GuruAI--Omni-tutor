"""
Progress model for GuruAI application.
Tracks user learning progress across subjects, chapters, and topics.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from models.database import Base


class Progress(Base):
    """
    Progress model for tracking user's learning progress.
    
    Attributes:
        progress_id: Unique identifier (UUID)
        user_id: Foreign key to User
        subject: Subject name (Physics, Chemistry, Mathematics, Biology)
        chapter: Chapter name or number
        topic: Specific topic within the chapter
        questions_attempted: Total number of questions attempted
        questions_correct: Number of questions answered correctly
        accuracy: Calculated accuracy percentage
        last_studied: Last time this topic was studied
    """
    __tablename__ = 'progress'
    
    progress_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.user_id'), nullable=False, index=True)
    subject = Column(String(50), nullable=False, index=True)
    chapter = Column(String(200), nullable=False)
    topic = Column(String(200), nullable=False)
    questions_attempted = Column(Integer, default=0, nullable=False)
    questions_correct = Column(Integer, default=0, nullable=False)
    accuracy = Column(Float, default=0.0, nullable=False)
    last_studied = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="progress_records")
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_user_subject_chapter_topic', 'user_id', 'subject', 'chapter', 'topic'),
    )
    
    def __init__(self, user_id, subject, chapter, topic):
        """
        Initialize a new Progress record.
        
        Args:
            user_id: User ID this progress belongs to
            subject: Subject name
            chapter: Chapter name
            topic: Topic name
        """
        self.progress_id = str(uuid.uuid4())
        self.user_id = user_id
        self.subject = subject
        self.chapter = chapter
        self.topic = topic
        self.questions_attempted = 0
        self.questions_correct = 0
        self.accuracy = 0.0
        self.last_studied = datetime.utcnow()
    
    def record_attempt(self, is_correct):
        """
        Record a question attempt and update statistics.
        
        Args:
            is_correct: Boolean indicating if the answer was correct
        """
        self.questions_attempted += 1
        if is_correct:
            self.questions_correct += 1
        
        # Recalculate accuracy
        self.calculate_accuracy()
        self.last_studied = datetime.utcnow()
    
    def calculate_accuracy(self):
        """
        Calculate and update the accuracy percentage.
        
        Returns:
            Calculated accuracy as a float
        """
        if self.questions_attempted > 0:
            self.accuracy = (self.questions_correct / self.questions_attempted) * 100.0
        else:
            self.accuracy = 0.0
        return self.accuracy
    
    def reset_progress(self):
        """Reset all progress statistics to zero."""
        self.questions_attempted = 0
        self.questions_correct = 0
        self.accuracy = 0.0
    
    def to_dict(self):
        """
        Convert progress record to dictionary.
        
        Returns:
            Dictionary representation of progress
        """
        return {
            'progress_id': self.progress_id,
            'user_id': self.user_id,
            'subject': self.subject,
            'chapter': self.chapter,
            'topic': self.topic,
            'questions_attempted': self.questions_attempted,
            'questions_correct': self.questions_correct,
            'accuracy': round(self.accuracy, 2),
            'last_studied': self.last_studied.isoformat() if self.last_studied else None
        }
    
    @staticmethod
    def get_subject_summary(progress_records):
        """
        Generate a summary of progress across subjects.
        
        Args:
            progress_records: List of Progress objects
            
        Returns:
            Dictionary with subject-wise statistics
        """
        summary = {}
        
        for record in progress_records:
            if record.subject not in summary:
                summary[record.subject] = {
                    'total_attempted': 0,
                    'total_correct': 0,
                    'accuracy': 0.0,
                    'chapters': set(),
                    'topics': set()
                }
            
            summary[record.subject]['total_attempted'] += record.questions_attempted
            summary[record.subject]['total_correct'] += record.questions_correct
            summary[record.subject]['chapters'].add(record.chapter)
            summary[record.subject]['topics'].add(record.topic)
        
        # Calculate overall accuracy for each subject
        for subject in summary:
            if summary[subject]['total_attempted'] > 0:
                summary[subject]['accuracy'] = (
                    summary[subject]['total_correct'] / 
                    summary[subject]['total_attempted']
                ) * 100.0
            summary[subject]['chapters'] = len(summary[subject]['chapters'])
            summary[subject]['topics'] = len(summary[subject]['topics'])
        
        return summary
    
    @staticmethod
    def get_weak_areas(progress_records, threshold=60.0):
        """
        Identify weak areas based on accuracy threshold.
        
        Args:
            progress_records: List of Progress objects
            threshold: Accuracy threshold below which topics are considered weak
            
        Returns:
            List of topics with accuracy below threshold, sorted by accuracy
        """
        weak_areas = [
            {
                'subject': record.subject,
                'chapter': record.chapter,
                'topic': record.topic,
                'accuracy': record.accuracy,
                'questions_attempted': record.questions_attempted
            }
            for record in progress_records
            if record.accuracy < threshold and record.questions_attempted > 0
        ]
        
        # Sort by accuracy (lowest first)
        weak_areas.sort(key=lambda x: x['accuracy'])
        
        return weak_areas
    
    def __repr__(self):
        return (f"<Progress(user_id='{self.user_id}', subject='{self.subject}', "
                f"chapter='{self.chapter}', topic='{self.topic}', accuracy={self.accuracy:.2f}%)>")
