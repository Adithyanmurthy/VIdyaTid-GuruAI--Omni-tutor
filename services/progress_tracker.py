"""
Progress Tracker Service for GuruAI application.
Handles progress recording, statistics aggregation, and recommendations.
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from models.progress import Progress
from models.user import User
from models.question import Question


class ProgressTracker:
    """
    Service for tracking and analyzing user learning progress.
    
    Provides functionality for:
    - Recording question attempts
    - Calculating accuracy statistics
    - Aggregating progress by subject/chapter/topic
    - Generating recommendations based on weak areas
    - Persisting progress data locally
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize the ProgressTracker.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
    
    def record_attempt(
        self,
        user_id: str,
        subject: str,
        chapter: str,
        topic: str,
        is_correct: bool,
        question_id: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> Dict:
        """
        Record a question attempt and update progress statistics.
        
        Args:
            user_id: User ID
            subject: Subject name (Physics, Chemistry, Mathematics, Biology)
            chapter: Chapter name or number
            topic: Specific topic within the chapter
            is_correct: Whether the answer was correct
            question_id: Optional question ID for tracking
            difficulty: Optional difficulty level
            
        Returns:
            Dictionary with updated progress information
        """
        # Find or create progress record
        progress = self.db.query(Progress).filter(
            and_(
                Progress.user_id == user_id,
                Progress.subject == subject,
                Progress.chapter == chapter,
                Progress.topic == topic
            )
        ).first()
        
        if not progress:
            progress = Progress(
                user_id=user_id,
                subject=subject,
                chapter=chapter,
                topic=topic
            )
            self.db.add(progress)
        
        # Record the attempt
        progress.record_attempt(is_correct)
        
        # Commit changes
        self.db.commit()
        
        return {
            'status': 'success',
            'progress': progress.to_dict(),
            'message': f'Recorded {"correct" if is_correct else "incorrect"} attempt for {topic}'
        }
    
    def get_user_progress(self, user_id: str) -> Dict:
        """
        Get comprehensive progress data for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with complete progress information including:
            - Subject-wise statistics
            - Chapter-wise breakdown
            - Overall accuracy
            - Recent activity
        """
        # Get all progress records for the user
        progress_records = self.db.query(Progress).filter(
            Progress.user_id == user_id
        ).all()
        
        if not progress_records:
            return {
                'user_id': user_id,
                'subjects': {},
                'overall_stats': {
                    'total_attempted': 0,
                    'total_correct': 0,
                    'accuracy': 0.0
                },
                'recent_activity': []
            }
        
        # Get subject summary
        subject_summary = Progress.get_subject_summary(progress_records)
        
        # Calculate overall statistics
        total_attempted = sum(r.questions_attempted for r in progress_records)
        total_correct = sum(r.questions_correct for r in progress_records)
        overall_accuracy = (total_correct / total_attempted * 100.0) if total_attempted > 0 else 0.0
        
        # Get recent activity (last 10 studied topics)
        recent_activity = sorted(
            progress_records,
            key=lambda x: x.last_studied,
            reverse=True
        )[:10]
        
        # Build chapter-wise breakdown
        chapters_by_subject = {}
        for record in progress_records:
            if record.subject not in chapters_by_subject:
                chapters_by_subject[record.subject] = {}
            
            if record.chapter not in chapters_by_subject[record.subject]:
                chapters_by_subject[record.subject][record.chapter] = {
                    'topics': [],
                    'total_attempted': 0,
                    'total_correct': 0,
                    'accuracy': 0.0
                }
            
            chapter_data = chapters_by_subject[record.subject][record.chapter]
            chapter_data['topics'].append(record.to_dict())
            chapter_data['total_attempted'] += record.questions_attempted
            chapter_data['total_correct'] += record.questions_correct
            
            if chapter_data['total_attempted'] > 0:
                chapter_data['accuracy'] = (
                    chapter_data['total_correct'] / chapter_data['total_attempted'] * 100.0
                )
        
        return {
            'user_id': user_id,
            'subjects': subject_summary,
            'chapters': chapters_by_subject,
            'overall_stats': {
                'total_attempted': total_attempted,
                'total_correct': total_correct,
                'accuracy': round(overall_accuracy, 2)
            },
            'recent_activity': [r.to_dict() for r in recent_activity]
        }
    
    def get_subject_progress(self, user_id: str, subject: str) -> Dict:
        """
        Get detailed progress for a specific subject.
        
        Args:
            user_id: User ID
            subject: Subject name
            
        Returns:
            Dictionary with subject-specific progress
        """
        progress_records = self.db.query(Progress).filter(
            and_(
                Progress.user_id == user_id,
                Progress.subject == subject
            )
        ).all()
        
        if not progress_records:
            return {
                'subject': subject,
                'chapters': {},
                'total_attempted': 0,
                'total_correct': 0,
                'accuracy': 0.0
            }
        
        # Group by chapter
        chapters = {}
        total_attempted = 0
        total_correct = 0
        
        for record in progress_records:
            if record.chapter not in chapters:
                chapters[record.chapter] = {
                    'topics': [],
                    'attempted': 0,
                    'correct': 0,
                    'accuracy': 0.0
                }
            
            chapters[record.chapter]['topics'].append(record.to_dict())
            chapters[record.chapter]['attempted'] += record.questions_attempted
            chapters[record.chapter]['correct'] += record.questions_correct
            
            total_attempted += record.questions_attempted
            total_correct += record.questions_correct
        
        # Calculate chapter accuracies
        for chapter in chapters.values():
            if chapter['attempted'] > 0:
                chapter['accuracy'] = round(
                    (chapter['correct'] / chapter['attempted']) * 100.0, 2
                )
        
        overall_accuracy = (total_correct / total_attempted * 100.0) if total_attempted > 0 else 0.0
        
        return {
            'subject': subject,
            'chapters': chapters,
            'total_attempted': total_attempted,
            'total_correct': total_correct,
            'accuracy': round(overall_accuracy, 2)
        }
    
    def get_chapter_progress(
        self,
        user_id: str,
        subject: str,
        chapter: str
    ) -> Dict:
        """
        Get detailed progress for a specific chapter.
        
        Args:
            user_id: User ID
            subject: Subject name
            chapter: Chapter name
            
        Returns:
            Dictionary with chapter-specific progress
        """
        progress_records = self.db.query(Progress).filter(
            and_(
                Progress.user_id == user_id,
                Progress.subject == subject,
                Progress.chapter == chapter
            )
        ).all()
        
        if not progress_records:
            return {
                'subject': subject,
                'chapter': chapter,
                'topics': [],
                'total_attempted': 0,
                'total_correct': 0,
                'accuracy': 0.0
            }
        
        total_attempted = sum(r.questions_attempted for r in progress_records)
        total_correct = sum(r.questions_correct for r in progress_records)
        accuracy = (total_correct / total_attempted * 100.0) if total_attempted > 0 else 0.0
        
        return {
            'subject': subject,
            'chapter': chapter,
            'topics': [r.to_dict() for r in progress_records],
            'total_attempted': total_attempted,
            'total_correct': total_correct,
            'accuracy': round(accuracy, 2)
        }
    
    def get_weak_areas(
        self,
        user_id: str,
        threshold: float = 60.0,
        min_attempts: int = 3
    ) -> List[Dict]:
        """
        Identify weak areas based on accuracy threshold.
        
        Args:
            user_id: User ID
            threshold: Accuracy threshold below which topics are considered weak (default: 60%)
            min_attempts: Minimum number of attempts required to consider a topic (default: 3)
            
        Returns:
            List of weak areas sorted by accuracy (lowest first)
        """
        progress_records = self.db.query(Progress).filter(
            and_(
                Progress.user_id == user_id,
                Progress.questions_attempted >= min_attempts
            )
        ).all()
        
        weak_areas = Progress.get_weak_areas(progress_records, threshold)
        
        return weak_areas
    
    def generate_recommendations(
        self,
        user_id: str,
        max_recommendations: int = 5
    ) -> List[Dict]:
        """
        Generate study recommendations based on user's weak areas.
        
        Args:
            user_id: User ID
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of recommended topics to study, prioritized by need
        """
        # Get weak areas (accuracy < 60%)
        weak_areas = self.get_weak_areas(user_id, threshold=60.0, min_attempts=3)
        
        # Get topics with no attempts or very few attempts
        all_progress = self.db.query(Progress).filter(
            Progress.user_id == user_id
        ).all()
        
        # Identify subjects/chapters that need more practice
        subject_coverage = {}
        for record in all_progress:
            if record.subject not in subject_coverage:
                subject_coverage[record.subject] = {
                    'total_attempts': 0,
                    'chapters': set()
                }
            subject_coverage[record.subject]['total_attempts'] += record.questions_attempted
            subject_coverage[record.subject]['chapters'].add(record.chapter)
        
        recommendations = []
        
        # Priority 1: Weak areas (low accuracy)
        for weak in weak_areas[:max_recommendations]:
            recommendations.append({
                'priority': 'high',
                'reason': f'Low accuracy ({weak["accuracy"]:.1f}%)',
                'subject': weak['subject'],
                'chapter': weak['chapter'],
                'topic': weak['topic'],
                'current_accuracy': weak['accuracy'],
                'questions_attempted': weak['questions_attempted'],
                'recommendation': 'Review fundamentals and practice more questions'
            })
        
        # Priority 2: Topics with few attempts
        if len(recommendations) < max_recommendations:
            few_attempts = [
                r for r in all_progress
                if 0 < r.questions_attempted < 5 and r.accuracy < 70.0
            ]
            few_attempts.sort(key=lambda x: x.questions_attempted)
            
            for record in few_attempts[:max_recommendations - len(recommendations)]:
                recommendations.append({
                    'priority': 'medium',
                    'reason': f'Limited practice ({record.questions_attempted} questions)',
                    'subject': record.subject,
                    'chapter': record.chapter,
                    'topic': record.topic,
                    'current_accuracy': record.accuracy,
                    'questions_attempted': record.questions_attempted,
                    'recommendation': 'Practice more questions to build confidence'
                })
        
        return recommendations
    
    def get_statistics_summary(self, user_id: str) -> Dict:
        """
        Get a comprehensive statistics summary for the user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with various statistics
        """
        progress_records = self.db.query(Progress).filter(
            Progress.user_id == user_id
        ).all()
        
        if not progress_records:
            return {
                'total_topics_studied': 0,
                'total_questions_attempted': 0,
                'total_questions_correct': 0,
                'overall_accuracy': 0.0,
                'subjects_covered': 0,
                'chapters_covered': 0,
                'strong_subjects': [],
                'weak_subjects': [],
                'study_streak': 0
            }
        
        # Calculate basic stats
        total_attempted = sum(r.questions_attempted for r in progress_records)
        total_correct = sum(r.questions_correct for r in progress_records)
        overall_accuracy = (total_correct / total_attempted * 100.0) if total_attempted > 0 else 0.0
        
        # Get subject summary
        subject_summary = Progress.get_subject_summary(progress_records)
        
        # Identify strong and weak subjects
        strong_subjects = []
        weak_subjects = []
        
        for subject, stats in subject_summary.items():
            if stats['accuracy'] >= 75.0 and stats['total_attempted'] >= 10:
                strong_subjects.append({
                    'subject': subject,
                    'accuracy': round(stats['accuracy'], 2),
                    'questions_attempted': stats['total_attempted']
                })
            elif stats['accuracy'] < 60.0 and stats['total_attempted'] >= 5:
                weak_subjects.append({
                    'subject': subject,
                    'accuracy': round(stats['accuracy'], 2),
                    'questions_attempted': stats['total_attempted']
                })
        
        # Sort by accuracy
        strong_subjects.sort(key=lambda x: x['accuracy'], reverse=True)
        weak_subjects.sort(key=lambda x: x['accuracy'])
        
        # Count unique subjects and chapters
        subjects_covered = len(set(r.subject for r in progress_records))
        chapters_covered = len(set((r.subject, r.chapter) for r in progress_records))
        
        # Calculate study streak (consecutive days with activity)
        study_streak = self._calculate_study_streak(progress_records)
        
        return {
            'total_topics_studied': len(progress_records),
            'total_questions_attempted': total_attempted,
            'total_questions_correct': total_correct,
            'overall_accuracy': round(overall_accuracy, 2),
            'subjects_covered': subjects_covered,
            'chapters_covered': chapters_covered,
            'strong_subjects': strong_subjects,
            'weak_subjects': weak_subjects,
            'study_streak': study_streak
        }
    
    def _calculate_study_streak(self, progress_records: List[Progress]) -> int:
        """
        Calculate the current study streak (consecutive days with activity).
        
        Args:
            progress_records: List of Progress objects
            
        Returns:
            Number of consecutive days with study activity
        """
        if not progress_records:
            return 0
        
        # Get unique study dates
        study_dates = set()
        for record in progress_records:
            if record.last_studied:
                study_dates.add(record.last_studied.date())
        
        if not study_dates:
            return 0
        
        # Sort dates in descending order
        sorted_dates = sorted(study_dates, reverse=True)
        
        # Check for consecutive days starting from most recent
        streak = 0
        today = datetime.utcnow().date()
        
        for i, date in enumerate(sorted_dates):
            expected_date = today if i == 0 else sorted_dates[i-1]
            
            # Allow for today or yesterday as starting point
            if i == 0 and (date == today or (today - date).days == 1):
                streak = 1
            elif i > 0 and (expected_date - date).days == 1:
                streak += 1
            else:
                break
        
        return streak
    
    def reset_topic_progress(
        self,
        user_id: str,
        subject: str,
        chapter: str,
        topic: str
    ) -> Dict:
        """
        Reset progress for a specific topic.
        
        Args:
            user_id: User ID
            subject: Subject name
            chapter: Chapter name
            topic: Topic name
            
        Returns:
            Dictionary with status
        """
        progress = self.db.query(Progress).filter(
            and_(
                Progress.user_id == user_id,
                Progress.subject == subject,
                Progress.chapter == chapter,
                Progress.topic == topic
            )
        ).first()
        
        if not progress:
            return {
                'status': 'error',
                'message': 'Progress record not found'
            }
        
        progress.reset_progress()
        self.db.commit()
        
        return {
            'status': 'success',
            'message': f'Progress reset for {topic}',
            'progress': progress.to_dict()
        }
    
    def delete_user_progress(self, user_id: str) -> Dict:
        """
        Delete all progress data for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with status
        """
        deleted_count = self.db.query(Progress).filter(
            Progress.user_id == user_id
        ).delete()
        
        self.db.commit()
        
        return {
            'status': 'success',
            'message': f'Deleted {deleted_count} progress records',
            'deleted_count': deleted_count
        }
    
    def export_progress(self, user_id: str) -> Dict:
        """
        Export all progress data for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with all progress data
        """
        progress_records = self.db.query(Progress).filter(
            Progress.user_id == user_id
        ).all()
        
        return {
            'user_id': user_id,
            'export_date': datetime.utcnow().isoformat(),
            'total_records': len(progress_records),
            'progress_data': [r.to_dict() for r in progress_records]
        }

        # Commit changes
        self.db.commit()
       