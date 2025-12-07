"""
Utility for importing previous year questions from various formats.
Supports CSV, JSON, and Excel formats.
"""
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Question, init_db
from models.database import SessionLocal


class QuestionImporter:
    """
    Utility class for importing questions from various file formats.
    """
    
    def __init__(self):
        """Initialize the question importer."""
        # Initialize database if not already done
        init_db()
        from models.database import SessionLocal as get_session
        self.db = get_session()
        self.imported_count = 0
        self.error_count = 0
        self.errors = []
    
    def import_from_json(self, file_path: str) -> Dict:
        """
        Import questions from a JSON file.
        
        Expected JSON format:
        [
            {
                "source": "JEE Main 2023",
                "year": 2023,
                "exam": "JEE Main",
                "subject": "Physics",
                "chapter": "Mechanics",
                "topic": "Laws of Motion",
                "difficulty": "medium",
                "question_text": "...",
                "question_type": "MCQ",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "solution": "...",
                "ncert_reference": "Class 11 Physics, Chapter 5",
                "marks": 4
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with import statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            for q_data in questions_data:
                try:
                    self._import_question(q_data)
                except Exception as e:
                    self.error_count += 1
                    self.errors.append({
                        'question': q_data.get('question_text', 'Unknown')[:50],
                        'error': str(e)
                    })
            
            self.db.commit()
            
            return self._get_import_stats()
            
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to import from JSON: {e}")
        finally:
            self.db.close()
    
    def import_from_csv(self, file_path: str) -> Dict:
        """
        Import questions from a CSV file.
        
        Expected CSV columns:
        source,year,exam,subject,chapter,topic,difficulty,question_text,
        question_type,options,correct_answer,solution,ncert_reference,marks
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dictionary with import statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Parse options if present
                        if row.get('options'):
                            row['options'] = json.loads(row['options'])
                        
                        # Convert year and marks to integers
                        row['year'] = int(row['year'])
                        row['marks'] = int(row.get('marks', 4))
                        
                        self._import_question(row)
                    except Exception as e:
                        self.error_count += 1
                        self.errors.append({
                            'question': row.get('question_text', 'Unknown')[:50],
                            'error': str(e)
                        })
            
            self.db.commit()
            
            return self._get_import_stats()
            
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to import from CSV: {e}")
        finally:
            self.db.close()
    
    def import_from_directory(self, directory_path: str, file_pattern: str = "*.json") -> Dict:
        """
        Import questions from all matching files in a directory.
        
        Args:
            directory_path: Path to directory containing question files
            file_pattern: File pattern to match (default: *.json)
            
        Returns:
            Dictionary with import statistics
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise Exception(f"Directory not found: {directory_path}")
        
        files = list(directory.glob(file_pattern))
        
        if not files:
            raise Exception(f"No files matching pattern '{file_pattern}' found in {directory_path}")
        
        print(f"Found {len(files)} files to import")
        
        for file_path in files:
            print(f"Importing from {file_path.name}...")
            
            if file_path.suffix == '.json':
                self.import_from_json(str(file_path))
            elif file_path.suffix == '.csv':
                self.import_from_csv(str(file_path))
            else:
                print(f"Skipping unsupported file type: {file_path.suffix}")
        
        return self._get_import_stats()
    
    def _import_question(self, q_data: Dict):
        """
        Import a single question into the database.
        
        Args:
            q_data: Dictionary containing question data
        """
        # Create Question object
        question = Question(
            source=q_data['source'],
            year=q_data['year'],
            exam=q_data['exam'],
            subject=q_data['subject'],
            chapter=q_data['chapter'],
            topic=q_data['topic'],
            difficulty=q_data['difficulty'],
            question_text=q_data['question_text'],
            correct_answer=q_data['correct_answer'],
            solution=q_data['solution'],
            ncert_reference=q_data['ncert_reference'],
            question_type=q_data.get('question_type', 'MCQ'),
            options=q_data.get('options'),
            marks=q_data.get('marks', 4)
        )
        
        self.db.add(question)
        self.imported_count += 1
    
    def _get_import_stats(self) -> Dict:
        """
        Get import statistics.
        
        Returns:
            Dictionary with import statistics
        """
        return {
            'imported': self.imported_count,
            'errors': self.error_count,
            'error_details': self.errors,
            'success_rate': (self.imported_count / (self.imported_count + self.error_count) * 100) 
                           if (self.imported_count + self.error_count) > 0 else 0
        }
    
    @staticmethod
    def create_sample_json_template(output_path: str):
        """
        Create a sample JSON template file for question import.
        
        Args:
            output_path: Path where template should be saved
        """
        template = [
            {
                "source": "JEE Main 2023",
                "year": 2023,
                "exam": "JEE Main",
                "subject": "Physics",
                "chapter": "Mechanics",
                "topic": "Laws of Motion",
                "difficulty": "medium",
                "question_text": "A block of mass 5 kg is placed on a horizontal surface...",
                "question_type": "MCQ",
                "options": [
                    "Option A",
                    "Option B",
                    "Option C",
                    "Option D"
                ],
                "correct_answer": "Option A",
                "solution": "Step 1: ...\nStep 2: ...\nNCERT Reference: ...",
                "ncert_reference": "Class 11 Physics, Chapter 5, Page 112",
                "marks": 4
            }
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"Sample template created at: {output_path}")
    
    @staticmethod
    def create_sample_csv_template(output_path: str):
        """
        Create a sample CSV template file for question import.
        
        Args:
            output_path: Path where template should be saved
        """
        headers = [
            'source', 'year', 'exam', 'subject', 'chapter', 'topic',
            'difficulty', 'question_text', 'question_type', 'options',
            'correct_answer', 'solution', 'ncert_reference', 'marks'
        ]
        
        sample_row = [
            'JEE Main 2023',
            '2023',
            'JEE Main',
            'Physics',
            'Mechanics',
            'Laws of Motion',
            'medium',
            'A block of mass 5 kg is placed on a horizontal surface...',
            'MCQ',
            '["Option A", "Option B", "Option C", "Option D"]',
            'Option A',
            'Step 1: ...\nStep 2: ...\nNCERT Reference: ...',
            'Class 11 Physics, Chapter 5, Page 112',
            '4'
        ]
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(sample_row)
        
        print(f"Sample template created at: {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import previous year questions')
    parser.add_argument('action', choices=['import', 'template'],
                       help='Action to perform')
    parser.add_argument('--file', help='File to import from')
    parser.add_argument('--directory', help='Directory to import from')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='File format')
    parser.add_argument('--output', help='Output path for template')
    
    args = parser.parse_args()
    
    if args.action == 'template':
        output = args.output or f'question_template.{args.format}'
        if args.format == 'json':
            QuestionImporter.create_sample_json_template(output)
        else:
            QuestionImporter.create_sample_csv_template(output)
    
    elif args.action == 'import':
        importer = QuestionImporter()
        
        if args.file:
            if args.format == 'json':
                stats = importer.import_from_json(args.file)
            else:
                stats = importer.import_from_csv(args.file)
        elif args.directory:
            pattern = f'*.{args.format}'
            stats = importer.import_from_directory(args.directory, pattern)
        else:
            print("Error: Please specify --file or --directory")
            return
        
        print("\nImport Statistics:")
        print(f"  Successfully imported: {stats['imported']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Success rate: {stats['success_rate']:.2f}%")
        
        if stats['error_details']:
            print("\nError Details:")
            for error in stats['error_details'][:5]:  # Show first 5 errors
                print(f"  - {error['question']}: {error['error']}")


if __name__ == '__main__':
    main()
