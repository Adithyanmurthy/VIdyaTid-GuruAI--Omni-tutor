"""
Database migration utilities for GuruAI.
Handles database initialization, schema creation, and migrations.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import init_db, create_tables, drop_tables, Base
from models.user import User
from models.progress import Progress
from models.session import Session
from config import Config


def initialize_database(database_uri=None, drop_existing=False):
    """
    Initialize the database with all tables.
    
    Args:
        database_uri: Optional database URI. If not provided, uses Config.SQLALCHEMY_DATABASE_URI
        drop_existing: If True, drop existing tables before creating new ones
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print("Initializing database...")
        
        # Initialize database engine
        engine = init_db(database_uri)
        print(f"Database engine created: {engine.url}")
        
        # Drop existing tables if requested
        if drop_existing:
            print("Dropping existing tables...")
            drop_tables()
            print("Existing tables dropped.")
        
        # Create all tables
        print("Creating tables...")
        create_tables()
        print("Tables created successfully:")
        print(f"  - {User.__tablename__}")
        print(f"  - {Progress.__tablename__}")
        print(f"  - {Session.__tablename__}")
        
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False


def verify_database():
    """
    Verify that the database is properly set up.
    
    Returns:
        True if database is valid, False otherwise
    """
    try:
        from models.database import engine
        from sqlalchemy import inspect
        
        if engine is None:
            print("Database engine not initialized.")
            return False
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        required_tables = ['users', 'progress', 'sessions']
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            print(f"Missing tables: {', '.join(missing_tables)}")
            return False
        
        print("Database verification successful.")
        print(f"Found tables: {', '.join(tables)}")
        return True
        
    except Exception as e:
        print(f"Error verifying database: {e}")
        return False


def reset_database(database_uri=None):
    """
    Reset the database by dropping and recreating all tables.
    WARNING: This will delete all data!
    
    Args:
        database_uri: Optional database URI
        
    Returns:
        True if successful, False otherwise
    """
    print("WARNING: This will delete all data in the database!")
    response = input("Are you sure you want to continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Database reset cancelled.")
        return False
    
    return initialize_database(database_uri, drop_existing=True)


def create_sample_data():
    """
    Create sample data for testing purposes.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from models.database import SessionLocal
        
        print("Creating sample data...")
        
        db = SessionLocal()
        
        # Create sample user
        sample_user = User(
            username="test_user",
            password="password123",
            preferences={
                "theme": "dark",
                "voice_enabled": True,
                "language": "en"
            }
        )
        db.add(sample_user)
        db.commit()
        db.refresh(sample_user)
        
        print(f"Created sample user: {sample_user.username} (ID: {sample_user.user_id})")
        
        # Create sample progress records
        subjects = ["Physics", "Chemistry", "Mathematics", "Biology"]
        chapters = {
            "Physics": ["Electromagnetism", "Mechanics", "Thermodynamics"],
            "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry"],
            "Mathematics": ["Calculus", "Algebra", "Trigonometry"],
            "Biology": ["Cell Biology", "Genetics", "Ecology"]
        }
        
        for subject in subjects:
            for chapter in chapters[subject]:
                progress = Progress(
                    user_id=sample_user.user_id,
                    subject=subject,
                    chapter=chapter,
                    topic=f"{chapter} - Topic 1"
                )
                # Simulate some attempts
                progress.record_attempt(True)
                progress.record_attempt(True)
                progress.record_attempt(False)
                
                db.add(progress)
        
        db.commit()
        print(f"Created {len(subjects) * 3} sample progress records.")
        
        # Create sample session
        sample_session = Session(user_id=sample_user.user_id)
        db.add(sample_session)
        db.commit()
        
        print(f"Created sample session: {sample_session.session_token}")
        
        db.close()
        
        print("Sample data created successfully.")
        return True
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return False


def main():
    """Main function for running migration commands."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database migration utilities for GuruAI')
    parser.add_argument(
        'command',
        choices=['init', 'verify', 'reset', 'sample'],
        help='Migration command to execute'
    )
    parser.add_argument(
        '--database-uri',
        help='Database URI (optional, uses config if not provided)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'init':
        success = initialize_database(args.database_uri)
        sys.exit(0 if success else 1)
    
    elif args.command == 'verify':
        # Initialize first
        init_db(args.database_uri)
        success = verify_database()
        sys.exit(0 if success else 1)
    
    elif args.command == 'reset':
        success = reset_database(args.database_uri)
        sys.exit(0 if success else 1)
    
    elif args.command == 'sample':
        # Initialize database first
        init_db(args.database_uri)
        success = create_sample_data()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
