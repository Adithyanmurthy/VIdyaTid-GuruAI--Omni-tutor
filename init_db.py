"""
Database initialization script for GuruAI.
Creates all necessary tables in the database.
"""
import logging
from models.database import init_db, create_tables
from models.user import User
from models.session import Session
from models.progress import Progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_database():
    """Initialize the database and create all tables."""
    try:
        logger.info("Initializing database...")
        
        # Initialize database engine
        engine = init_db()
        logger.info(f"Database engine initialized: {engine.url}")
        
        # Create all tables
        create_tables()
        logger.info("All tables created successfully")
        
        logger.info("Database initialization complete!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == '__main__':
    initialize_database()
