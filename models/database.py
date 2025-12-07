"""
Database configuration and session management for GuruAI.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from config import Config

# Create the declarative base
Base = declarative_base()

# Database engine (will be initialized by init_db)
engine = None
SessionLocal = None


def init_db(database_uri=None):
    """
    Initialize the database engine and session factory.
    
    Args:
        database_uri: Optional database URI. If not provided, uses Config.SQLALCHEMY_DATABASE_URI
    """
    global engine, SessionLocal
    
    if database_uri is None:
        database_uri = Config.SQLALCHEMY_DATABASE_URI
    
    engine = create_engine(
        database_uri,
        connect_args={"check_same_thread": False} if "sqlite" in database_uri else {},
        echo=False
    )
    
    SessionLocal = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    
    return engine


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all tables in the database."""
    Base.metadata.drop_all(bind=engine)


def get_db():
    """
    Get a database session.
    Yields a session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
