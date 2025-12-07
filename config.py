"""
Configuration settings for GuruAI application.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Flask Configuration
class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/guruai.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File Upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
    
    # AI Models - Local (Fallback)
    MODEL_DIR = BASE_DIR / 'ai_models'
    LLM_MODEL_PATH = MODEL_DIR / 'qwen2.5-7b-instruct-q4_k_m.gguf'
    EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # 768 dims to match vector DB
    
    # Google Gemini Configuration (Free - Best Quality)
    USE_GEMINI = os.getenv('USE_GEMINI', 'false').lower() == 'true'
    
    # Multiple API keys support (comma-separated for automatic rotation)
    # System will automatically switch when one key hits rate limit
    GEMINI_API_KEYS = [k.strip() for k in os.getenv('GEMINI_API_KEYS', '').split(',') if k.strip()]
    
    # Legacy single key support (will be used if GEMINI_API_KEYS is not set)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # ElevenLabs Configuration (Voice Output)
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')
    
    # Separate ElevenLabs key for video generation (to manage quota separately)
    ELEVENLABS_VIDEO_API_KEY = os.getenv('ELEVENLABS_VIDEO_API_KEY', '')
    
    # Cloudflare Workers AI Configuration
    USE_CLOUDFLARE_AI = os.getenv('USE_CLOUDFLARE_AI', 'false').lower() == 'true'
    CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID', '')
    CLOUDFLARE_API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN', '')
    
    # Cloudflare AI Models
    CLOUDFLARE_CHAT_MODEL = '@cf/meta/llama-3.1-8b-instruct'
    CLOUDFLARE_EMBEDDING_MODEL = '@cf/baai/bge-base-en-v1.5'
    CLOUDFLARE_IMAGE_MODEL = '@cf/microsoft/resnet-50'
    
    # LLM Configuration
    LLM_N_CTX = 4096  # Context window size (Qwen2.5 supports up to 32k)
    LLM_N_GPU_LAYERS = 0  # Number of layers to offload to GPU (0 for CPU only)
    LLM_TEMPERATURE = 0.7  # Default temperature for generation (Qwen2.5 works well at 0.7)
    LLM_MAX_TOKENS = 512  # Default max tokens for generation
    
    # Model Manager Settings
    MODEL_IDLE_TIMEOUT = 600  # 10 minutes in seconds
    MODEL_MEMORY_LIMIT = 4 * 1024 * 1024 * 1024  # 4GB
    
    # Settings Configuration
    MIN_MEMORY_LIMIT = 2  # GB
    MAX_MEMORY_LIMIT = 16  # GB
    MIN_IDLE_TIMEOUT = 1  # minutes
    MAX_IDLE_TIMEOUT = 60  # minutes
    DEFAULT_MEMORY_LIMIT = 6  # GB
    DEFAULT_IDLE_TIMEOUT = 10  # minutes
    
    # Vector Store
    VECTOR_STORE_DIR = BASE_DIR / 'vector_store'
    CHROMA_DB_PATH = VECTOR_STORE_DIR  # Use vector_store for consistency
    CHROMA_COLLECTION_NAME = 'ncert_content'
    
    # NCERT Content
    NCERT_CONTENT_DIR = BASE_DIR / 'ncert_content'
    NCERT_PDF_DIR = NCERT_CONTENT_DIR / 'pdfs'
    DIAGRAMS_DIR = BASE_DIR / 'diagrams'
    PREVIOUS_PAPERS_DIR = BASE_DIR / 'previous_papers'
    
    # RAG Settings
    RAG_TOP_K = 3  # Reduced for faster processing
    RAG_CHUNK_SIZE = 500
    RAG_CHUNK_OVERLAP = 50
    
    # OCR Settings
    TESSERACT_CMD = os.getenv('TESSERACT_CMD', 'tesseract')
    OCR_LANGUAGES = ['eng']
    
    # Voice Settings
    VOICE_INPUT_ENABLED = True
    VOICE_OUTPUT_ENABLED = True
    SUPPORTED_VOICE_LANGUAGES = ['en', 'hi', 'hinglish']
    DEFAULT_VOICE_LANGUAGE = 'en'
    DEFAULT_VOICE_SPEED = 1.0
    VOICE_RECOGNITION_TIMEOUT = 10  # seconds
    
    # Handwriting Recognition Settings
    HANDWRITING_RECOGNITION_ENABLED = True
    HANDWRITING_CONFIDENCE_THRESHOLD = 0.7
    SUPPORTED_HANDWRITING_SCRIPTS = ['latin', 'devanagari']
    
    # Deep Explanation Settings
    DEEP_EXPLANATION_ENABLED = True
    DEEP_EXPLANATION_LAYERS = ['basic', 'intermediate', 'advanced']
    MIN_EXAMPLES_PER_EXPLANATION = 2
    
    # Peer Comparison Settings
    PEER_COMPARISON_ENABLED = True
    PEER_COMPARISON_OPT_IN_DEFAULT = False
    PEER_STATS_UPDATE_INTERVAL = 3600  # 1 hour in seconds
    ANONYMIZATION_SALT = os.getenv('ANONYMIZATION_SALT', 'change-this-salt')
    
    # Session Settings
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours
    
    # Security
    BCRYPT_LOG_ROUNDS = 12
    MAX_LOGIN_ATTEMPTS = 5
    ACCOUNT_LOCKOUT_DURATION = 900  # 15 minutes


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # Cloudflare D1 Database Configuration
    USE_CLOUDFLARE_D1 = os.getenv('USE_CLOUDFLARE_D1', 'false').lower() == 'true'
    CLOUDFLARE_D1_DATABASE_ID = os.getenv('CLOUDFLARE_D1_DATABASE_ID', '')
    
    # Cloudflare Workers AI Configuration (override base config for production)
    USE_CLOUDFLARE_AI = os.getenv('USE_CLOUDFLARE_AI', 'true').lower() == 'true'
    # CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN inherited from base Config
    
    # Cloudflare R2 Storage Configuration
    CLOUDFLARE_R2_BUCKET = os.getenv('CLOUDFLARE_R2_BUCKET', '')
    CLOUDFLARE_R2_ACCESS_KEY = os.getenv('CLOUDFLARE_R2_ACCESS_KEY', '')
    CLOUDFLARE_R2_SECRET_KEY = os.getenv('CLOUDFLARE_R2_SECRET_KEY', '')
    CLOUDFLARE_R2_ENDPOINT = os.getenv('CLOUDFLARE_R2_ENDPOINT', '')
    
    # Cloudflare KV Namespace Configuration
    CLOUDFLARE_KV_NAMESPACE = os.getenv('CLOUDFLARE_KV_NAMESPACE', '')
    CLOUDFLARE_KV_NAMESPACE_ID = os.getenv('CLOUDFLARE_KV_NAMESPACE_ID', '')
    
    # Performance and Caching Settings
    CACHE_TTL = 300  # 5 minutes for subscription data
    TIER_CONFIG_CACHE_TTL = 3600  # 1 hour for tier configuration
    ENABLE_EDGE_CACHING = True
    
    # Production-specific settings
    SQLALCHEMY_ECHO = False  # Disable SQL query logging in production
    SQLALCHEMY_POOL_SIZE = 10
    SQLALCHEMY_POOL_RECYCLE = 3600
    SQLALCHEMY_MAX_OVERFLOW = 20


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
