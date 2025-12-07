import os
from flask import Flask, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging using custom logger
from utils.logger import setup_logging
logger = setup_logging(
    app_name='guruai',
    log_level=os.getenv('LOG_LEVEL', 'INFO'),
    console_output=True,
    file_output=True,
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5
)

# Create Flask app
app = Flask(__name__)
app.config.from_object('config.Config')

# Enable CORS
CORS(app)

# Register error handlers
from utils.flask_error_handlers import register_error_handlers, register_request_logging
include_traceback = app.config.get('DEBUG', False)
register_error_handlers(app, include_traceback=include_traceback)
register_request_logging(app)
logger.info("Error handlers and request logging registered")

# Initialize database
from models.database import init_db, create_tables
init_db()
create_tables()
logger.info("Database initialized")

# Import and register blueprints
from routes.query_routes import query_bp, init_query_handler
from routes.problem_routes import problem_bp, init_problem_solver
from routes.progress_routes import progress_bp, init_progress_tracker
from routes.search_routes import search_bp, init_search_service
from routes.auth_routes import auth_bp
from routes.paper_routes import paper_bp, init_question_generator
from routes.diagram_routes import diagram_bp, init_diagram_service
from routes.settings_routes import settings_bp
from routes.offline_routes import offline_bp
from routes.prediction_routes import prediction_bp
from routes.payment_routes import payment_bp
from routes.subscription_routes import subscription_bp
from routes.usage_routes import usage_bp
from routes.user_routes import user_bp
from routes.exam_routes import exam_bp
from routes.voice_routes import voice_bp
from video_generator.routes import video_bp

app.register_blueprint(query_bp)
app.register_blueprint(problem_bp)
app.register_blueprint(progress_bp)
app.register_blueprint(search_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(paper_bp)
app.register_blueprint(diagram_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(offline_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(payment_bp)
app.register_blueprint(subscription_bp)
app.register_blueprint(usage_bp)
app.register_blueprint(user_bp)
app.register_blueprint(exam_bp)
app.register_blueprint(voice_bp)
app.register_blueprint(video_bp)

# Lazy initialization flag
_services_initialized = {
    'query_handler': False,
    'problem_solver': False,
    'progress_tracker': False,
    'search_service': False,
    'question_generator': False,
    'diagram_service': False,
    'offline_mode': False,
    'scheduled_tasks': False
}

def init_services_lazy():
    """
    Initialize services on first request (lazy loading).
    This significantly speeds up app startup time.
    """
    # Only initialize lightweight services at startup
    with app.app_context():
        # Initialize progress tracker (lightweight - just database)
        if not _services_initialized['progress_tracker']:
            try:
                init_progress_tracker()
                _services_initialized['progress_tracker'] = True
                logger.info("Progress tracker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize progress tracker: {e}")
        
        # Initialize diagram service (lightweight - just database)
        if not _services_initialized['diagram_service']:
            try:
                init_diagram_service()
                _services_initialized['diagram_service'] = True
                logger.info("Diagram service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize diagram service: {e}")
        
        # Enable offline mode (lightweight)
        if not _services_initialized['offline_mode']:
            try:
                from utils.offline_monitor import get_offline_verifier
                verifier = get_offline_verifier()
                verifier.enable_offline_mode()
                _services_initialized['offline_mode'] = True
                logger.info("Offline mode enabled")
            except Exception as e:
                logger.error(f"Failed to enable offline mode: {e}")
        
        # Start scheduled tasks (lightweight)
        if not _services_initialized['scheduled_tasks']:
            try:
                from services.scheduled_tasks import start_scheduled_tasks
                start_scheduled_tasks()
                _services_initialized['scheduled_tasks'] = True
                logger.info("Scheduled tasks started")
            except Exception as e:
                logger.error(f"Failed to start scheduled tasks: {e}")

def init_heavy_services():
    """
    Initialize heavy services (LLM, RAG) only when needed.
    Called on first API request that needs these services.
    """
    with app.app_context():
        # Initialize query handler (heavy - loads LLM and RAG)
        if not _services_initialized['query_handler']:
            try:
                logger.info("Initializing query handler (this may take a moment)...")
                init_query_handler()
                _services_initialized['query_handler'] = True
                logger.info("Query handler initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize query handler: {e}")
        
        # Initialize problem solver (heavy - uses LLM)
        if not _services_initialized['problem_solver']:
            try:
                logger.info("Initializing problem solver...")
                init_problem_solver()
                _services_initialized['problem_solver'] = True
                logger.info("Problem solver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize problem solver: {e}")
        
        # Initialize search service (medium - loads vector store)
        if not _services_initialized['search_service']:
            try:
                logger.info("Initializing search service...")
                init_search_service()
                _services_initialized['search_service'] = True
                logger.info("Search service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize search service: {e}")
        
        # Initialize question generator (medium)
        if not _services_initialized['question_generator']:
            try:
                logger.info("Initializing question generator...")
                init_question_generator()
                _services_initialized['question_generator'] = True
                logger.info("Question generator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize question generator: {e}")

# Initialize lightweight services at startup
logger.info("Initializing lightweight services...")
init_services_lazy()
logger.info("App startup complete! Heavy services will load on first use.")

# Middleware to initialize heavy services on first API request
@app.before_request
def before_request():
    """Initialize heavy services on first API request."""
    from flask import request
    
    # Check if this is an API request that needs heavy services
    if request.path.startswith('/api/'):
        # Skip health check and settings endpoints
        if request.path in ['/api/health', '/api/settings', '/api/system/']:
            return
        
        # Initialize heavy services if not already done
        if not all([
            _services_initialized['query_handler'],
            _services_initialized['problem_solver'],
            _services_initialized['search_service'],
            _services_initialized['question_generator']
        ]):
            logger.info("First API request detected - initializing heavy services...")
            init_heavy_services()


@app.route('/')
def home():
    """Renders the home/landing page."""
    return render_template('home.html')


@app.route('/chat')
def chat():
    """Renders the main chat interface."""
    return render_template('index.html')


@app.route('/progress')
def progress_page():
    """Renders the progress tracking page."""
    return render_template('progress.html')


@app.route('/search')
def search_page():
    """Renders the search page."""
    return render_template('search.html')


@app.route('/question-paper')
def question_paper_page():
    """Renders the question paper generation page."""
    return render_template('question-paper.html')


@app.route('/predictions')
def predictions_page():
    """Renders the AI predictions page."""
    return render_template('predictions.html')


@app.route('/settings')
def settings_page():
    """Renders the settings page."""
    return render_template('settings.html')


@app.route('/profile')
def profile_page():
    """Renders the user profile page."""
    return render_template('profile.html')


@app.route('/pricing')
def pricing_page():
    """Renders the pricing/subscription page."""
    return render_template('pricing.html')


@app.route('/video-generator')
def video_generator_page():
    """Renders the video generator page (alternative route)."""
    return render_template('video_generator.html')


@app.route('/health')
def health():
    """Basic health check endpoint."""
    return {'status': 'ok', 'message': 'GuruAI is running'}, 200


# Cleanup handler for graceful shutdown
import atexit

def cleanup():
    """Cleanup function called on app shutdown."""
    logger.info("Shutting down application...")
    
    # Stop scheduled tasks
    if _services_initialized.get('scheduled_tasks', False):
        try:
            from services.scheduled_tasks import stop_scheduled_tasks
            stop_scheduled_tasks()
            logger.info("Scheduled tasks stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduled tasks: {e}")

# Register cleanup handler
atexit.register(cleanup)


if __name__ == '__main__':
    try:
        app.run(debug=True, port=5001)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        cleanup()
