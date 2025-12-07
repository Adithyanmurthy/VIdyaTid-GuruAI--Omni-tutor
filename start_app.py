"""
GuruAI Startup Script with Progress Indicator

Usage:
    python start_app.py [--mode fast|normal|full] [--port 5001]
"""

import sys
import time
import argparse
from pathlib import Path

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║                    GuruAI - Offline Tutor                 ║
    ║                  JEE & NEET AI Study Partner              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_progress(message, status="info"):
    """Print progress message with status indicator."""
    icons = {
        "info": "ℹ",
        "success": "✓",
        "error": "✗",
        "loading": "⏳"
    }
    icon = icons.get(status, "•")
    print(f"  {icon} {message}")

def check_requirements():
    """Check if all requirements are met."""
    print_progress("Checking requirements...", "loading")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print_progress("Python 3.8+ required", "error")
        return False
    print_progress(f"Python {sys.version_info.major}.{sys.version_info.minor} detected", "success")
    
    # Check required files
    required_files = [
        "app.py",
        "config.py",
        "requirements.txt"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print_progress(f"Missing required file: {file}", "error")
            return False
    print_progress("All required files present", "success")
    
    # Check required directories
    required_dirs = [
        "templates",
        "static",
        "routes",
        "services"
    ]
    
    for dir in required_dirs:
        if not Path(dir).exists():
            print_progress(f"Missing required directory: {dir}", "error")
            return False
    print_progress("All required directories present", "success")
    
    return True

def check_model():
    """Check if LLM model exists (optional - using Gemini API)."""
    print_progress("Checking AI configuration...", "loading")
    
    try:
        import os
        
        # Check if using Gemini (preferred)
        if os.getenv('USE_GEMINI', 'false').lower() == 'true':
            # Check for multiple API keys first
            api_keys_str = os.getenv('GEMINI_API_KEYS', '')
            gemini_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
            
            # Fall back to single key
            if not gemini_keys:
                single_key = os.getenv('GEMINI_API_KEY', '')
                if single_key:
                    gemini_keys = [single_key]
            
            if gemini_keys:
                key_count = len(gemini_keys)
                rotation_status = "with key rotation" if key_count > 1 else ""
                print_progress(f"Using Google Gemini API ({key_count} key(s)) {rotation_status}", "success")
                return True
            else:
                print_progress("Gemini API key not configured", "error")
                return False
        
        # Check if using Cloudflare
        if os.getenv('USE_CLOUDFLARE_AI', 'false').lower() == 'true':
            print_progress("Using Cloudflare Workers AI (cloud-based)", "success")
            return True
        
        # Check for local model (fallback)
        from config import Config
        model_path = Path(Config.LLM_MODEL_PATH)
        
        if model_path.exists():
            size_gb = model_path.stat().st_size / (1024**3)
            print_progress(f"Local model found: {model_path.name} ({size_gb:.1f} GB)", "success")
            return True
        else:
            print_progress("No AI configuration found (Gemini/Cloudflare/Local)", "error")
            print_progress("Configure GEMINI_API_KEY in .env file", "info")
            return False
    except Exception as e:
        print_progress(f"Error checking AI config: {e}", "error")
        return False

def check_database():
    """Check if database exists."""
    print_progress("Checking database...", "loading")
    
    db_file = Path("guruai.db")
    if db_file.exists():
        size_mb = db_file.stat().st_size / (1024**2)
        print_progress(f"Database found ({size_mb:.1f} MB)", "success")
    else:
        print_progress("Database will be created on first run", "info")
    
    return True

def start_app(mode="fast", port=5001, debug=True):
    """Start the Flask application."""
    print_progress(f"Starting GuruAI in {mode.upper()} mode...", "loading")
    print_progress(f"Server will run on http://localhost:{port}", "info")
    print()
    
    # Set environment variable for startup mode
    import os
    os.environ['GURUAI_STARTUP_MODE'] = mode
    
    # Import and run app
    try:
        from app import app
        print_progress("Flask app loaded successfully", "success")
        print()
        print("="*60)
        print(f"  🚀 GuruAI is running at http://localhost:{port}")
        print(f"  📊 Mode: {mode.upper()}")
        print(f"  🔧 Debug: {'ON' if debug else 'OFF'}")
        print("="*60)
        print()
        print("  Press Ctrl+C to stop the server")
        print()
        
        app.run(debug=debug, port=port, host='0.0.0.0')
        
    except KeyboardInterrupt:
        print()
        print_progress("Server stopped by user", "info")
    except Exception as e:
        print_progress(f"Error starting app: {e}", "error")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Start GuruAI application')
    parser.add_argument('--mode', choices=['fast', 'normal', 'full'], default='fast',
                       help='Startup mode (default: fast)')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to run on (default: 5001)')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug mode')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip pre-flight checks')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Run pre-flight checks
    if not args.skip_checks:
        print("Running pre-flight checks...")
        print()
        
        if not check_requirements():
            print()
            print_progress("Pre-flight checks failed", "error")
            sys.exit(1)
        
        check_model()  # Warning only, not fatal
        check_database()
        
        print()
        print_progress("Pre-flight checks complete", "success")
        print()
    
    # Start the app
    start_app(
        mode=args.mode,
        port=args.port,
        debug=not args.no_debug
    )

if __name__ == "__main__":
    main()
