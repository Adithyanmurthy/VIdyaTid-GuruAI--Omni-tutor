"""
Startup Configuration for GuruAI

Controls which services are loaded at startup vs on-demand.
Modify these settings to optimize startup time based on your needs.
"""

# Startup modes
STARTUP_MODE = "fast"  # Options: "fast", "normal", "full"

# Service loading configuration
SERVICES_CONFIG = {
    "fast": {
        # Fast startup - only essential services
        "load_at_startup": [
            "database",
            "progress_tracker",
            "diagram_service",
            "offline_mode"
        ],
        "load_on_demand": [
            "query_handler",  # Heavy - LLM + RAG
            "problem_solver",  # Heavy - LLM
            "search_service",  # Medium - Vector store
            "question_generator"  # Medium
        ]
    },
    "normal": {
        # Normal startup - lightweight + medium services
        "load_at_startup": [
            "database",
            "progress_tracker",
            "diagram_service",
            "search_service",
            "question_generator",
            "offline_mode"
        ],
        "load_on_demand": [
            "query_handler",  # Heavy - LLM + RAG
            "problem_solver"  # Heavy - LLM
        ]
    },
    "full": {
        # Full startup - all services (slowest but ready immediately)
        "load_at_startup": [
            "database",
            "query_handler",
            "problem_solver",
            "progress_tracker",
            "search_service",
            "question_generator",
            "diagram_service",
            "offline_mode"
        ],
        "load_on_demand": []
    }
}

# Get current configuration
def get_startup_config():
    """Get the current startup configuration."""
    return SERVICES_CONFIG.get(STARTUP_MODE, SERVICES_CONFIG["fast"])

# Service initialization timeouts (seconds)
SERVICE_TIMEOUTS = {
    "query_handler": 60,  # LLM loading can take time
    "problem_solver": 30,
    "search_service": 20,
    "question_generator": 15,
    "diagram_service": 5,
    "progress_tracker": 5,
    "offline_mode": 2
}

# Model loading configuration
MODEL_CONFIG = {
    "preload_model": False,  # Set to True to load LLM at startup
    "use_gpu": True,  # Use GPU if available
    "n_gpu_layers": -1,  # -1 = use all GPU layers, 0 = CPU only
    "n_ctx": 4096,  # Context window size
    "n_threads": 4,  # CPU threads for inference
}

# Cache configuration
CACHE_CONFIG = {
    "enable_response_cache": True,
    "cache_size_mb": 500,
    "cache_ttl_hours": 24
}

# Performance tuning
PERFORMANCE_CONFIG = {
    "lazy_load_embeddings": True,  # Load embeddings on first use
    "batch_size": 32,
    "max_workers": 4,  # For parallel processing
    "enable_profiling": False  # Enable to debug slow operations
}

# Startup messages
STARTUP_MESSAGES = {
    "fast": "⚡ Fast startup mode - Heavy services will load on first use",
    "normal": "🚀 Normal startup mode - Core services ready",
    "full": "🔥 Full startup mode - All services loaded (may take longer)"
}

def get_startup_message():
    """Get the startup message for current mode."""
    return STARTUP_MESSAGES.get(STARTUP_MODE, STARTUP_MESSAGES["fast"])

# Development mode settings
DEV_MODE = {
    "auto_reload": True,
    "debug": True,
    "show_sql": False,
    "log_level": "INFO"
}

# Production mode settings
PROD_MODE = {
    "auto_reload": False,
    "debug": False,
    "show_sql": False,
    "log_level": "WARNING"
}
