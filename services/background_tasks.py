"""
Background task manager for GuruAI.
Handles periodic tasks like model idle checking and cleanup.
"""
import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """
    Manages background tasks for the application.
    
    Features:
    - Periodic model idle checking
    - Graceful shutdown
    - Thread-safe operation
    """
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize the background task manager.
        
        Args:
            check_interval: Seconds between idle checks (default: 60)
        """
        self.check_interval = check_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.model_manager = None
        
        logger.info(f"BackgroundTaskManager initialized (check_interval={check_interval}s)")
    
    def set_model_manager(self, model_manager):
        """
        Set the model manager to monitor.
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        logger.info("Model manager registered with background task manager")
    
    def _run_tasks(self):
        """Internal method that runs the background tasks."""
        logger.info("Background task thread started")
        
        while self.running:
            try:
                # Check if model should be unloaded
                if self.model_manager is not None:
                    unloaded = self.model_manager.check_idle_and_unload()
                    if unloaded:
                        logger.info("Model auto-unloaded due to idle timeout")
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in background task: {e}")
                time.sleep(self.check_interval)
        
        logger.info("Background task thread stopped")
    
    def start(self):
        """Start the background task thread."""
        if self.running:
            logger.warning("Background tasks already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_tasks, daemon=True)
        self.thread.start()
        logger.info("Background tasks started")
    
    def stop(self):
        """Stop the background task thread."""
        if not self.running:
            logger.warning("Background tasks not running")
            return
        
        logger.info("Stopping background tasks...")
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logger.warning("Background task thread did not stop gracefully")
            else:
                logger.info("Background tasks stopped")
    
    def is_running(self) -> bool:
        """Check if background tasks are running."""
        return self.running


# Global instance
_background_manager: Optional[BackgroundTaskManager] = None


def get_background_manager(check_interval: int = 60) -> BackgroundTaskManager:
    """
    Get or create the global background task manager.
    
    Args:
        check_interval: Seconds between checks (only used on first call)
    
    Returns:
        BackgroundTaskManager instance
    """
    global _background_manager
    
    if _background_manager is None:
        _background_manager = BackgroundTaskManager(check_interval)
    
    return _background_manager


def initialize_background_tasks(model_manager, check_interval: int = 60):
    """
    Initialize and start background tasks.
    
    Args:
        model_manager: ModelManager instance to monitor
        check_interval: Seconds between checks
    """
    manager = get_background_manager(check_interval)
    manager.set_model_manager(model_manager)
    manager.start()
    logger.info("Background tasks initialized and started")


def shutdown_background_tasks():
    """Shutdown background tasks gracefully."""
    global _background_manager
    
    if _background_manager is not None:
        _background_manager.stop()
        _background_manager = None
        logger.info("Background tasks shut down")
