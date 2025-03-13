import os
import uuid
import tempfile
import logging

logger = logging.getLogger(__name__)

def configure_app(app):
    """Configure Flask application with necessary settings"""
    # Set secret key for session security
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_' + str(uuid.uuid4()))
    
    # Configure session handling
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = tempfile.gettempdir()
    
    logger.info("Application configured with session type: %s", app.config['SESSION_TYPE'])
    logger.debug("Session directory: %s", app.config['SESSION_FILE_DIR'])
    
    return app

def get_deepseek_api_key():
    """Get the DeepSeek API key from environment variables or .env file"""
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    
    if not api_key:
        # Check if we can load from .env file directly as fallback
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('DEEPSEEK_API_KEY')
            if api_key:
                logger.info("API key loaded from .env file")
        except ImportError:
            logger.warning("python-dotenv not installed, cannot load from .env file")
    
    if api_key:
        # Mask key for logging
        masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 9 else "***"
        logger.info(f"DeepSeek API key configured: {masked_key}")
    else:
        logger.warning("DeepSeek API key not found in environment variables or .env file")
    
    return api_key or "dummy_key_replace_me"