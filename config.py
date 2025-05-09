import os
import uuid
import tempfile
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

def configure_app(app):
    """Configure Flask application with necessary settings"""
    # Set secret key for session security
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_' + str(uuid.uuid4()))
    
    # This filesystem session configuration will work on Heroku
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = tempfile.gettempdir()
    
    # Set Flask to production mode on Heroku
    if os.environ.get('FLASK_CONFIG') == 'production':
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
    
    logger.info("Application configured with session type: %s", app.config['SESSION_TYPE'])
    
    return app


def get_deepseek_api_key():
    """Get the DeepSeek API key from environment variables or .env file"""
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    
    if api_key:
        # Mask key for logging
        masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 9 else "***"
        logger.info(f"DeepSeek API key configured: {masked_key}")
    else:
        logger.warning("DeepSeek API key not found in environment variables or .env file")
    
    return api_key or "dummy_key_replace_me"

def get_openai_api_key():
    """Get the OpenAI API key from environment variables or .env file"""
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if api_key:
        # Mask key for logging
        masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 9 else "***"
        logger.info(f"OpenAI API key configured: {masked_key}")
    else:
        logger.warning("OpenAI API key not found in environment variables or .env file")
    
    return api_key or "dummy_key_replace_me"

def get_anthropic_api_key():
    """Get the Anthropic/Claude API key from environment variables or .env file"""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if api_key:
        # Mask key for logging
        masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 9 else "***"
        logger.info(f"Anthropic/Claude API key configured: {masked_key}")
    else:
        logger.warning("Anthropic/Claude API key not found in environment variables or .env file")
    
    return api_key or "dummy_key_replace_me"

def get_available_models():
    """Get configuration for all available models"""
    models = {
        "deepseek": {
            "name": "DeepSeek Chat",
            "id": "deepseek",
            "model_name": "deepseek-chat",
            "api_key": get_deepseek_api_key(),
            "enabled": bool(get_deepseek_api_key() and get_deepseek_api_key() != "dummy_key_replace_me")
        },
        "openai": {
            "name": "OpenAI o4 mini",
            "id": "openai",
            "model_name": "o4-mini",
            "api_key": get_openai_api_key(),
            "enabled": bool(get_openai_api_key() and get_openai_api_key() != "dummy_key_replace_me")
        },
        "claude": {
            "name": "Claude 3.7 Sonnet",
            "id": "claude",
            "model_name": "claude-3-7-sonnet-20250219",
            "api_key": get_anthropic_api_key(),
            "enabled": bool(get_anthropic_api_key() and get_anthropic_api_key() != "dummy_key_replace_me")
        }
    }
    
    # Check which models are available based on API keys
    available_models = {model_id: model for model_id, model in models.items() if model["enabled"]}
    
    if not available_models:
        logger.warning("No LLM models are available with valid API keys")
        # Return at least one model as a fallback, even if the key is dummy
        return {"deepseek": models["deepseek"]}
    
    return available_models

def get_available_meta_models():
    """Get configuration for the available meta models (models used for aggregation)"""
    meta_models = {
        "majority": {
            "name": "Majority Vote",
            "id": "majority",
            "description": "Simple majority voting for model outputs"
        },
        "weighted": {
            "name": "Weighted Vote",
            "id": "weighted",
            "description": "Weighted voting based on model confidence"
        },
        "openai": {
            "name": "OpenAI Meta-Analysis",
            "id": "openai_meta",
            "description": "Use OpenAI to analyze and combine the results from other models",
            "enabled": bool(get_openai_api_key() and get_openai_api_key() != "dummy_key_replace_me"),
            "model_name": "o4-mini"
        },
        "deepseek": {
            "name": "DeepSeek Meta-Analysis",
            "id": "deepseek_meta",
            "description": "Use DeepSeek to analyze and combine the results from other models",
            "enabled": bool(get_deepseek_api_key() and get_deepseek_api_key() != "dummy_key_replace_me"),
            "model_name": "deepseek-chat"
        },
        "claude": {
            "name": "Claude Meta-Analysis",
            "id": "claude_meta",
            "description": "Use Claude to analyze and combine the results from other models",
            "enabled": bool(get_anthropic_api_key() and get_anthropic_api_key() != "dummy_key_replace_me"),
            "model_name": "claude-3-7-sonnet-20250219"
        }
    }
    
    # Always include the non-LLM based meta-models
    available_meta_models = {
        "majority": meta_models["majority"],
        "weighted": meta_models["weighted"]
    }
    
    # Add LLM-based meta-models if API keys are available
    for model_id in ["openai", "deepseek", "claude"]:
        if meta_models[model_id].get("enabled", False):
            available_meta_models[model_id] = meta_models[model_id]
    
    return available_meta_models