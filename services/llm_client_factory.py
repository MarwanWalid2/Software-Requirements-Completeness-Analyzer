import logging
from services.deepseek_client import initialize_deepseek_client
from services.openai_client import initialize_openai_client
from services.claude_client import initialize_claude_client
from config import get_available_models

logger = logging.getLogger(__name__)

class LLMClientFactory:
    """Factory class for managing LLM clients"""
    
    _clients = {}  # Cache for initialized clients
    
    @classmethod
    def get_client(cls, model_id):
        """Get or initialize a client for the specified model"""
        if model_id in cls._clients:
            logger.debug(f"Using cached client for {model_id}")
            return cls._clients[model_id]
        
        logger.info(f"Initializing new client for {model_id}")
        
        # Initialize the appropriate client based on model_id
        if model_id == "deepseek":
            client = initialize_deepseek_client()
        elif model_id == "openai":
            client = initialize_openai_client()
        elif model_id == "claude":
            client = initialize_claude_client()
        else:
            logger.error(f"Unknown model ID: {model_id}")
            raise ValueError(f"Unknown model ID: {model_id}")
        
        # Cache the client for future use
        cls._clients[model_id] = client
        return client
    
    @classmethod
    def get_model_info(cls, model_id):
        """Get model configuration information"""
        available_models = get_available_models()
        if model_id in available_models:
            return available_models[model_id]
        
        logger.error(f"Unknown model ID: {model_id}")
        raise ValueError(f"Unknown model ID: {model_id}")
    
    @classmethod
    def get_available_models(cls):
        """Get available models"""
        return get_available_models()