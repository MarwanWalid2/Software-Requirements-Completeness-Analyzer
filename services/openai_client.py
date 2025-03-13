import openai
import logging
from config import get_openai_api_key

logger = logging.getLogger(__name__)

def initialize_openai_client():
    """Initialize and return an OpenAI client"""
    api_key = get_openai_api_key()
    
    # Initialize OpenAI client with explicit API key
    client = openai.OpenAI(
        api_key=api_key,
        timeout=200.0
    )
    
    logger.info("OpenAI client initialized")
    return client