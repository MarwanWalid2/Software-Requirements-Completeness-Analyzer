import openai
import logging
from config import get_deepseek_api_key

logger = logging.getLogger(__name__)

def initialize_deepseek_client():
    """Initialize and return an OpenAI client configured for DeepSeek"""
    api_key = get_deepseek_api_key()
    
    # Initialize OpenAI client with explicit API key
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=400.0
    )
    
    logger.info("OpenAI client initialized with DeepSeek base URL")
    return client