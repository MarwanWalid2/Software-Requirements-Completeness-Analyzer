import anthropic
import logging
from config import get_anthropic_api_key

logger = logging.getLogger(__name__)

def initialize_claude_client():
    """Initialize and return an Anthropic/Claude client"""
    api_key = get_anthropic_api_key()
    
    # Initialize Anthropic client with explicit API key
    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=200.0
    )
    
    logger.info("Anthropic/Claude client initialized")
    return client