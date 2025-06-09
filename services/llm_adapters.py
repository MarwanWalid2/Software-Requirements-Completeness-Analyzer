import logging
import json
import traceback
from services.llm_client_factory import LLMClientFactory
import time
from datetime import datetime, timezone
import anthropic
logger = logging.getLogger(__name__)

class LLMAdapterBase:
    """Base adapter class for LLM requests"""
    
    def __init__(self, model_id):
        self.model_id = model_id
        self.model_info = LLMClientFactory.get_model_info(model_id)
        self.client = LLMClientFactory.get_client(model_id)
        self.model_name = self.model_info["model_name"]
        self.max_retries = 2
        self.retry_delay = 5  # seconds
    
    def generate_response(self, messages, temperature=1.0):
        """
        Generic method to generate a response from the LLM
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method")


class DeepSeekAdapter(LLMAdapterBase):
    """Adapter for DeepSeek LLM"""
    
    def generate_response(self, messages, temperature=1.0):
        """Generate a response from DeepSeek LLM"""
        logger.info(f"Generating response from DeepSeek model: {self.model_name}")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"DeepSeek request attempt {attempt+1}/{self.max_retries}")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # temperature=temperature
                )
                
                if not response.choices or not response.choices[0].message:
                    raise ValueError("Empty response from DeepSeek API")
                
                result = {
                    "model_id": self.model_id,
                    "content": response.choices[0].message.content,
                    "raw_response": response
                }
                
                return result
                
            except Exception as e:
                logger.error(f"DeepSeek API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All DeepSeek API attempts failed")
                    raise
        
        # This should not be reached due to the raise in the loop, but just in case
        raise RuntimeError("Failed to generate response from DeepSeek API")


class OpenAIAdapter(LLMAdapterBase):
    """Adapter for OpenAI LLM"""
    
    def generate_response(self, messages, temperature=1.0):
        """Generate a response from OpenAI LLM"""
        logger.info(f"Generating response from OpenAI model: {self.model_name}")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"OpenAI request attempt {attempt+1}/{self.max_retries}")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # temperature=temperature
                    reasoning_effort="high"
                )
                
                if not response.choices or not response.choices[0].message:
                    raise ValueError("Empty response from OpenAI API")
                
                result = {
                    "model_id": self.model_id,
                    "content": response.choices[0].message.content,
                    "raw_response": response
                }
                
                return result
                
            except Exception as e:
                logger.error(f"OpenAI API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All OpenAI API attempts failed")
                    raise
        
        # This should not be reached due to the raise in the loop, but just in case
        raise RuntimeError("Failed to generate response from OpenAI API")

class ClaudeAdapter(LLMAdapterBase):
    """Adapter for Claude/Anthropic LLM"""

    def __init__(self, config):
        super().__init__(config)
        self.rate_limit_reset_time = None
    
    def generate_response(self, messages, temperature=1.0):
        """Generate a response from Claude LLM"""
        logger.info(f"Generating response from Claude model: {self.model_name}")
        
        # Claude Sonnet 4 supports up to 64k output tokens
        max_tokens = 64000
        thinking_budget = 63999  
        
        # Check if we need to wait for rate limit reset
        if self.rate_limit_reset_time:
            current_time = datetime.now(timezone.utc)
            if current_time < self.rate_limit_reset_time:
                wait_time = (self.rate_limit_reset_time - current_time).total_seconds()
                logger.info(f"Rate limit hit, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time + 3) 

        for attempt in range(self.max_retries):
            try:
                # Convert OpenAI message format to Anthropic format
                system_message = None
                user_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        user_messages.append(msg)
                
                # Create the request parameters
                request_params = {
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    # "temperature": temperature,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": thinking_budget
                    }
                }
                
                # Add system message only if it exists
                if system_message:
                    request_params["system"] = system_message
                
                # Handle message formatting properly
                if len(user_messages) == 1 and user_messages[0]["role"] == "user":
                    # Single user message
                    request_params["messages"] = [
                        {"role": "user", "content": user_messages[0]["content"]}
                    ]
                else:
                    # Multiple messages - convert to Anthropic's format
                    # Anthropic requires alternating user/assistant messages
                    anthropic_messages = []
                    for msg in user_messages:
                        if msg["role"] == "user":
                            anthropic_messages.append({
                                "role": "user", 
                                "content": msg["content"]
                            })
                        elif msg["role"] == "assistant":
                            anthropic_messages.append({
                                "role": "assistant", 
                                "content": msg["content"]
                            })
                    
                    # Ensure we start with a user message
                    if anthropic_messages and anthropic_messages[0]["role"] != "user":
                        anthropic_messages.insert(0, {
                            "role": "user",
                            "content": "Please continue with the following:"
                        })
                    
                    request_params["messages"] = anthropic_messages
                
                # Make the API call
                response = self.client.messages.create(**request_params)
                self.rate_limit_reset_time = None
                # Extract content properly when thinking is enabled
                # Claude Sonnet 4 with thinking returns multiple content blocks
                content_text = ""
                thinking_content = ""
                
                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == 'text':
                            # Regular text content
                            content_text += content_block.text
                        elif content_block.type == 'thinking':
                            # Thinking content (internal reasoning)
                            thinking_content += getattr(content_block, 'content', str(content_block))
                    else:
                        # Fallback for unknown content types
                        if hasattr(content_block, 'text'):
                            content_text += content_block.text
                        else:
                            content_text += str(content_block)
                
                # If no regular text content found, log warning
                if not content_text and thinking_content:
                    logger.warning("Only thinking content found, no regular text output")
                    logger.debug(f"Thinking content preview: {thinking_content[:200]}...")
                
                result = {
                    "model_id": self.model_id,
                    "content": content_text,
                    "thinking_content": thinking_content,  # Store thinking for debugging
                    "raw_response": response
                }
                
                logger.info(f"Claude response received successfully (text length: {len(content_text)}, thinking length: {len(thinking_content)})")
                
                # Log thinking usage if available
                if hasattr(response, 'usage') and hasattr(response.usage, 'thinking_tokens'):
                    logger.info(f"Thinking tokens used: {response.usage.thinking_tokens}")
                
                return result
                
            except anthropic.RateLimitError as e:
                # Extract rate limit info from error
                error_dict = e.response.json() if hasattr(e.response, 'json') else {}
                
                # Get retry-after header if available
                if hasattr(e.response, 'headers'):
                    retry_after = e.response.headers.get('retry-after')
                    reset_time = e.response.headers.get('anthropic-ratelimit-output-tokens-reset')
                    
                    if retry_after:
                        wait_time = int(retry_after) + 5  # Add buffer
                        logger.info(f"Rate limit hit, waiting {wait_time} seconds")
                        time.sleep(wait_time)
                    elif reset_time:
                        # Parse ISO format reset time
                        self.rate_limit_reset_time = datetime.fromisoformat(reset_time.replace('Z', '+00:00'))
                        wait_time = (self.rate_limit_reset_time - datetime.now(timezone.utc)).total_seconds() + 5
                        logger.info(f"Rate limit hit, waiting until {reset_time} ({wait_time:.1f} seconds)")
                        time.sleep(max(wait_time, 0))
                    else:
                        # Default wait time
                        wait_time = 70
                        logger.info(f"Rate limit hit, waiting {wait_time} seconds (default)")
                        time.sleep(wait_time)
                else:
                    # If no headers, wait a default time
                    time.sleep(70)
                
                # Retry if we have attempts left
                if attempt < self.max_retries - 1:
                    continue
                else:
                    raise
        
        # This should not be reached due to the raise in the loop, but just in case
        raise RuntimeError("Failed to generate response from Claude API")


def get_adapter(model_id):
    """Factory function to get the appropriate adapter for a model"""
    if model_id == "deepseek":
        return DeepSeekAdapter(model_id)
    elif model_id == "openai":
        return OpenAIAdapter(model_id)
    elif model_id == "claude":
        return ClaudeAdapter(model_id)
    else:
        raise ValueError(f"Unknown model ID: {model_id}")