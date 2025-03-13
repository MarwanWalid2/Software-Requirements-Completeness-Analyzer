import logging
import json
import time
import traceback
from services.llm_client_factory import LLMClientFactory

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
    
    def generate_response(self, messages, temperature=0.5):
        """
        Generic method to generate a response from the LLM
        To be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement this method")


class DeepSeekAdapter(LLMAdapterBase):
    """Adapter for DeepSeek LLM"""
    
    def generate_response(self, messages, temperature=0.5):
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
    
    def generate_response(self, messages, temperature=0.7):
        """Generate a response from OpenAI LLM"""
        logger.info(f"Generating response from OpenAI model: {self.model_name}")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"OpenAI request attempt {attempt+1}/{self.max_retries}")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # temperature=temperature
                    reasoning_effort="medium"
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
    
    def generate_response(self, messages, temperature=0.7):
        """Generate a response from Claude LLM"""
        logger.info(f"Generating response from Claude model: {self.model_name}")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Claude request attempt {attempt+1}/{self.max_retries}")
                
                # Convert OpenAI message format to Anthropic format
                system_message = None
                user_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        user_messages.append(msg)
                
                # If there's only one user message, use it directly
                if len(user_messages) == 1 and user_messages[0]["role"] == "user":
                    prompt = user_messages[0]["content"]
                    
                    response = self.client.messages.create(
                        model=self.model_name,
                        # max_tokens=4000,
                        # temperature=temperature,
                        system=system_message,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                else:
                    # Convert the conversation history to Anthropic's format
                    anthropic_messages = []
                    for msg in user_messages:
                        anthropic_messages.append({
                            "role": "user" if msg["role"] == "user" else "assistant",
                            "content": msg["content"]
                        })
                    
                    response = self.client.messages.create(
                        model=self.model_name,
                        # max_tokens=4000,
                        # temperature=temperature,
                        system=system_message,
                        messages=anthropic_messages
                    )
                
                result = {
                    "model_id": self.model_id,
                    "content": response.content[0].text,
                    "raw_response": response
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Claude API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All Claude API attempts failed")
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