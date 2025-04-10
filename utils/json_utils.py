import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_json_from_response(response_text):
    """
    Enhanced version that can handle control characters and other problematic JSON issues
    
    Args:
        response_text (str): The text response that contains JSON
        
    Returns:
        dict: Parsed JSON as a dictionary, or None if parsing failed
    """
    # First, check if the response is wrapped in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)
        logger.debug("Extracted JSON from markdown code block")
    
    # Try to parse the JSON directly
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON decode error: {str(e)}")
        
        # Clean the response - handle control characters
        cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', response_text)
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            logger.warning("JSON decode error after removing control characters")
            
            # Try more aggressively to extract any JSON-like structure
            possible_json = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if possible_json:
                try:
                    extracted_json = possible_json.group(0)
                    
                    # Fix common issues in JSON
                    extracted_json = re.sub(r',\s*}', '}', extracted_json)  # Remove trailing commas
                    extracted_json = re.sub(r',\s*]', ']', extracted_json)  # Remove trailing commas in arrays
                    
                    # Fix unquoted keys
                    extracted_json = re.sub(r'(\w+)(:)', r'"\1"\2', extracted_json)
                    
                    # Try to parse the cleaned JSON
                    result = json.loads(extracted_json)
                    logger.info("Successfully extracted JSON using regex and cleaning")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON-like content: {str(e)}")
            
            try:
                import demjson
                result = demjson.decode(cleaned_text)
                logger.info("Successfully parsed JSON using demjson library")
                return result
            except ImportError:
                logger.warning("demjson library not available for fallback JSON parsing")
            except Exception as e:
                logger.error(f"demjson parsing failed: {str(e)}")
        
        logger.error("Could not extract valid JSON from response")
        return None

def validate_domain_model(domain_model):
    """
    Validate that a domain model has all required fields and structure.
    If fields are missing, add defaults.
    
    Args:
        domain_model (dict): The domain model to validate
        
    Returns:
        dict: A valid domain model with all required fields
    """
    if not domain_model:
        logger.warning("Domain model is None or empty, creating default")
        return {
            "classes": [],
            "relationships": [],
            "plantuml": "@startuml\n@enduml"
        }
    
    # Check if the domain model has the required fields
    required_keys = ["classes", "relationships", "plantuml"]
    missing_keys = [k for k in required_keys if k not in domain_model]
    
    if missing_keys:
        logger.warning(f"Domain model missing required fields: {missing_keys}")
        
        # Add defaults for missing keys
        for key in missing_keys:
            if key in ["classes", "relationships"]:
                domain_model[key] = []
            elif key == "plantuml":
                domain_model[key] = "@startuml\n@enduml"
    
    return domain_model

def create_default_analysis():
    """
    Create a default analysis structure with empty collections
    
    Returns:
        dict: A default analysis structure
    """
    return {
        "requirement_issues": [],
        "missing_requirements": [],
        "domain_model_issues": [],
        "requirement_completeness": []
    }