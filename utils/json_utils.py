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
        
        # Try to parse again after basic cleaning
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            logger.warning("JSON decode error after removing control characters")
            
            # Apply more aggressive JSON fixes
            fixed_json = fix_json_syntax(cleaned_text)
            
            try:
                result = json.loads(fixed_json)
                logger.info("Successfully parsed JSON after applying fixes")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error after applying fixes: {str(e)}")
            
            # Try to extract any JSON-like structure
            possible_json = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if possible_json:
                try:
                    extracted_json = possible_json.group(0)
                    
                    # Apply fixes to extracted JSON
                    extracted_json = fix_json_syntax(extracted_json)
                    
                    # Try to parse the cleaned JSON
                    result = json.loads(extracted_json)
                    logger.info("Successfully extracted JSON using regex and cleaning")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON-like content: {str(e)}")
            
            # Try demjson as last resort
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

def fix_json_syntax(json_str):
    """
    Apply various fixes to common JSON syntax errors
    
    Args:
        json_str (str): JSON string with potential syntax errors
        
    Returns:
        str: Fixed JSON string
    """
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing commas between array elements
    # Look for patterns like "}\n{" or "]\n[" without comma
    json_str = re.sub(r'}\s*\n\s*{', '},\n{', json_str)
    json_str = re.sub(r']\s*\n\s*\[', '],\n[', json_str)
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    
    # Fix missing commas between object properties
    # Look for patterns like "value"\n"key": or "value"\n"key"
    json_str = re.sub(r'(true|false|null|"[^"]*"|\d+)\s*\n\s*"', r'\1,\n"', json_str)
    json_str = re.sub(r'(}|])\s*\n\s*"', r'\1,\n"', json_str)
    
    # Fix unquoted keys (simple cases)
    # This regex looks for word characters followed by colon, not already in quotes
    # Be careful not to affect URLs or other valid strings
    json_str = re.sub(r'(?<!")(\b\w+)\s*:', r'"\1":', json_str)
    
    # Fix incomplete strings (missing closing quote)
    # This is tricky and might not always work correctly
    lines = json_str.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        # Count quotes in the line
        quote_count = line.count('"') - line.count('\\"')
        if quote_count % 2 == 1:  # Odd number of quotes
            # Check if line ends with incomplete string
            if re.search(r'"[^"]*$', line) and not line.strip().endswith('",'):
                line = line.rstrip() + '"'
        fixed_lines.append(line)
    json_str = '\n'.join(fixed_lines)
    
    # Fix truncated JSON (ensure it ends with proper closing braces/brackets)
    # Count opening and closing braces/brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Add missing closing braces/brackets
    while close_braces < open_braces:
        json_str += '}'
        close_braces += 1
    
    while close_brackets < open_brackets:
        json_str += ']'
        close_brackets += 1
    
    return json_str

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
    
    # Validate classes structure
    if isinstance(domain_model.get("classes"), list):
        for cls in domain_model["classes"]:
            if not isinstance(cls, dict):
                continue
            # Ensure required class fields exist
            if "name" not in cls:
                cls["name"] = "UnknownClass"
            if "attributes" not in cls:
                cls["attributes"] = []
            if "methods" not in cls:
                cls["methods"] = []
    
    # Validate relationships structure
    if isinstance(domain_model.get("relationships"), list):
        for rel in domain_model["relationships"]:
            if not isinstance(rel, dict):
                continue
            # Ensure required relationship fields exist
            for field in ["source", "target", "type"]:
                if field not in rel:
                    rel[field] = "Unknown"
    
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

def safe_parse_json(json_str, default=None):
    """
    Safely parse JSON with a default fallback
    
    Args:
        json_str (str): JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        result = extract_json_from_response(json_str)
        return result if result is not None else default
    except Exception as e:
        logger.error(f"Error in safe_parse_json: {str(e)}")
        return default
