# json_utils.py
import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple, Union # Added Union

logger = logging.getLogger(__name__)

# --- Constants for JSON parsing ---
JSON_OBJECT_START = '{'
JSON_OBJECT_END = '}'
JSON_ARRAY_START = '['
JSON_ARRAY_END = ']'
JSON_STRING_DELIMITER = '"'
JSON_ESCAPE_CHAR = '\\'


def _is_inside_string(text: str, index: int) -> bool:
    """
    Checks if a character at a given index in a string is part of a JSON string.
    Considers escaped quotes. This is a simplified helper; assumes text up to index is somewhat valid.
    """
    if index == 0:
        return False # Cannot be inside a string if it's the first character before a quote

    in_str = False
    i = 0
    while i < index:
        char = text[i]
        if char == JSON_STRING_DELIMITER:
            if i > 0 and text[i-1] == JSON_ESCAPE_CHAR:
                # Count preceding escape characters to determine if this quote is escaped
                num_escapes = 0
                k = i - 1
                while k >= 0 and text[k] == JSON_ESCAPE_CHAR:
                    num_escapes += 1
                    k -= 1
                if num_escapes % 2 == 1: # Odd number of escapes means this quote is escaped (part of string content)
                    pass
                else: # Even number of escapes (or zero) means this quote is a delimiter
                    in_str = not in_str
            else: # No preceding escape character
                in_str = not in_str
        i += 1
    return in_str


def _find_balanced_structure_indices(text: str, start_char_index: int) -> Optional[Tuple[int, int]]:
    """
    Finds the start and end indices of a balanced JSON structure (object or array)
    starting from `start_char_index`.
    Returns (start_index, end_index_inclusive) or None if not balanced or not found.
    """
    if start_char_index >= len(text):
        return None

    open_char = text[start_char_index]
    if open_char == JSON_OBJECT_START:
        close_char = JSON_OBJECT_END
    elif open_char == JSON_ARRAY_START:
        close_char = JSON_ARRAY_END
    else:
        return None # Not starting with a known JSON structure char

    balance = 0
    
    # More robust string tracking for this specific balanced search
    in_current_search_string = False # Tracks if we are inside a string *during this scan*

    for i in range(start_char_index, len(text)):
        char = text[i]

        if char == JSON_STRING_DELIMITER:
            # Check for escaped quote: if previous char is an escape char,
            # and that escape char itself is not escaped.
            is_escaped_quote = False
            if i > start_char_index and text[i-1] == JSON_ESCAPE_CHAR:
                num_escapes = 0
                k = i - 1
                while k >= start_char_index and text[k] == JSON_ESCAPE_CHAR:
                    num_escapes += 1
                    k -= 1
                if num_escapes % 2 == 1:
                    is_escaped_quote = True
            
            if not is_escaped_quote:
                in_current_search_string = not in_current_search_string
        
        if not in_current_search_string:
            if char == open_char:
                balance += 1
            elif char == close_char:
                balance -= 1
            
            if balance == 0: # Found a balanced structure
                return start_char_index, i
    
    return None # Unbalanced (likely truncated or malformed within the structure)


def extract_json_from_markdown(response_text: str) -> Optional[str]:
    """
    Extracts JSON content from a markdown code block.
    Tries ```json ... ``` first, then generic ``` ... ```.
    """
    if not response_text:
        return None
    
    # Try specific ```json ... ```
    json_specific_match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text, re.IGNORECASE)
    if json_specific_match:
        content = json_specific_match.group(1).strip()
        if content:
            logger.debug("Extracted JSON from '```json' markdown code block.")
            return content

    # Try generic ``` ... ``` and check if content looks like JSON
    json_generic_match = re.search(r"```\s*([\s\S]+?)\s*```", response_text, re.IGNORECASE)
    if json_generic_match:
        content = json_generic_match.group(1).strip()
        if content and \
           ((content.startswith(JSON_OBJECT_START) and content.endswith(JSON_OBJECT_END)) or \
            (content.startswith(JSON_ARRAY_START) and content.endswith(JSON_ARRAY_END)) or \
            (content.startswith(JSON_OBJECT_START) or content.startswith(JSON_ARRAY_START))): # Allow for truncated ends
            logger.debug("Extracted JSON-like content from generic '```' markdown code block.")
            return content
            
    logger.debug("No markdown code block with JSON content found.")
    return None


def extract_and_parse_object_robust(text: str) -> Optional[Any]:
    """
    Finds and parses the largest, first encountered, well-formed JSON object or array.
    """
    if not text or not text.strip():
        return None

    best_parsed_obj = None
    max_len = 0
    
    i = 0
    while i < len(text):
        if text[i] == JSON_OBJECT_START or text[i] == JSON_ARRAY_START:
            indices = _find_balanced_structure_indices(text, i)
            if indices:
                start_idx, end_idx = indices
                substring = text[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(substring)
                    current_len = len(substring)
                    # We want the first *largest* complete object. If multiple top-level, this gets tricky.
                    # This logic finds the largest among potentially overlapping or sequential balanced structures.
                    if current_len > max_len:
                        best_parsed_obj = parsed
                        max_len = current_len
                    # To get the *first complete* object and stop:
                    # logger.debug(f"extract_and_parse_object_robust: Parsed from index {start_idx} to {end_idx}.")
                    # return parsed 
                except json.JSONDecodeError:
                    logger.debug(f"extract_and_parse_object_robust: Balanced segment '{substring[:50]}...' not parsable.")
                i = end_idx + 1 # Move past this balanced segment
                continue
        i += 1
    
    if best_parsed_obj:
        logger.debug(f"extract_and_parse_object_robust: Returning largest parsed structure (len: {max_len}).")
    else:
        logger.debug("extract_and_parse_object_robust: No parsable balanced structure found.")
    return best_parsed_obj


def clean_control_chars(text: str) -> str:
    """Removes ASCII control characters from text."""
    if not text: return ""
    return re.sub(r'[\x00-\x1F\x7F]', '', text)


def parse_with_basic_cleaning(text: str) -> Optional[Any]:
    """Basic cleaning (control chars) and parsing."""
    if not text or not text.strip(): return None
    cleaned = clean_control_chars(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug(f"parse_with_basic_cleaning failed: {e}")
        return None


def parse_with_common_fixes(text: str) -> Optional[Any]:
    """Apply common syntax fixes and try to parse."""
    if not text or not text.strip(): return None
    fixed = text

    # 1. Normalize line endings
    fixed = fixed.replace('\r\n', '\n')

    # 2. Fix trailing commas before closing brace/bracket
    fixed = re.sub(r',\s*(\}|\])', r'\1', fixed)

    # 3. Add missing commas between elements on new lines (heuristic)
    #    - Between " and " (e.g. "value"\n"key")
    fixed = re.sub(r'(")\s*\n\s*(")', r'\1,\n\2', fixed)
    #    - Between } and " (e.g. }\n"key")
    fixed = re.sub(r'(})\s*\n\s*(")', r'\1,\n\2', fixed)
    #    - Between ] and " (e.g. ]\n"key")
    fixed = re.sub(r'(])\s*\n\s*(")', r'\1,\n\2', fixed)
    #    - Between } and { (e.g. }\n{)
    fixed = re.sub(r'(})\s*\n\s*(\{)', r'\1,\n\2', fixed)
    #    - Between ] and { (e.g. ]\n{)
    fixed = re.sub(r'(])\s*\n\s*(\{)', r'\1,\n\2', fixed)
    #    - Between a primitive/number and a new key " (e.g. true\n"key" or 123\n"key")
    fixed = re.sub(r'(true|false|null|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\n\s*(")', r'\1,\n\2', fixed)


    # 4. Attempt to fix unquoted keys (LLM mistake) - carefully
    #    Looks for `(whitespace or { or ,)` then `word_chars` then `whitespace and :`
    #    This is risky and could misinterpret valid string values. Best effort.
    #    Only apply if it looks like a line starting with an unquoted key.
    fixed_lines = []
    for line in fixed.split('\n'):
        # Matches lines like:  key  : value  or key: value (Python identifiers for keys)
        # Does not match if already quoted "key": value
        # Or if it's part of a larger string value e.g. "text with key: value"
        if not line.strip().startswith('"') and re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', line):
             fixed_lines.append(re.sub(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', line, 1))
        else:
            fixed_lines.append(line)
    fixed = '\n'.join(fixed_lines)

    # 5. Balance braces and brackets (append missing closers)
    fixed = balance_closing_brackets_and_braces(fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        logger.debug(f"parse_with_common_fixes failed: {e}")
        return None


def balance_closing_brackets_and_braces(json_str: str) -> str:
    """
    Appends missing closing braces {} and brackets [] to balance the string.
    Does not remove extra closing characters or fix internal mismatches.
    """
    if not json_str: return ""
    logger.debug("Attempting 'balance_closing_brackets_and_braces'")
    
    open_chars_stack = []
    in_str_scan = False # Tracks if scanner is inside a string
    
    # Iterate through the string, being mindful of JSON strings
    i = 0
    temp_str_list = list(json_str) # Working with list for potential future modifications

    while i < len(temp_str_list):
        char = temp_str_list[i]
        
        if char == JSON_STRING_DELIMITER:
            is_escaped_quote = False
            if i > 0 and temp_str_list[i-1] == JSON_ESCAPE_CHAR:
                num_escapes = 0
                k_esc = i - 1
                while k_esc >= 0 and temp_str_list[k_esc] == JSON_ESCAPE_CHAR:
                    num_escapes += 1
                    k_esc -= 1
                if num_escapes % 2 == 1:
                    is_escaped_quote = True
            if not is_escaped_quote:
                in_str_scan = not in_str_scan
        
        elif not in_str_scan: # Only process structural chars if not in a string
            if char == JSON_OBJECT_START or char == JSON_ARRAY_START:
                open_chars_stack.append(char)
            elif char == JSON_OBJECT_END:
                if open_chars_stack and open_chars_stack[-1] == JSON_OBJECT_START:
                    open_chars_stack.pop()
                # else: Mismatch or extra '}'. This function only appends.
            elif char == JSON_ARRAY_END:
                if open_chars_stack and open_chars_stack[-1] == JSON_ARRAY_START:
                    open_chars_stack.pop()
                # else: Mismatch or extra ']'. This function only appends.
        i += 1

    # Append missing closing characters
    balanced_str = "".join(temp_str_list) # Use the original string content
    for open_char in reversed(open_chars_stack):
        if open_char == JSON_OBJECT_START:
            balanced_str += JSON_OBJECT_END
        elif open_char == JSON_ARRAY_START:
            balanced_str += JSON_ARRAY_END
    
    if len(balanced_str) != len(json_str):
        logger.debug(f"Balanced closing brackets/braces. Original len: {len(json_str)}, New len: {len(balanced_str)}")
    return balanced_str


def extract_json_from_response(response_text: str,
                               expected_top_level_keys: Optional[List[str]] = None
                               ) -> Optional[Dict[Any, Any]]:
    """
    Robustly extracts a JSON dictionary from LLM response text.
    Handles markdown, malformations, and tries to return the intended top-level object.
    """
    if not response_text or not response_text.strip():
        logger.warning("Empty or whitespace-only response_text provided.")
        return None

    logger.debug(f"Extracting JSON from response (length: {len(response_text)}). Expected top keys: {expected_top_level_keys}")
    
    # Text from markdown stripping (if any) or original text
    text_after_markdown = extract_json_from_markdown(response_text)
    
    # Determine the primary text to work on.
    # If markdown extraction yielded something, that's preferred. Otherwise, use original.
    primary_text_to_parse = text_after_markdown if text_after_markdown else response_text.strip()
    if not primary_text_to_parse: # Should not happen if response_text was not empty
        logger.error("Primary text to parse became empty, which is unexpected.")
        return None

    # --- Strategy Order ---
    # 1. Direct parse of the primary text (markdown-stripped or original)
    # 2. Parse with basic cleaning (control chars)
    # 3. Parse with common fixes (commas, trailing commas, balancing)
    # 4. Extract largest/first robustly balanced object/array
    # 5. Specific reconstructors (if certain patterns are known for failure modes)

    parsing_attempts_log = []

    # Attempt 1: Direct Parse
    try:
        parsed_json = json.loads(primary_text_to_parse)
        parsing_attempts_log.append(f"Direct parse of primary text (len {len(primary_text_to_parse)}): SUCCESS type {type(parsed_json)}")
        validated = validate_and_normalize_parsed_json(parsed_json, expected_top_level_keys)
        if validated: return validated
    except json.JSONDecodeError as e:
        parsing_attempts_log.append(f"Direct parse of primary text: FAILED ({e})")

    # Attempt 2: Parse with basic cleaning
    cleaned_text = clean_control_chars(primary_text_to_parse)
    if cleaned_text != primary_text_to_parse: # Only try if cleaning did something
        try:
            parsed_json = json.loads(cleaned_text)
            parsing_attempts_log.append(f"Parse with basic cleaning: SUCCESS type {type(parsed_json)}")
            validated = validate_and_normalize_parsed_json(parsed_json, expected_top_level_keys)
            if validated: return validated
        except json.JSONDecodeError as e:
            parsing_attempts_log.append(f"Parse with basic cleaning: FAILED ({e})")
    else:
        parsing_attempts_log.append("Parse with basic cleaning: SKIPPED (no change after cleaning)")


    # Attempt 3: Parse with common fixes
    # This function is more aggressive and includes balancing.
    # We apply it to the `primary_text_to_parse` as it's the most likely candidate.
    parsed_json = parse_with_common_fixes(primary_text_to_parse)
    if parsed_json is not None:
        parsing_attempts_log.append(f"Parse with common fixes: SUCCESS type {type(parsed_json)}")
        validated = validate_and_normalize_parsed_json(parsed_json, expected_top_level_keys)
        if validated: return validated
    else:
        parsing_attempts_log.append("Parse with common fixes: FAILED (returned None)")

    # Attempt 4: Extract largest/first robustly balanced object/array
    # This is good for truncated JSON or JSON embedded in other text.
    parsed_json = extract_and_parse_object_robust(primary_text_to_parse)
    if parsed_json is not None:
        parsing_attempts_log.append(f"Extract and parse object robust: SUCCESS type {type(parsed_json)}")
        validated = validate_and_normalize_parsed_json(parsed_json, expected_top_level_keys)
        if validated: return validated
    else:
        parsing_attempts_log.append("Extract and parse object robust: FAILED (returned None)")


    # Attempt 5: Specific reconstructors (if text contains known keywords)
    # These are very targeted.
    reconstruct_texts_to_try = [primary_text_to_parse]
    if text_after_markdown and text_after_markdown.strip().lower() != response_text.strip().lower():
        # If markdown stripping produced something different than original and was non-empty,
        # also try reconstructors on the original in case keywords were in surrounding text.
        reconstruct_texts_to_try.append(response_text.strip())

    for i_recon, text_for_recon in enumerate(reconstruct_texts_to_try):
        recon_variant_name = "primary_text" if i_recon == 0 else "original_text"
        logger.debug(f"--- Applying RECONSTRUCTION strategies to {recon_variant_name} ---")

        if '"requirement_completeness"' in text_for_recon:
            parsed_json = reconstruct_requirement_completeness(text_for_recon)
            if parsed_json:
                parsing_attempts_log.append(f"Reconstruct requirement_completeness on {recon_variant_name}: SUCCESS")
                validated = validate_and_normalize_parsed_json(parsed_json, ["requirement_completeness"])
                if validated: return validated
        
        if '"requirement_issues"' in text_for_recon: # Check for both, often together
            parsed_json = reconstruct_requirement_issues_and_domain_issues(text_for_recon) # Combined logic for this pair
            if parsed_json:
                parsing_attempts_log.append(f"Reconstruct req_issues & domain_issues on {recon_variant_name}: SUCCESS")
                # Expected keys depend on what was actually found by reconstructor
                current_expected = [k for k in ["requirement_issues", "domain_model_issues"] if k in parsed_json]
                validated = validate_and_normalize_parsed_json(parsed_json, current_expected if current_expected else None)
                if validated: return validated
        
        # Check for domain model structure specifically
        if '"classes"' in text_for_recon and '"relationships"' in text_for_recon:
            parsed_json = reconstruct_domain_model_structure(text_for_recon)
            if parsed_json:
                parsing_attempts_log.append(f"Reconstruct domain_model_structure on {recon_variant_name}: SUCCESS")
                validated = validate_and_normalize_parsed_json(parsed_json, ["classes", "relationships"])
                if validated: return validated

    logger.debug(f"JSON extraction attempts summary:\n" + "\n".join(parsing_attempts_log))
    logger.error("All JSON extraction strategies failed to produce a valid dictionary "
                 "or one meeting expected top-level key criteria.")
    return None


def validate_and_normalize_parsed_json(parsed_json: Any, 
                                       expected_top_keys: Optional[List[str]]=None
                                       ) -> Optional[Dict[Any, Any]]:
    """
    Validates if parsed_json is a dict and (optionally) contains expected_top_keys.
    If parsed_json is a list containing a single suitable dict, that dict is returned.
    """
    logger.debug(f"Validating parsed JSON. Expected top keys: {expected_top_keys}. Type: {type(parsed_json)}")
    
    candidate_dict = None

    if isinstance(parsed_json, dict):
        candidate_dict = parsed_json
    elif isinstance(parsed_json, list):
        logger.debug("Parsed JSON is a list. Checking for suitable dict within.")
        # Prioritize dicts that match expected_top_keys if provided
        # Or, if only one dict in the list, that's a good candidate.
        dicts_in_list = [item for item in parsed_json if isinstance(item, dict)]
        if not dicts_in_list:
            logger.warning("Parsed JSON is a list with no dictionaries.")
            return None
        
        if expected_top_keys:
            for item_dict in dicts_in_list:
                if all(key in item_dict for key in expected_top_keys):
                    logger.info("Found dict in list matching all expected top keys.")
                    candidate_dict = item_dict
                    break
            if not candidate_dict: # No dict matched all keys
                 logger.warning(f"No dict in list matched all expected keys: {expected_top_keys}. Found dict keys: {[list(d.keys()) for d in dicts_in_list]}")
                 # Fallback: if only one dict, consider it, even if keys don't fully match
                 if len(dicts_in_list) == 1:
                     logger.info("Choosing the single dictionary found in the list as candidate despite key mismatch.")
                     candidate_dict = dicts_in_list[0]
                 else: # Multiple dicts, none fully matching
                     return None
        elif len(dicts_in_list) == 1: # No expected keys, list has one dict
            logger.info("Parsed JSON is a list with a single dictionary. Using that dictionary.")
            candidate_dict = dicts_in_list[0]
        else: # No expected keys, list has multiple dicts. Ambiguous.
            logger.warning(f"Parsed JSON is a list with multiple dictionaries ({len(dicts_in_list)}) and no expected_top_keys to disambiguate. Cannot select one.")
            return None # Or you could return the first, or wrap the list like {"data": parsed_json}

    if not isinstance(candidate_dict, dict):
        logger.warning(f"After processing, candidate is not a dictionary (type: {type(candidate_dict)}). Original parsed type was {type(parsed_json)}")
        return None

    # Now, check the chosen candidate_dict against expected_top_keys
    if expected_top_keys:
        if all(key in candidate_dict for key in expected_top_keys):
            logger.debug(f"Validation successful: Candidate dict has all expected top-level keys: {expected_top_keys}.")
            return candidate_dict
        else:
            missing_keys = [k for k in expected_top_keys if k not in candidate_dict]
            logger.warning(f"Validation failed: Candidate dict (keys: {list(candidate_dict.keys())}) "
                           f"is missing expected top-level keys: {missing_keys}.")
            return None
    else: # No expected keys, any dict is fine
        logger.debug("Validation successful: Candidate is a dict (no expected top-level keys to check).")
        return candidate_dict


# --- Domain Model Specific Validation ---
def validate_domain_model(domain_model_data: Optional[Dict[Any, Any]]) -> Dict[str, Any]:
    """
    Validates and normalizes a domain model dictionary.
    Ensures 'classes', 'relationships', 'plantuml' exist and have correct types.
    Adds defaults for missing fields.
    """
    required_top_level_keys = ["classes", "relationships", "plantuml"]
    default_empty_model = {
        "classes": [],
        "relationships": [],
        "plantuml": "@startuml\n@enduml"
    }

    if not domain_model_data or not isinstance(domain_model_data, dict):
        logger.warning("validate_domain_model: Input is None, empty, or not a dict. Returning default empty model.")
        return default_empty_model.copy()

    # Heuristic: If it has "name" AND "attributes" AND "methods" at the root,
    # but NOT "classes" or "relationships", it's highly likely a single class object from a malformed full model.
    is_likely_single_class = (
        all(k in domain_model_data for k in ["name", "attributes", "methods"]) and
        not any(k in domain_model_data for k in required_top_level_keys if k != "plantuml") # plantuml might exist solo
    )
    if is_likely_single_class:
        logger.error(f"validate_domain_model: Input appears to be a single class object "
                     f"(keys: {list(domain_model_data.keys())}) rather than a root domain model. "
                     "This might indicate an upstream parsing error where only a fragment was extracted. "
                     "Returning default empty model.")
        return default_empty_model.copy()

    # Work on a copy to avoid modifying the original dict if it's passed around elsewhere.
    validated_model = domain_model_data.copy()

    for key in required_top_level_keys:
        if key not in validated_model:
            logger.warning(f"validate_domain_model: Root domain model missing '{key}'. Adding default.")
            validated_model[key] = default_empty_model[key]
    
    if not isinstance(validated_model.get("classes"), list):
        logger.warning(f"validate_domain_model: 'classes' is type {type(validated_model.get('classes'))}, not list. Resetting.")
        validated_model["classes"] = []
    if not isinstance(validated_model.get("relationships"), list):
        logger.warning(f"validate_domain_model: 'relationships' is type {type(validated_model.get('relationships'))}, not list. Resetting.")
        validated_model["relationships"] = []
    if not isinstance(validated_model.get("plantuml"), str):
        logger.warning(f"validate_domain_model: 'plantuml' is type {type(validated_model.get('plantuml'))}, not str. Resetting.")
        validated_model["plantuml"] = default_empty_model["plantuml"]
    
    # Validate structure of individual classes
    valid_classes = []
    for cls_obj in validated_model.get("classes", []):
        if not isinstance(cls_obj, dict):
            logger.warning(f"validate_domain_model: Item in 'classes' list is not a dict: {type(cls_obj)}. Skipping.")
            continue
        
        cls_obj.setdefault("name", "UnknownClassValidated")
        if not isinstance(cls_obj["name"], str) or not cls_obj["name"].strip(): cls_obj["name"] = "UnnamedClassValidated"
        cls_obj.setdefault("attributes", [])
        if not isinstance(cls_obj["attributes"], list): cls_obj["attributes"] = []
        cls_obj.setdefault("methods", [])
        if not isinstance(cls_obj["methods"], list): cls_obj["methods"] = []
        cls_obj.setdefault("description", "")
        if not isinstance(cls_obj["description"], str): cls_obj["description"] = ""
        valid_classes.append(cls_obj)
    validated_model["classes"] = valid_classes

    # Validate structure of individual relationships
    valid_relationships = []
    for rel_obj in validated_model.get("relationships", []):
        if not isinstance(rel_obj, dict):
            logger.warning(f"validate_domain_model: Item in 'relationships' list is not a dict: {type(rel_obj)}. Skipping.")
            continue
        for field in ["source", "target", "type"]: # Add other required rel fields if any
            rel_obj.setdefault(field, "UnknownValidated")
            if not isinstance(rel_obj[field], str) or not rel_obj[field].strip(): rel_obj[field] = "UnnamedRelPartValidated"
        valid_relationships.append(rel_obj)
    validated_model["relationships"] = valid_relationships

    return validated_model


# --- Regex-based Reconstruction Functions (Targeted for specific malformed patterns) ---
# These are powerful but can be brittle. Test them extensively with example malformed inputs.
# The general idea is to capture known fields and reconstruct the JSON object.
# Helper for unescaping JSON string content captured by regex:
def _safe_json_loads_str_val(val: str) -> str:
    try:
        return json.loads(f'"{val}"') # Wrap in quotes to parse as a JSON string
    except json.JSONDecodeError:
        logger.warning(f"Failed to json.loads string value: {val[:50]}... Returning as is after basic unescaping.")
        return val.replace('\\"', '"').replace("\\\\", "\\") # Basic unescaping if json.loads fails

def reconstruct_requirement_completeness(text: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    logger.debug("Attempting 'reconstruct_requirement_completeness'")
    items = []
    # Pattern designed to be somewhat forgiving of whitespace and resilient to complex text.
    # Uses non-capturing groups (?:...) for alternatives or optional parts.
    # (?P<name>...) creates named capture groups.
    # (?:\\.|[^"\\])* handles escaped quotes and other characters within string fields.
    pattern = re.compile(
        r'\{\s*'
        r'"requirement_id"\s*:\s*"(?P<req_id>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"requirement_text"\s*:\s*"(?P<req_text>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"completeness_score"\s*:\s*(?P<score>\d+(?:\.\d+)?)\s*,\s*' # Allows int or float for score
        r'"missing_elements"\s*:\s*\[(?P<missing_elems>(?:[^\[\]"]*(?:(?:"(?:\\.|[^"\\])*")[^\[\]"]*)*)*?)\]\s*,\s*' # More robust missing_elements array
        r'"suggested_improvement"\s*:\s*"(?P<suggestion>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"rationale"\s*:\s*"(?P<rationale>(?:\\.|[^"\\])*)"\s*'
        r'\}', re.DOTALL
    )
    for match in pattern.finditer(text):
        data = match.groupdict()
        missing_elements_list = []
        if data['missing_elems'].strip():
            # Extract elements from the missing_elems string
            elem_matches = re.findall(r'"((?:\\.|[^"\\])*)"', data['missing_elems'])
            missing_elements_list = [_safe_json_loads_str_val(elem) for elem in elem_matches]
        
        try:
            items.append({
                "requirement_id": _safe_json_loads_str_val(data['req_id']),
                "requirement_text": _safe_json_loads_str_val(data['req_text']),
                "completeness_score": int(float(data['score'])), # Ensure score is int after parsing
                "missing_elements": missing_elements_list,
                "suggested_improvement": _safe_json_loads_str_val(data['suggestion']),
                "rationale": _safe_json_loads_str_val(data['rationale'])
            })
        except Exception as e:
            logger.error(f"Error reconstructing requirement_completeness item: {e}. Data: {data}")
            continue # Skip this item

    return {"requirement_completeness": items} if items else None


def reconstruct_requirement_issues_and_domain_issues(text: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Attempts to reconstruct both "requirement_issues" and "domain_model_issues"
    if their respective patterns are found in the text.
    Returns a dict with one or both keys if successful.
    """
    logger.debug("Attempting 'reconstruct_requirement_issues_and_domain_issues'")
    reconstructed_data = {}

    # --- Reconstruct Requirement Issues ---
    req_issues_items = []
    # Pattern for a single requirement issue block
    req_block_pattern = re.compile(
        r'\{\s*'
        r'"requirement_id"\s*:\s*"(?P<req_id>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"requirement_text"\s*:\s*"(?P<req_text>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"issues"\s*:\s*\[(?P<issues_array_content>(?:[^\[\]"]*(?:(?:"(?:\\.|[^"\\])*")[^\[\]"]*)*)*?)\]\s*' # Content of issues array
        r'\}', re.DOTALL
    )
    # Pattern for a single issue object within the issues_array_content
    issue_object_pattern = re.compile(
        r'\{\s*'
        r'"issue_type"\s*:\s*"(?P<issue_type>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"severity"\s*:\s*"(?P<severity>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"description"\s*:\s*"(?P<description>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"suggested_fix"\s*:\s*"(?P<suggested_fix>(?:\\.|[^"\\])*)"\s*'
        r'(?:,\s*"affected_model_elements"\s*:\s*\[(?P<affected_elements_array>(?:[^\[\]"]*(?:(?:"(?:\\.|[^"\\])*")[^\[\]"]*)*)*?)\])?\s*' # Optional affected_model_elements
        r'\}', re.DOTALL
    )

    for req_match in req_block_pattern.finditer(text):
        req_data = req_match.groupdict()
        current_issues_list = []
        for issue_match in issue_object_pattern.finditer(req_data['issues_array_content']):
            issue_data = issue_match.groupdict()
            affected_elements = []
            if issue_data.get('affected_elements_array') and issue_data['affected_elements_array'].strip():
                elem_matches = re.findall(r'"((?:\\.|[^"\\])*)"', issue_data['affected_elements_array'])
                affected_elements = [_safe_json_loads_str_val(elem) for elem in elem_matches]
            
            try:
                current_issues_list.append({
                    "issue_type": _safe_json_loads_str_val(issue_data['issue_type']),
                    "severity": _safe_json_loads_str_val(issue_data['severity']),
                    "description": _safe_json_loads_str_val(issue_data['description']),
                    "suggested_fix": _safe_json_loads_str_val(issue_data['suggested_fix']),
                    "affected_model_elements": affected_elements
                })
            except Exception as e:
                logger.error(f"Error reconstructing req issue item: {e}. Data: {issue_data}")
                continue
        
        if current_issues_list: # Only add if issues were found for this req_id
            try:
                req_issues_items.append({
                    "requirement_id": _safe_json_loads_str_val(req_data['req_id']),
                    "requirement_text": _safe_json_loads_str_val(req_data['req_text']),
                    "issues": current_issues_list
                })
            except Exception as e:
                 logger.error(f"Error reconstructing req issue block: {e}. Data: {req_data}")
                 continue
    
    if req_issues_items:
        reconstructed_data["requirement_issues"] = req_issues_items

    # --- Reconstruct Domain Model Issues ---
    domain_issues_items = []
    domain_issue_pattern = re.compile(
        r'\{\s*'
        r'"element_type"\s*:\s*"(?P<elem_type>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"element_name"\s*:\s*"(?P<elem_name>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"issue_type"\s*:\s*"(?P<issue_type_domain>(?:\\.|[^"\\])*)"\s*,\s*' # Renamed to avoid clash
        r'"severity"\s*:\s*"(?P<severity_domain>(?:\\.|[^"\\])*)"\s*,\s*' # Renamed
        r'"description"\s*:\s*"(?P<description_domain>(?:\\.|[^"\\])*)"\s*,\s*' # Renamed
        r'"suggested_fix"\s*:\s*"(?P<suggested_fix_domain>(?:\\.|[^"\\])*)"\s*,\s*' # Renamed
        r'"affected_requirements"\s*:\s*\[(?P<affected_reqs_array>(?:[^\[\]"]*(?:(?:"(?:\\.|[^"\\])*")[^\[\]"]*)*)*?)\]\s*'
        r'\}', re.DOTALL
    )
    for domain_match in domain_issue_pattern.finditer(text):
        d_issue_data = domain_match.groupdict()
        affected_reqs = []
        if d_issue_data['affected_reqs_array'].strip():
            req_name_matches = re.findall(r'"((?:\\.|[^"\\])*)"', d_issue_data['affected_reqs_array'])
            affected_reqs = [_safe_json_loads_str_val(req_name) for req_name in req_name_matches]
        
        try:
            domain_issues_items.append({
                "element_type": _safe_json_loads_str_val(d_issue_data['elem_type']),
                "element_name": _safe_json_loads_str_val(d_issue_data['elem_name']),
                "issue_type": _safe_json_loads_str_val(d_issue_data['issue_type_domain']),
                "severity": _safe_json_loads_str_val(d_issue_data['severity_domain']),
                "description": _safe_json_loads_str_val(d_issue_data['description_domain']),
                "suggested_fix": _safe_json_loads_str_val(d_issue_data['suggested_fix_domain']),
                "affected_requirements": affected_reqs
            })
        except Exception as e:
            logger.error(f"Error reconstructing domain issue item: {e}. Data: {d_issue_data}")
            continue

    if domain_issues_items:
        reconstructed_data["domain_model_issues"] = domain_issues_items
        
    return reconstructed_data if reconstructed_data else None


def reconstruct_domain_model_structure(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to reconstruct a domain model structure {"classes": [...], "relationships": [...], "plantuml": "..."}
    This is complex due to nested arrays and objects.
    """
    logger.debug("Attempting 'reconstruct_domain_model_structure'")
    
    # Heuristic: Try to find the overall JSON object that seems to contain these keys.
    # This is less about regex-parsing each individual class/relationship from flat text,
    # and more about finding a somewhat intact larger JSON object.
    
    # Find a block that looks like it starts a domain model
    # Prefer a block that explicitly contains "classes", "relationships", and "plantuml"
    # This regex is very broad and aims to find the largest plausible JSON object.
    match = re.search(r'\{\s*("classes"\s*:\s*\[.*?],\s*"relationships"\s*:\s*\[.*?],\s*"plantuml"\s*:\s*".*?")\s*\}', text, re.DOTALL | re.IGNORECASE)
    if not match: # Fallback if specific structure isn't perfectly matched
         match = re.search(r'(\{[\s\S]*\})', text) # Grab any block starting with { and ending with }

    if match:
        potential_json_str = match.group(1)
        if not potential_json_str.startswith('{'): potential_json_str = '{' + potential_json_str
        if not potential_json_str.endswith('}'): potential_json_str = potential_json_str + '}'
        
        # Try parsing this candidate with fixes
        parsed_model = parse_with_common_fixes(potential_json_str)
        if isinstance(parsed_model, dict) and \
           "classes" in parsed_model and \
           "relationships" in parsed_model and \
           "plantuml" in parsed_model:
            logger.info("Reconstructed domain model structure using broad match and fixes.")
            # Further validation can be done by validate_domain_model downstream
            return parsed_model
        else: # Try robust extraction on the candidate if common fixes failed.
            parsed_model = extract_and_parse_object_robust(potential_json_str)
            if isinstance(parsed_model, dict) and \
               "classes" in parsed_model and \
               "relationships" in parsed_model and \
               "plantuml" in parsed_model:
                logger.info("Reconstructed domain model structure using robust extraction on candidate.")
                return parsed_model

    logger.warning("Failed to reconstruct a complete domain model structure with primary keys.")
    return None


# --- Other Utility Functions (previously parse_by_reconstructing, etc.) ---
# The generic reconstruction should be handled by extract_and_parse_object_robust or parse_with_common_fixes.
# The `parse_with_line_by_line_fix` is highly heuristic and often less reliable than `parse_with_common_fixes`.
# `try_progressive_json_repair` and `fix_json_comprehensively` are aggressive and can be folded into the main strategies if needed,
# or kept separate if their specific brand of repair is desired as a last resort.
# For this version, focusing on the core strategies in extract_json_from_response.

def create_default_analysis() -> Dict[str, List]:
    """Creates a default structure for analysis results."""
    return {
        "requirement_issues": [],
        "missing_requirements": [], # Assuming this key might be used
        "domain_model_issues": [],
        "requirement_completeness": []
    }

def safe_json_parse(json_string: str, 
                      default_return: Any = None, 
                      expected_top_level_keys: Optional[List[str]] = None
                      ) -> Any:
    """
    Wrapper for extract_json_from_response to provide a simple safe parse with a default.
    """
    try:
        parsed = extract_json_from_response(json_string, expected_top_level_keys)
        return parsed if parsed is not None else default_return
    except Exception as e:
        logger.error(f"Exception in safe_json_parse for string '{json_string[:100]}...': {e}")
        return default_return