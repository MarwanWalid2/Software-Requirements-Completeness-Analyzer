import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

def save_problematic_json(response_text: str, error_msg: str, context: str = "") -> str:
    """
    Save problematic JSON responses to temporary files for debugging
    
    Args:
        response_text (str): The raw response text that failed to parse
        error_msg (str): The error message from JSON parsing
        context (str): Additional context about where this came from
        
    Returns:
        str: Path to the saved debug file
    """
    try:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create debug filename
        debug_filename = f"json_debug_{timestamp}.json"
        debug_path = os.path.join(tempfile.gettempdir(), debug_filename)
        
        # Create debug info
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_message": error_msg,
            "response_length": len(response_text),
            "response_preview": response_text[:500] + "..." if len(response_text) > 500 else response_text,
            "response_suffix": "..." + response_text[-500:] if len(response_text) > 500 else "",
            "full_response": response_text
        }
        
        # Save to file
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved problematic JSON debug info to: {debug_path}")
        return debug_path
        
    except Exception as e:
        logger.error(f"Failed to save JSON debug info: {str(e)}")
        return ""

def analyze_json_structure(json_str: str) -> dict:
    """
    Analyze the structure of a potentially malformed JSON string
    
    Args:
        json_str (str): JSON string to analyze
        
    Returns:
        dict: Analysis results
    """
    analysis = {
        "length": len(json_str),
        "lines": len(json_str.split('\n')),
        "brace_balance": 0,
        "bracket_balance": 0,
        "quote_issues": [],
        "common_issues": [],
        "structure_depth": 0
    }
    
    try:
        # Count braces and brackets
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        max_depth = 0
        current_depth = 0
        line_num = 1
        
        for i, char in enumerate(json_str):
            if char == '\n':
                line_num += 1
                
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == '}':
                    brace_count -= 1
                    current_depth -= 1
                elif char == '[':
                    bracket_count += 1
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == ']':
                    bracket_count -= 1
                    current_depth -= 1
        
        analysis["brace_balance"] = brace_count
        analysis["bracket_balance"] = bracket_count
        analysis["structure_depth"] = max_depth
        
        # Check for common issues
        if brace_count != 0:
            analysis["common_issues"].append(f"Unbalanced braces: {brace_count} extra opening braces")
        if bracket_count != 0:
            analysis["common_issues"].append(f"Unbalanced brackets: {bracket_count} extra opening brackets")
        if in_string:
            analysis["common_issues"].append("Unclosed string at end of JSON")
            
        # Look for missing commas
        lines = json_str.split('\n')
        for i, line in enumerate(lines[:-1]):  # Don't check last line
            line = line.strip()
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            
            # Check for missing commas between objects/arrays
            if (line.endswith('}') or line.endswith(']')) and next_line.startswith(('{', '[')):
                analysis["common_issues"].append(f"Possible missing comma after line {i + 1}")
            
            # Check for missing commas after quoted strings
            if line.endswith('"') and next_line.startswith('"') and not line.endswith('",'):
                analysis["common_issues"].append(f"Possible missing comma after quoted string on line {i + 1}")
        
    except Exception as e:
        analysis["analysis_error"] = str(e)
    
    return analysis

def find_json_error_location(json_str: str, error_msg: str) -> Optional[dict]:
    """
    Try to find the location of JSON parsing errors
    
    Args:
        json_str (str): JSON string with error
        error_msg (str): Error message from JSON parser
        
    Returns:
        dict: Information about error location if found
    """
    try:
        # Parse error message to extract line/column info
        import re
        
        # Look for patterns like "line 165 column 5" or "char 8084"
        line_col_match = re.search(r'line (\d+) column (\d+)', error_msg)
        char_match = re.search(r'char (\d+)', error_msg)
        
        result = {"error_context": error_msg}
        
        if char_match:
            char_pos = int(char_match.group(1))
            result["char_position"] = char_pos
            
            # Get context around the error
            start = max(0, char_pos - 100)
            end = min(len(json_str), char_pos + 100)
            result["error_context_text"] = json_str[start:end]
            result["error_char"] = json_str[char_pos] if char_pos < len(json_str) else "EOF"
            
            # Find line number
            lines_before_error = json_str[:char_pos].count('\n')
            result["estimated_line"] = lines_before_error + 1
        
        if line_col_match:
            line_num = int(line_col_match.group(1))
            col_num = int(line_col_match.group(2))
            result["line_number"] = line_num
            result["column_number"] = col_num
            
            # Get the problematic line
            lines = json_str.split('\n')
            if 0 <= line_num - 1 < len(lines):
                problematic_line = lines[line_num - 1]
                result["problematic_line"] = problematic_line
                
                # Get context lines
                start_line = max(0, line_num - 3)
                end_line = min(len(lines), line_num + 2)
                result["context_lines"] = lines[start_line:end_line]
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing JSON error location: {str(e)}")
        return {"analysis_error": str(e)}

def suggest_json_fixes(json_str: str, error_info: dict) -> list:
    """
    Suggest specific fixes for JSON parsing errors
    
    Args:
        json_str (str): Problematic JSON string
        error_info (dict): Error information from find_json_error_location
        
    Returns:
        list: List of suggested fixes
    """
    suggestions = []
    
    try:
        error_msg = error_info.get("error_context", "").lower()
        
        if "expecting ',' delimiter" in error_msg:
            suggestions.append("Add missing comma between JSON elements")
            if "line_number" in error_info:
                line_num = error_info["line_number"]
                suggestions.append(f"Check line {line_num} for missing comma")
        
        elif "expecting ':' delimiter" in error_msg:
            suggestions.append("Add missing colon after object key")
            
        elif "unterminated string" in error_msg:
            suggestions.append("Add missing closing quote for string")
            
        elif "expecting property name" in error_msg:
            suggestions.append("Object key should be quoted string")
            
        # Check structural issues
        analysis = analyze_json_structure(json_str)
        
        if analysis["brace_balance"] != 0:
            if analysis["brace_balance"] > 0:
                suggestions.append(f"Add {analysis['brace_balance']} closing braces '}}' ")
            else:
                suggestions.append(f"Remove {-analysis['brace_balance']} extra closing braces '}}' ")
        
        if analysis["bracket_balance"] != 0:
            if analysis["bracket_balance"] > 0:
                suggestions.append(f"Add {analysis['bracket_balance']} closing brackets ']' ")
            else:
                suggestions.append(f"Remove {-analysis['bracket_balance']} extra closing brackets ']' ")
        
        # Add common fixes
        suggestions.extend([
            "Remove trailing commas before } or ]",
            "Ensure all string values are properly quoted",
            "Check for unescaped quotes within strings",
            "Verify proper nesting of objects and arrays"
        ])
        
    except Exception as e:
        suggestions.append(f"Error analyzing JSON: {str(e)}")
    
    return suggestions