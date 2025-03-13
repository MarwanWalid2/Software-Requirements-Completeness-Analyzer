import json
import logging
import time
import traceback

from services.deepseek_client import initialize_deepseek_client
from services.plantuml_service import generate_plantuml_image
from utils.json_utils import extract_json_from_response, validate_domain_model, create_default_analysis

logger = logging.getLogger(__name__)

class DomainModelAnalyzer:
    def __init__(self):
        self.client = initialize_deepseek_client()
        self.model_name = "deepseek-chat"
        self.max_retries = 1
        self.retry_delay = 5  # seconds
    
    def create_domain_model(self, requirements):
        """Generate a domain model from requirements using deepseek-reasoner"""
        logger.info("Creating domain model")
        
        # Simplify the prompt to reduce response size
        prompt = """
        You are an expert software architect. Create a concise domain model for these requirements.
        
        Keep your model focused on the key entities and relationships. Be efficient and direct.
        
        FORMAT YOUR RESPONSE AS JSON with this structure:
        {
            "classes": [
                {
                    "name": "ClassName",
                    "attributes": [
                        {"name": "attributeName", "type": "dataType", "description": "description"}
                    ],
                    "methods": [
                        {"name": "methodName", "parameters": [{"name": "paramName", "type": "paramType"}], "returnType": "returnType", "description": "description"}
                    ],
                    "description": "Class responsibility description"
                }
            ],
            "relationships": [
                {
                    "source": "SourceClass",
                    "target": "TargetClass",
                    "type": "association|composition|aggregation|inheritance|realization",
                    "sourceMultiplicity": "1|0..1|0..*|1..*",
                    "targetMultiplicity": "1|0..1|0..*|1..*",
                    "description": "Description of relationship"
                }
            ],
            "plantuml": "PlantUML code that represents this domain model"
        }
        
        Here are the requirements:
        
        """
        
        messages = [{"role": "user", "content": prompt + requirements}]
        logger.debug(f"Sending request to DeepSeek API with {len(messages[0]['content'])} characters")
        
        # Try with the OpenAI SDK, with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling DeepSeek API (attempt {attempt+1}/{self.max_retries})")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                
                logger.info("Received response from DeepSeek API")
                
                # Extract and validate the response content
                domain_model_json = response.choices[0].message.content
                if not domain_model_json:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Content length: {len(domain_model_json)}")
                logger.debug(f"Content sample: {domain_model_json[:200]}...")
                
                # Try to parse and validate the JSON
                try:
                    domain_model = extract_json_from_response(domain_model_json)
                    if not domain_model:
                        raise ValueError("Failed to extract valid JSON")
                    
                    # Validate and fix the domain model
                    domain_model = validate_domain_model(domain_model)
                    
                    logger.info("Domain model successfully created")
                    logger.info(f"Model contains {len(domain_model.get('classes', []))} classes and {len(domain_model.get('relationships', []))} relationships")
                    
                    # OpenAI's model doesn't provide reasoning_content like DeepSeek
                    reasoning_content = "Reasoning information not available with this model"
                    
                    return {
                        "domain_model": domain_model,
                        "reasoning": reasoning_content
                    }
                except Exception as e:
                    logger.warning(f"Error processing domain model: {str(e)}")
                    
                    # Save the raw response for debugging
                    with open(f"log\raw_domain_model_response_{attempt}.txt", "w") as f:
                        f.write(domain_model_json)
                    
                    # If not the last attempt, try again
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying API call after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                    
            except Exception as e:
                logger.error(f"API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If all attempts failed, return a default domain model
        logger.error("All API attempts failed. Returning default domain model")
        return {
            "domain_model": {
                "classes": [],
                "relationships": [],
                "plantuml": "@startuml\n@enduml"
            },
            "error": "All API attempts failed",
            "reasoning": "Error occurred during model generation"
        }
    
    def detect_missing_requirements(self, requirements, domain_model):
        """Specialized function to detect missing requirements based on domain model and natural language"""
        logger.info("Detecting missing requirements")
        
        # Create a default response in case all API calls fail
        default_response = {
            "missing_requirements": []
        }
        
        # Enhanced prompt focused specifically on finding missing requirements
        prompt = """
        You are an expert requirements analyst. Your task is to identify MISSING requirements that should exist 
        based on the domain model and the provided requirements.
        
        Focus on identifying:
        1. Functionality that should exist based on the domain model entities but is not mentioned
        2. Missing operations for entities (create, read, update, delete)
        3. Missing business rules or validations
        4. Missing non-functional requirements (security, performance, etc.)
        5. Missing edge cases or error handling
        
        FORMAT YOUR RESPONSE AS JSON:
        {
            "missing_requirements": [
                {
                    "id": "MR1",
                    "description": "Description of what's missing",
                    "category": "Functional|Business Rule|CRUD Operation|Non-Functional|Error Handling",
                    "severity": "CRITICAL|HIGH|MEDIUM|LOW",
                    "suggested_requirement": "Suggested text for the requirement",
                    "affected_model_elements": ["Class1", "Relationship2"],
                    "rationale": "Why this requirement should exist"
                }
            ]
        }
        
        REQUIREMENTS:
        
        """
        
        try:
            domain_model_text = json.dumps(domain_model, indent=2)
            logger.debug(f"Domain model JSON length for missing req detection: {len(domain_model_text)}")
        except Exception as e:
            logger.error(f"Error serializing domain model: {str(e)}")
            return {"missing_requirements": [], "error": f"Error serializing domain model: {str(e)}"}
        
        full_prompt = prompt + requirements + "\n\nDOMAIN MODEL:\n" + domain_model_text
        messages = [{"role": "user", "content": full_prompt}]
        
        logger.debug(f"Missing requirements detection prompt length: {len(full_prompt)}")
        
        # Call the API with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending missing requirements detection request (attempt {attempt+1}/{self.max_retries})")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                
                logger.info("Received missing requirements response")
                
                # Extract content
                result_json = response.choices[0].message.content
                if not result_json:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Missing requirements content sample: {result_json[:200]}...")
                
                # Save raw response for debugging
                with open(f"log\raw_missing_requirements_{attempt}.txt", "w") as f:
                    f.write(result_json)
                
                # Parse and validate the JSON
                try:
                    result = extract_json_from_response(result_json)
                    if not result:
                        raise ValueError("Failed to extract valid JSON")
                    
                    # Ensure missing_requirements key exists
                    if "missing_requirements" not in result:
                        logger.warning("Missing 'missing_requirements' key in response")
                        result["missing_requirements"] = []
                    
                    logger.info(f"Found {len(result.get('missing_requirements', []))} missing requirements")
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Error processing missing requirements: {str(e)}")
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying missing requirements detection after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                
            except Exception as e:
                logger.error(f"Missing requirements API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying missing requirements detection after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If we've exhausted all retries, return a default response
        logger.error("All missing requirements detection attempts failed, returning default")
        return default_response
    
    def analyze_requirement_completeness(self, requirements, domain_model):
        """Specialized function to analyze individual requirement completeness"""
        logger.info("Analyzing individual requirement completeness")
        
        # Create a default response in case all API calls fail
        default_response = {
            "requirement_completeness": []
        }
        
        # Prompt focused on analyzing the completeness of individual requirements
        prompt = """
        You are an expert requirements analyst. Your task is to analyze the completeness of each individual requirement.
        
        For each requirement, check if it contains all necessary elements:
        1. For functional requirements: actor, action, object, result/outcome
        2. For non-functional requirements: quality attribute, measure, context
        3. For constraints: clear boundary condition, rationale
        
        FORMAT YOUR RESPONSE AS JSON:
        {
            "requirement_completeness": [
                {
                    "requirement_id": "R1",
                    "requirement_text": "text",
                    "completeness_score": 0-100,
                    "missing_elements": ["actor", "outcome", etc.],
                    "suggested_improvement": "Suggested improved text",
                    "rationale": "Why this improvement is needed"
                }
            ]
        }
        
        REQUIREMENTS:
        
        """
        
        try:
            domain_model_text = json.dumps(domain_model, indent=2)
            logger.debug(f"Domain model JSON length for completeness analysis: {len(domain_model_text)}")
        except Exception as e:
            logger.error(f"Error serializing domain model: {str(e)}")
            return {"requirement_completeness": [], "error": f"Error serializing domain model: {str(e)}"}
        
        full_prompt = prompt + requirements + "\n\nDOMAIN MODEL:\n" + domain_model_text
        messages = [{"role": "user", "content": full_prompt}]
        
        logger.debug(f"Requirement completeness analysis prompt length: {len(full_prompt)}")
        
        # Call the API with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending requirement completeness analysis request (attempt {attempt+1}/{self.max_retries})")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                
                logger.info("Received requirement completeness response")
                
                # Extract content
                result_json = response.choices[0].message.content
                if not result_json:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Requirement completeness content sample: {result_json[:200]}...")
                
                # Save raw response for debugging
                with open(f"log\raw_requirement_completeness_{attempt}.txt", "w") as f:
                    f.write(result_json)
                
                # Parse and validate the JSON
                try:
                    result = extract_json_from_response(result_json)
                    if not result:
                        raise ValueError("Failed to extract valid JSON")
                    
                    # Ensure requirement_completeness key exists
                    if "requirement_completeness" not in result:
                        logger.warning("Missing 'requirement_completeness' key in response")
                        result["requirement_completeness"] = []
                    
                    logger.info(f"Analyzed completeness of {len(result.get('requirement_completeness', []))} requirements")
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Error processing requirement completeness: {str(e)}")
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying requirement completeness analysis after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                
            except Exception as e:
                logger.error(f"Requirement completeness API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying requirement completeness analysis after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If we've exhausted all retries, return a default response
        logger.error("All requirement completeness analysis attempts failed, returning default")
        return default_response
    
    def analyze_requirements_completeness(self, requirements, domain_model):
        """Analyze requirements for completeness against the domain model (original function, kept for compatibility)"""
        logger.info("Analyzing requirements completeness")
        
        # Create a default analysis result in case all API calls fail
        default_analysis = create_default_analysis()
        
        # Get more detailed missing requirements using the specialized function
        missing_requirements_result = self.detect_missing_requirements(requirements, domain_model)
        
        # Get individual requirement completeness analysis
        requirement_completeness_result = self.analyze_requirement_completeness(requirements, domain_model)
        
        # Simplified prompt to reduce response size
        prompt = """
        You are a requirements analyst. Identify requirement issues compared to the domain model.
        
        Be concise and focus on the most important issues. Limit your analysis to critical problems.
        
        FORMAT YOUR RESPONSE AS JSON:
        {
            "requirement_issues": [
                {
                    "requirement_id": "R1",
                    "requirement_text": "text",
                    "issues": [
                        {
                            "issue_type": "Incomplete|Missing|Conflict|Inconsistency",
                            "severity": "MUST FIX|SHOULD FIX|SUGGESTION",
                            "description": "issue description",
                            "suggested_fix": "suggested fix",
                            "affected_model_elements": ["Class1", "Relationship2"]
                        }
                    ]
                }
            ],
            "domain_model_issues": [
                {
                    "element_type": "Class|Relationship|Attribute|Method",
                    "element_name": "element name",
                    "issue_type": "Missing|Incomplete|Inconsistent",
                    "severity": "MUST FIX|SHOULD FIX|SUGGESTION",
                    "description": "issue description",
                    "suggested_fix": "suggested fix",
                    "affected_requirements": ["R1"]
                }
            ]
        }
        
        REQUIREMENTS:
        
        """
        
        try:
            domain_model_text = json.dumps(domain_model, indent=2)
            logger.debug(f"Domain model JSON length: {len(domain_model_text)}")
        except Exception as e:
            logger.error(f"Error serializing domain model: {str(e)}")
            return {
                "analysis": default_analysis, 
                "error": f"Error serializing domain model: {str(e)}",
                "missing_requirements": missing_requirements_result.get("missing_requirements", []),
                "requirement_completeness": requirement_completeness_result.get("requirement_completeness", [])
            }
        
        full_prompt = prompt + requirements + "\n\nDOMAIN MODEL:\n" + domain_model_text
        messages = [{"role": "user", "content": full_prompt}]
        
        logger.debug(f"Analysis prompt length: {len(full_prompt)}")
        
        # Try with the OpenAI SDK, with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending requirements analysis request (attempt {attempt+1}/{self.max_retries})")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                
                logger.info("Received analysis response from DeepSeek API")
                
                # Extract content
                analysis_json = response.choices[0].message.content
                if not analysis_json:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Analysis content length: {len(analysis_json)}")
                logger.debug(f"Analysis content sample: {analysis_json[:200]}...")
                
                # Save raw analysis response for debugging
                with open(f"log\raw_analysis_response_{attempt}.txt", "w") as f:
                    f.write(analysis_json)
                
                # Attempt to parse and validate the JSON
                try:
                    analysis = extract_json_from_response(analysis_json)
                    if not analysis:
                        raise ValueError("Failed to extract valid JSON")
                    
                    # Check for required keys and create defaults if missing
                    expected_keys = ["requirement_issues", "domain_model_issues"]
                    for key in expected_keys:
                        if key not in analysis:
                            logger.warning(f"Analysis missing expected key: {key}")
                            analysis[key] = []
                    
                    # Add missing requirements from the specialized function
                    analysis["missing_requirements"] = missing_requirements_result.get("missing_requirements", [])
                    
                    # Add requirement completeness data
                    analysis["requirement_completeness"] = requirement_completeness_result.get("requirement_completeness", [])
                    
                    logger.info("Requirements analysis completed successfully")
                    logger.info(f"Found {len(analysis.get('requirement_issues', []))} requirement issues")
                    logger.info(f"Found {len(analysis.get('missing_requirements', []))} missing requirements")
                    logger.info(f"Found {len(analysis.get('domain_model_issues', []))} domain model issues")
                    logger.info(f"Found {len(analysis.get('requirement_completeness', []))} requirement completeness entries")
                    
                    # Try to extract reasoning content if available
                    try:
                        reasoning_content = response.choices[0].message.reasoning_content
                    except:
                        reasoning_content = "Reasoning information not available"
                    
                    return {
                        "analysis": analysis,
                        "reasoning": reasoning_content
                    }
                except Exception as e:
                    logger.warning(f"Error processing analysis: {str(e)}")
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying analysis after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                
            except Exception as e:
                logger.error(f"Analysis API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying analysis after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If we've exhausted all retries, return a default analysis with our specialized data included
        logger.error("All analysis attempts failed, returning default analysis with specialized data")
        default_analysis["missing_requirements"] = missing_requirements_result.get("missing_requirements", [])
        default_analysis["requirement_completeness"] = requirement_completeness_result.get("requirement_completeness", [])
        
        return {
            "analysis": default_analysis,
            "error": "Failed to get analysis from API after multiple attempts",
            "reasoning": "Analysis could not be completed due to API errors"
        }
    
    def update_domain_model(self, domain_model, accepted_changes):
        """Update the domain model based on accepted changes"""
        logger.info(f"Updating domain model with {len(accepted_changes)} accepted changes")
        
        if not accepted_changes:
            return domain_model
            
        # Create a prompt to update the domain model
        prompt = """
        You are an expert software architect. Update the domain model based on these accepted changes.
        
        FORMAT YOUR RESPONSE AS JSON with the same structure as the input domain model.
        
        ORIGINAL DOMAIN MODEL:
        """
        
        domain_model_text = json.dumps(domain_model, indent=2)
        
        changes_text = "ACCEPTED CHANGES:\n"
        for i, change in enumerate(accepted_changes):
            changes_text += f"{i+1}. {change['type']}: {change['description']}\n"
            if 'suggested_text' in change:
                changes_text += f"   Suggested text: {change['suggested_text']}\n"
            if 'affected_elements' in change:
                changes_text += f"   Affected elements: {', '.join(change['affected_elements'])}\n"
            changes_text += "\n"
        
        full_prompt = prompt + domain_model_text + "\n\n" + changes_text
        messages = [{"role": "user", "content": full_prompt}]
        
        logger.debug(f"Domain model update prompt length: {len(full_prompt)}")
        
        # Call the API with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending domain model update request (attempt {attempt+1}/{self.max_retries})")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                
                logger.info("Received domain model update response")
                
                # Extract content
                result_json = response.choices[0].message.content
                if not result_json:
                    raise ValueError("Empty response content")
                
                # Parse the updated domain model
                try:
                    updated_domain_model = extract_json_from_response(result_json)
                    if not updated_domain_model:
                        raise ValueError("Failed to extract valid JSON")
                    
                    # Validate and fix the updated domain model
                    updated_domain_model = validate_domain_model(updated_domain_model)
                    
                    logger.info("Domain model successfully updated")
                    return updated_domain_model
                    
                except Exception as e:
                    logger.warning(f"Error processing updated domain model: {str(e)}")
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying domain model update after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error("Failed to parse updated domain model JSON, returning original")
                        return domain_model
                
            except Exception as e:
                logger.error(f"Domain model update API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying domain model update after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    logger.error("All domain model update attempts failed, returning original")
                    return domain_model
        
        # If we reached here, all attempts failed
        return domain_model
    
    def generate_plantUML_image(self, plantuml_code):
        """Generate a PNG image from PlantUML code"""
        return generate_plantuml_image(plantuml_code)