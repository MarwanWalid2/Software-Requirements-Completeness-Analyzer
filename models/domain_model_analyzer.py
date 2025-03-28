import json
import logging
import time
import traceback
import concurrent.futures
from typing import List, Dict, Any

from services.llm_adapters import get_adapter
from services.results_aggregator import ResultsAggregator
from services.plantuml_service import generate_plantuml_image
from utils.json_utils import extract_json_from_response, validate_domain_model, create_default_analysis

logger = logging.getLogger(__name__)

class DomainModelAnalyzer:
    """Analyzes requirements and generates domain models using multiple LLM backends"""
    
    def __init__(self):
        """Initialize the analyzer"""
        # We no longer initialize specific clients here
        # Instead, we'll use the adapters as needed
        self.max_retries = 1
        self.retry_delay = 5  # seconds


    def extract_requirements_from_srs(self, srs_content, selected_models=None, meta_model_id=None, model_weights=None):
        """
        Extract requirements from a complete SRS document
        
        Args:
            srs_content (str): The full SRS document content
            selected_models (list): List of model IDs to use
            meta_model_id (str): ID of the meta model to use for aggregation
            model_weights (dict): Custom weights for each model when using weighted voting
        
        Returns:
            dict: Extracted requirements and metadata
        """
        logger.info(f"Extracting requirements from SRS using {len(selected_models) if selected_models else 0} models")
        
        if not selected_models:
            # Use Deepseek as fallback for backward compatibility
            logger.warning("No models specified, falling back to DeepSeek")
            selected_models = ["deepseek"]
        
        # Prompt for requirement extraction with detailed instructions
        prompt = """
        You are an expert in software requirements analysis. Your task is to extract all requirements from the provided Software Requirements Specification (SRS) document.

        Guidelines for extraction:
        1. Identify requirements (clearly stated, often with "shall" or "must", or if even it is in a scenario format) and their sub-requirements (ex: REQ1 and REQ 1.1 REQ1.2 REQ1.3 etc)
        2. Understand the system context first before extracting requirements to ensure accurate extraction
        3. Maintain traceability by keeping requirement IDs if present or assigning new IDs (e.g., REQ-001, REQ-002)
        4. Group related requirements together
        5. Each extracted requirement should be self-contained and clearly expressed
        6. Include both functional requirements (what the system should do) and non-functional requirements (constraints)
        7. Format each requirement with ID and description

        OUTPUT FORMAT:
        Output ONLY the extracted requirements in this format, with ONE requirement per line:
        
        REQ-ID: Requirement description
        
        For example:
        REQ-001: The system shall allow users to register an account with email and password.
        REQ-002: The system shall validate all user inputs for security purposes.
        
        DO NOT include any other text, explanations, or commentary in your response. 
        ONLY return the numbered requirements list. RETURN THE RREQUIREMENTS IN THE SAME WORDING AS IN THE SRS DOCUMENT. ALWAYS USE ORIGINAL TEXT.

        Here is the SRS document:
        """
        
        messages = [{"role": "user", "content": prompt + srs_content}]
        
        # If only one model selected, use it directly
        if len(selected_models) == 1:
            model_id = selected_models[0]
            return self._extract_requirements_with_model(model_id, messages)
        
        # Use multiple models and aggregate results
        model_results = self._run_models_in_parallel(
            selected_models, 
            "extract_requirements", 
            messages
        )
        
        logger.info(f"Got {len(model_results)} results from models")
        
        # Add debugging for the model results
        for i, result in enumerate(model_results):
            model_id = result.get("model_id", f"model-{i}")
            reqs_count = result.get("requirements_count", 0)
            has_error = "error" in result
            logger.info(f"Model {model_id} extracted {reqs_count} requirements, has error: {has_error}")
            if has_error:
                logger.info(f"Error: {result['error']}")
        
        # Aggregate results
        aggregator = ResultsAggregator(meta_model_id or "majority", model_weights)
        return aggregator.aggregate_extracted_requirements(model_results)

    def _extract_requirements_with_model(self, model_id, messages):
        """
        Extract requirements using a specific LLM
        
        Args:
            model_id (str): ID of the model to use
            messages (list): Messages to send to the LLM
        
        Returns:
            dict: Extracted requirements and metadata
        """
        logger.info(f"Extracting requirements with {model_id}")
        
        # Get the adapter for this model
        adapter = get_adapter(model_id)
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling {model_id} API for requirements extraction (attempt {attempt+1}/{self.max_retries})")
                
                # Generate response
                response = adapter.generate_response(messages)
                
                # Extract the response content
                extracted_text = response["content"]
                if not extracted_text:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Extracted content length: {len(extracted_text)}")
                logger.debug(f"Extracted content sample: {extracted_text[:500]}...")
                
                # Process the extracted requirements
                requirements_list = []
                for line in extracted_text.strip().split('\n'):
                    line = line.strip()
                    if line and (':' in line):
                        requirements_list.append(line)
                
                logger.info(f"Extracted {len(requirements_list)} requirements")
                
                # Combine the requirements into a single string
                extracted_requirements = '\n'.join(requirements_list)
                
                return {
                    "model_id": model_id,
                    "extracted_requirements": extracted_requirements,
                    "requirements_count": len(requirements_list),
                    "requirements_list": requirements_list
                }
                    
            except Exception as e:
                logger.error(f"{model_id} API request error for requirements extraction (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If all attempts failed, return an empty result
        logger.error(f"All {model_id} API attempts failed for requirements extraction")
        return {
            "model_id": model_id,
            "extracted_requirements": "",
            "requirements_count": 0,
            "requirements_list": [],
            "error": f"All {model_id} API attempts failed"
        }
        
    def create_domain_model(self, requirements, selected_models=None, meta_model_id=None, model_weights=None):
        """
        Generate a domain model from requirements using one or more LLMs
        
        Args:
            requirements (str): The requirements text
            selected_models (list): List of model IDs to use
            meta_model_id (str): ID of the meta model to use for aggregation
            model_weights (dict): Custom weights for each model when using weighted voting
        
        Returns:
            dict: Domain model and reasoning
        """
        logger.info(f"Creating domain model using {len(selected_models) if selected_models else 0} models")
        
        if not selected_models:
            # Use Deepseek as fallback for backward compatibility
            logger.warning("No models specified, falling back to DeepSeek")
            selected_models = ["deepseek"]
        
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
        
        # If only one model selected, use it directly
        if len(selected_models) == 1:
            model_id = selected_models[0]
            return self._create_domain_model_with_model(model_id, messages)
        
        # Use multiple models and aggregate results
        model_results = self._run_models_in_parallel(
            selected_models, 
            "create_domain_model", 
            messages
        )
        
        # Aggregate results
        aggregator = ResultsAggregator(meta_model_id or "majority", model_weights)
        return aggregator.aggregate_domain_models(model_results)
    
    def _create_domain_model_with_model(self, model_id, messages,model_weights=None):
        """
        Generate a domain model using a specific LLM
        
        Args:
            model_id (str): ID of the model to use
            messages (list): Messages to send to the LLM
        
        Returns:
            dict: Domain model and reasoning
        """
        logger.info(f"Creating domain model with {model_id}")
        
        # Get the adapter for this model
        adapter = get_adapter(model_id)
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling {model_id} API (attempt {attempt+1}/{self.max_retries})")
                
                # Generate response
                response = adapter.generate_response(messages)
                
                # Extract and validate the response content
                domain_model_json = response["content"]
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
                    reasoning_content = f"Domain model generated by {model_id}"
                    
                    return {
                        "model_id": model_id,
                        "domain_model": domain_model,
                        "reasoning": reasoning_content
                    }
                except Exception as e:
                    logger.warning(f"Error processing domain model: {str(e)}")
                    
                    # Save the raw response for debugging
                    with open(f"log/raw_domain_model_response_{model_id}_{attempt}.txt", "w") as f:
                        f.write(domain_model_json)
                    
                    # If not the last attempt, try again
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying API call after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                    
            except Exception as e:
                logger.error(f"{model_id} API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If all attempts failed, return a default domain model
        logger.error(f"All {model_id} API attempts failed. Returning default domain model")
        return {
            "model_id": model_id,
            "domain_model": {
                "classes": [],
                "relationships": [],
                "plantuml": "@startuml\n@enduml"
            },
            "error": f"All {model_id} API attempts failed",
            "reasoning": "Error occurred during model generation"
        }
    
    def detect_missing_requirements(self, requirements, domain_model, selected_models=None, meta_model_id=None, model_weights=None):
        """
        Specialized function to detect missing requirements based on domain model and natural language
        
        Args:
            requirements (str): The requirements text
            domain_model (dict): The domain model
            selected_models (list): List of model IDs to use
            meta_model_id (str): ID of the meta model to use for aggregation
        
        Returns:
            dict: Missing requirements analysis
        """
        logger.info(f"Detecting missing requirements using {len(selected_models) if selected_models else 0} models")
        
        if not selected_models:
            # Use Deepseek as fallback for backward compatibility
            logger.warning("No models specified, falling back to DeepSeek")
            selected_models = ["deepseek"]
        
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
        
        # If only one model selected, use it directly
        if len(selected_models) == 1:
            model_id = selected_models[0]
            return self._detect_missing_requirements_with_model(model_id, messages, default_response)
        
        # Use multiple models and aggregate results
        model_results = self._run_models_in_parallel(
            selected_models, 
            "detect_missing_requirements", 
            messages, 
            default_response
        )
        
        # Aggregate results - use the missing_requirements field only
        aggregated_results = []
        for result in model_results:
            if result and "missing_requirements" in result:
                result_copy = result.copy()
                result_copy["analysis"] = {"missing_requirements": result["missing_requirements"]}
                aggregated_results.append(result_copy)
        
        # Aggregate results
        aggregator = ResultsAggregator(meta_model_id or "majority", model_weights)
        aggregated = aggregator.aggregate_analysis_results(aggregated_results)
        
        # Extract just the missing_requirements
        if "analysis" in aggregated and "missing_requirements" in aggregated["analysis"]:
            return {"missing_requirements": aggregated["analysis"]["missing_requirements"]}
        
        return default_response
    
    def _detect_missing_requirements_with_model(self, model_id, messages, default_response, model_weights=None):
        """
        Detect missing requirements using a specific LLM
        
        Args:
            model_id (str): ID of the model to use
            messages (list): Messages to send to the LLM
            default_response (dict): Default response in case of failure
        
        Returns:
            dict: Missing requirements analysis
        """
        logger.info(f"Detecting missing requirements with {model_id}")
        
        # Get the adapter for this model
        adapter = get_adapter(model_id)
        
        # Call the API with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending missing requirements detection request to {model_id} (attempt {attempt+1}/{self.max_retries})")
                
                # Generate response
                response = adapter.generate_response(messages)
                
                # Extract content
                result_json = response["content"]
                if not result_json:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Missing requirements content sample: {result_json[:200]}...")
                
                # Save raw response for debugging
                with open(f"log/raw_missing_requirements_{model_id}_{attempt}.txt", "w") as f:
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
                    
                    # Add model ID
                    result["model_id"] = model_id
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Error processing missing requirements: {str(e)}")
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying missing requirements detection after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                
            except Exception as e:
                logger.error(f"Missing requirements API request error to {model_id} (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying missing requirements detection after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If we've exhausted all retries, return a default response
        logger.error(f"All missing requirements detection attempts with {model_id} failed, returning default")
        default_response["model_id"] = model_id
        return default_response
    
    def analyze_requirement_completeness(self, requirements, domain_model, selected_models=None, meta_model_id=None, model_weights=None):
        """
        Specialized function to analyze individual requirement completeness
        
        Args:
            requirements (str): The requirements text
            domain_model (dict): The domain model
            selected_models (list): List of model IDs to use
            meta_model_id (str): ID of the meta model to use for aggregation
        
        Returns:
            dict: Requirement completeness analysis
        """
        logger.info(f"Analyzing individual requirement completeness using {len(selected_models) if selected_models else 0} models")
        
        if not selected_models:
            # Use Deepseek as fallback for backward compatibility
            logger.warning("No models specified, falling back to DeepSeek")
            selected_models = ["deepseek"]
        
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
        
        # If only one model selected, use it directly
        if len(selected_models) == 1:
            model_id = selected_models[0]
            return self._analyze_requirement_completeness_with_model(model_id, messages, default_response)
        
        # Use multiple models and aggregate results
        model_results = self._run_models_in_parallel(
            selected_models, 
            "analyze_requirement_completeness", 
            messages, 
            default_response
        )
        
        # Aggregate results - use the requirement_completeness field only
        aggregated_results = []
        for result in model_results:
            if result and "requirement_completeness" in result:
                result_copy = result.copy()
                result_copy["analysis"] = {"requirement_completeness": result["requirement_completeness"]}
                aggregated_results.append(result_copy)
        
        # Aggregate results
        aggregator = ResultsAggregator(meta_model_id or "majority", model_weights)
        aggregated = aggregator.aggregate_analysis_results(aggregated_results)
        
        # Extract just the requirement_completeness
        if "analysis" in aggregated and "requirement_completeness" in aggregated["analysis"]:
            return {"requirement_completeness": aggregated["analysis"]["requirement_completeness"]}
        
        return default_response
    
    def _analyze_requirement_completeness_with_model(self, model_id, messages, default_response, model_weights=None):
        """
        Analyze requirement completeness using a specific LLM
        
        Args:
            model_id (str): ID of the model to use
            messages (list): Messages to send to the LLM
            default_response (dict): Default response in case of failure
        
        Returns:
            dict: Requirement completeness analysis
        """
        logger.info(f"Analyzing requirement completeness with {model_id}")
        
        # Get the adapter for this model
        adapter = get_adapter(model_id)
        
        # Call the API with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending requirement completeness analysis request to {model_id} (attempt {attempt+1}/{self.max_retries})")
                
                # Generate response
                response = adapter.generate_response(messages)
                
                # Extract content
                result_json = response["content"]
                if not result_json:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Requirement completeness content sample: {result_json[:200]}...")
                
                # Save raw response for debugging
                with open(f"log/raw_requirement_completeness_{model_id}_{attempt}.txt", "w") as f:
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
                    
                    # Add model ID
                    result["model_id"] = model_id
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Error processing requirement completeness: {str(e)}")
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying requirement completeness analysis after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                
            except Exception as e:
                logger.error(f"Requirement completeness API request error to {model_id} (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying requirement completeness analysis after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If we've exhausted all retries, return a default response
        logger.error(f"All requirement completeness analysis attempts with {model_id} failed, returning default")
        default_response["model_id"] = model_id
        return default_response
    
    def analyze_requirements_completeness(self, requirements, domain_model, selected_models=None, meta_model_id=None, model_weights=None):
        """
        Analyze requirements for completeness against the domain model
        
        Args:
            requirements (str): The requirements text
            domain_model (dict): The domain model
            selected_models (list): List of model IDs to use
            meta_model_id (str): ID of the meta model to use for aggregation
        
        Returns:
            dict: Analysis results and reasoning
        """
        logger.info(f"Analyzing requirements completeness using {len(selected_models) if selected_models else 0} models")
        
        if not selected_models:
            # Use Deepseek as fallback for backward compatibility
            logger.warning("No models specified, falling back to DeepSeek")
            selected_models = ["deepseek"]
        
        # Get more detailed missing requirements using the specialized function
        missing_requirements_result = self.detect_missing_requirements(requirements, domain_model, selected_models, meta_model_id, model_weights)
        
        # Get individual requirement completeness analysis
        requirement_completeness_result = self.analyze_requirement_completeness(requirements, domain_model, selected_models, meta_model_id, model_weights)
        
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
                "analysis": create_default_analysis(), 
                "error": f"Error serializing domain model: {str(e)}",
                "missing_requirements": missing_requirements_result.get("missing_requirements", []),
                "requirement_completeness": requirement_completeness_result.get("requirement_completeness", [])
            }
        
        full_prompt = prompt + requirements + "\n\nDOMAIN MODEL:\n" + domain_model_text
        messages = [{"role": "user", "content": full_prompt}]
        
        logger.debug(f"Analysis prompt length: {len(full_prompt)}")
        
        # If only one model selected, use it directly
        if len(selected_models) == 1:
            model_id = selected_models[0]
            main_analysis = self._analyze_requirements_with_model(
                model_id, 
                messages, 
                missing_requirements_result, 
                requirement_completeness_result
            )
            return main_analysis
        
        # Use multiple models and aggregate results
        model_results = self._run_models_in_parallel(
            selected_models, 
            "analyze_requirements", 
            messages
        )
        
        # Add missing requirements and completeness to each result
        for result in model_results:
            if "analysis" in result:
                result["analysis"]["missing_requirements"] = missing_requirements_result.get("missing_requirements", [])
                result["analysis"]["requirement_completeness"] = requirement_completeness_result.get("requirement_completeness", [])
        
        # Aggregate results
        aggregator = ResultsAggregator(meta_model_id or "majority", model_weights)
        return aggregator.aggregate_analysis_results(model_results)
    
    def _analyze_requirements_with_model(self, model_id, messages, missing_requirements_result, requirement_completeness_result, model_weights=None):
        """
        Analyze requirements using a specific LLM
        
        Args:
            model_id (str): ID of the model to use
            messages (list): Messages to send to the LLM
            missing_requirements_result (dict): Results from missing requirements analysis
            requirement_completeness_result (dict): Results from requirement completeness analysis
        
        Returns:
            dict: Analysis results and reasoning
        """
        logger.info(f"Analyzing requirements with {model_id}")
        
        # Create a default analysis result in case all API calls fail
        default_analysis = create_default_analysis()
        
        # Get the adapter for this model
        adapter = get_adapter(model_id)
        
        # Try with the API, with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending requirements analysis request to {model_id} (attempt {attempt+1}/{self.max_retries})")
                
                # Generate response
                response = adapter.generate_response(messages)
                
                # Extract content
                analysis_json = response["content"]
                if not analysis_json:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Analysis content length: {len(analysis_json)}")
                logger.debug(f"Analysis content sample: {analysis_json[:200]}...")
                
                # Save raw analysis response for debugging
                with open(f"log/raw_analysis_response_{model_id}_{attempt}.txt", "w") as f:
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
                    
                    return {
                        "model_id": model_id,
                        "analysis": analysis,
                        "reasoning": f"Analysis generated by {model_id}"
                    }
                except Exception as e:
                    logger.warning(f"Error processing analysis: {str(e)}")
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying analysis after error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                
            except Exception as e:
                logger.error(f"Analysis API request error to {model_id} (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying analysis after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If we've exhausted all retries, return a default analysis with our specialized data included
        logger.error(f"All analysis attempts with {model_id} failed, returning default analysis with specialized data")
        default_analysis["missing_requirements"] = missing_requirements_result.get("missing_requirements", [])
        default_analysis["requirement_completeness"] = requirement_completeness_result.get("requirement_completeness", [])
        
        return {
            "model_id": model_id,
            "analysis": default_analysis,
            "error": f"Failed to get analysis from {model_id} API after multiple attempts",
            "reasoning": "Analysis could not be completed due to API errors"
        }
    
    def update_domain_model(self, domain_model, accepted_changes, selected_models=None, meta_model_id=None, model_weights=None):
        """
        Update the domain model based on accepted changes
        
        Args:
            domain_model (dict): The domain model to update
            accepted_changes (list): List of accepted changes
            selected_models (list): List of model IDs to use
            meta_model_id (str): ID of the meta model to use for aggregation
        
        Returns:
            dict: Updated domain model
        """
        logger.info(f"Updating domain model with {len(accepted_changes)} accepted changes using {len(selected_models) if selected_models else 0} models")
        
        if not selected_models:
            # Use Deepseek as fallback for backward compatibility
            logger.warning("No models specified, falling back to DeepSeek")
            selected_models = ["deepseek"]
        
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
        
        # If only one model selected, use it directly
        if len(selected_models) == 1:
            model_id = selected_models[0]
            return self._update_domain_model_with_model(model_id, messages, domain_model)
        
        # Use multiple models and aggregate results
        model_results = self._run_models_in_parallel(
            selected_models, 
            "update_domain_model", 
            messages, 
            {"domain_model": domain_model}
        )
        
        # Aggregate results
        aggregator = ResultsAggregator(meta_model_id or "majority", model_weights)
        aggregated = aggregator.aggregate_domain_models(model_results)
        
        return aggregated.get("domain_model", domain_model)
    
    def _update_domain_model_with_model(self, model_id, messages, domain_model, model_weights=None):
        """
        Update the domain model using a specific LLM
        
        Args:
            model_id (str): ID of the model to use
            messages (list): Messages to send to the LLM
            domain_model (dict): The original domain model
        
        Returns:
            dict: Updated domain model
        """
        logger.info(f"Updating domain model with {model_id}")
        
        # Get the adapter for this model
        adapter = get_adapter(model_id)
        
        # Call the API with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending domain model update request to {model_id} (attempt {attempt+1}/{self.max_retries})")
                
                # Generate response
                response = adapter.generate_response(messages)
                
                # Extract content
                result_json = response["content"]
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
                logger.error(f"Domain model update API request error to {model_id} (attempt {attempt+1}): {str(e)}")
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
    
    def _run_models_in_parallel(self, model_ids, operation_type, messages, default_result=None):
        """
        Run multiple models in parallel and collect their results
        
        Args:
            model_ids (list): List of model IDs to use
            operation_type (str): Type of operation to perform
            messages (list): Messages to send to LLMs
            default_result (dict): Default result to use in case of failure
        
        Returns:
            list: Results from all models
        """
        logger.info(f"Running {operation_type} with {len(model_ids)} models in parallel")
        
        results = []
        
        # Define the worker function based on operation type
        def worker(model_id):
            try:
                if operation_type == "create_domain_model":
                    return self._create_domain_model_with_model(model_id, messages)
                elif operation_type == "detect_missing_requirements":
                    return self._detect_missing_requirements_with_model(model_id, messages, default_result or {"missing_requirements": []})
                elif operation_type == "analyze_requirement_completeness":
                    return self._analyze_requirement_completeness_with_model(model_id, messages, default_result or {"requirement_completeness": []})
                elif operation_type == "analyze_requirements":
                    # For this operation, we don't have the missing_requirements_result and requirement_completeness_result
                    # So we'll just create default ones
                    missing_req = {"missing_requirements": []}
                    req_completeness = {"requirement_completeness": []}
                    return self._analyze_requirements_with_model(model_id, messages, missing_req, req_completeness)
                elif operation_type == "update_domain_model":
                    return {"model_id": model_id, "domain_model": self._update_domain_model_with_model(model_id, messages, default_result.get("domain_model", {}))}
                elif operation_type == "extract_requirements":
                    return self._extract_requirements_with_model(model_id, messages)
                else:
                    logger.error(f"Unknown operation type: {operation_type}")
                    return {"error": f"Unknown operation type: {operation_type}"}
            except Exception as e:
                logger.error(f"Error in {operation_type} worker for {model_id}: {str(e)}")
                logger.error(traceback.format_exc())
                return {"model_id": model_id, "error": str(e)}
        
        # Use ThreadPoolExecutor to run models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_ids)) as executor:
            # Submit all tasks
            future_to_model = {executor.submit(worker, model_id): model_id for model_id in model_ids}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        logger.warning(f"Null result from {model_id}")
                except Exception as e:
                    logger.error(f"Exception in thread for {model_id}: {str(e)}")
                    logger.error(traceback.format_exc())
        
        return results