import json
import logging
import time
import traceback
import concurrent.futures
import os
from datetime import datetime
from typing import List, Dict, Any

from services.llm_adapters import get_adapter
from services.results_aggregator import ResultsAggregator
from services.plantuml_service import generate_plantuml_image
from utils.json_utils import extract_json_from_response, validate_domain_model, create_default_analysis
from flask import current_app
from flask import session


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
            # Use openai as fallback for backward compatibility
            logger.warning("No models specified, falling back to openai")
            selected_models = ["openai"]
        
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
        return aggregator.aggregate_extraction_results(model_results)

    def _extract_context_with_model(self, model_id, messages):
        """
        Extract context information using a specific LLM
        
        Args:
            model_id (str): ID of the model to use
            messages (list): Messages to send to the LLM
        
        Returns:
            dict: Extracted context and metadata
        """
        logger.info(f"Extracting context with {model_id}")
        
        # Get the adapter for this model
        adapter = get_adapter(model_id)
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling {model_id} API for context extraction (attempt {attempt+1}/{self.max_retries})")
                
                # Generate response
                response = adapter.generate_response(messages)
                
                # Extract the response content
                extracted_text = response["content"]
                if not extracted_text:
                    raise ValueError("Empty response content")
                
                logger.debug(f"Extracted context length: {len(extracted_text)}")
                logger.debug(f"Extracted context sample: {extracted_text[:500]}...")
                
                # Try to extract JSON from the response
                try:
                    context_data = extract_json_from_response(extracted_text)
                    if not context_data:
                        raise ValueError("Failed to extract valid JSON from context extraction")
                    
                    # Add model ID
                    context_data["model_id"] = model_id
                    
                    return context_data
                except Exception as json_error:
                    logger.warning(f"Error extracting JSON from context extraction response: {str(json_error)}")
                    # Fall back to returning the raw text if we can't parse JSON
                    return {
                        "model_id": model_id,
                        "raw_context": extracted_text,
                        "json_error": str(json_error)
                    }
                    
            except Exception as e:
                logger.error(f"{model_id} API request error for context extraction (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If all attempts failed, return an empty result
        logger.error(f"All {model_id} API attempts failed for context extraction")
        return {
            "model_id": model_id,
            "system_overview": "",
            "stakeholders": [],
            "terminology": {},
            "assumptions": [],
            "external_systems": [],
            "business_rules": [],
            "error": f"All {model_id} API attempts failed"
        }
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
            # Use openai as fallback for backward compatibility
            logger.warning("No models specified, falling back to openai")
            selected_models = ["openai"]
        
        # Simplify the prompt to reduce response size
        prompt = """
You are an expert software architect. Create a domain model that ONLY includes elements mentioned in the requirements.

STRICT RULES:
1. do no add any elements, attributes, or methods based on your own knowledge or assumptions
2. do not "complete" the model with what you think should be there, but rather but what's actually there 
3. If requirements are minimal, your domain model should be equally minimal
4. NEVER invent or assume any elements not stated in the requirements
5. If a requirement mentions an entity, attribute, or relationship, it must be included in the domain model
6. If a requirement mentions an operation, it must be included in the domain model
7. If a requirement mentions a class, it must be included in the domain model
8. If a requirement mentions an attribute, it must be included in the domain model
9. If a requirement mentions a relationship, it must be included in the domain model

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
    
    def _create_domain_model_with_model(self, model_id, messages, model_weights=None):
        """
        Generate a domain model using a specific LLM with improved error handling
        
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
                
                # Save debug information for this attempt
                self._save_domain_model_debug(
                    model_id=model_id,
                    attempt=attempt + 1,
                    messages=messages,
                    response_content=domain_model_json
                )
                
                # Try to parse and validate the JSON with enhanced error handling
                try:
                    from utils.json_utils import extract_json_from_response, validate_domain_model
                    from utils.json_debug_utils import save_problematic_json, find_json_error_location, suggest_json_fixes
                    
                    domain_model = extract_json_from_response(domain_model_json, expected_top_level_keys=["classes", "relationships"])
                    if not domain_model:
                        # Enhanced debugging for JSON parsing failures
                        error_context = f"{model_id}_domain_model_attempt_{attempt+1}"
                        debug_file = save_problematic_json(domain_model_json, "Failed to extract valid JSON", error_context)
                        
                        # Save debug info with error
                        self._save_domain_model_debug(
                            model_id=model_id,
                            attempt=attempt + 1,
                            messages=messages,
                            response_content=domain_model_json,
                            error="Failed to extract valid JSON"
                        )
                        
                        logger.error(f"Failed to extract valid JSON from {model_id} domain model response")
                        logger.error(f"Debug file saved: {debug_file}")
                        
                        raise ValueError("Failed to extract valid JSON")
                    
                    # Validate and fix the domain model
                    domain_model = validate_domain_model(domain_model)
                    
                    logger.info("Domain model successfully created")
                    logger.info(f"Model contains {len(domain_model.get('classes', []))} classes and {len(domain_model.get('relationships', []))} relationships")
                    
                    # Save successful debug info
                    self._save_domain_model_debug(
                        model_id=model_id,
                        attempt=attempt + 1,
                        messages=messages,
                        response_content=domain_model_json,
                        domain_model=domain_model
                    )
                    
                    reasoning_content = f"Domain model generated by {model_id}"
                    
                    return {
                        "model_id": model_id,
                        "domain_model": domain_model,
                        "reasoning": reasoning_content
                    }
                    
                except json.JSONDecodeError as json_error:
                    # Enhanced JSON error handling
                    logger.error(f"JSON parsing error in {model_id} domain model response: {str(json_error)}")
                    
                    # Save debug information with JSON error
                    self._save_domain_model_debug(
                        model_id=model_id,
                        attempt=attempt + 1,
                        messages=messages,
                        response_content=domain_model_json,
                        error=f"JSON parsing error: {str(json_error)}"
                    )
                    
                    # Save debug information
                    error_context = f"{model_id}_domain_model_json_error_attempt_{attempt+1}"
                    debug_file = save_problematic_json(domain_model_json, str(json_error), error_context)
                    
                    # Get detailed error analysis
                    from utils.json_debug_utils import find_json_error_location, suggest_json_fixes
                    error_info = find_json_error_location(domain_model_json, str(json_error))
                    if error_info:
                        logger.error(f"JSON error location: {error_info}")
                        suggestions = suggest_json_fixes(domain_model_json, error_info)
                        logger.error(f"Suggested fixes: {suggestions}")
                    
                    # If not the last attempt, try again
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying domain model generation after JSON error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"All JSON parsing attempts failed for {model_id} domain model")
                        
                except Exception as e:
                    logger.warning(f"Error processing domain model: {str(e)}")
                    
                    # Save debug information for any processing error
                    self._save_domain_model_debug(
                        model_id=model_id,
                        attempt=attempt + 1,
                        messages=messages,
                        response_content=domain_model_json,
                        error=f"Processing error: {str(e)}"
                    )
                    
                    error_context = f"{model_id}_domain_model_processing_error_attempt_{attempt+1}"
                    debug_file = save_problematic_json(domain_model_json, str(e), error_context)
                    
                    # If not the last attempt, try again
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying domain model generation after processing error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                        
            except Exception as e:
                logger.error(f"{model_id} API request error (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # Save debug information for API errors
                self._save_domain_model_debug(
                    model_id=model_id,
                    attempt=attempt + 1,
                    messages=messages,
                    response_content="",
                    error=f"API request error: {str(e)}"
                )
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If all attempts failed, return a default domain model
        logger.error(f"All {model_id} API attempts failed. Returning default domain model")
        
        # Save final failure debug info
        self._save_domain_model_debug(
            model_id=model_id,
            attempt="final_failure",
            messages=messages,
            response_content="",
            error=f"All {model_id} API attempts failed"
        )
        
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

    def extract_context_from_srs(self, srs_content, selected_models=None, meta_model_id=None):
        """
        Extract important contextual information from the SRS document,
        separate from the requirements themselves.
        
        Args:
            srs_content (str): The full SRS document content
            selected_models (list): List of model IDs to use
            meta_model_id (str): ID of the meta model for aggregation
            
        Returns:
            dict: Extracted context and metadata
        """
        prompt = """
        You are an expert in software requirements analysis. Your task is to extract ONLY the important contextual information
        from this Software Requirements Specification (SRS) document - NOT the requirements themselves.
        
        Extract:
        1. System overview and purpose
        2. Stakeholders and user descriptions
        3. Definitions and terminology
        4. Assumptions and dependencies
        5. External interfaces and systems
        6. Business rules and constraints
        
        DO NOT include the actual requirements in your response.
        
        RESPONSE FORMAT:
        {
            "system_overview": "Brief description of the system",
            "stakeholders": ["List of stakeholders"],
            "terminology": {"term": "definition", ...},
            "assumptions": ["List of assumptions"],
            "external_systems": ["List of external systems"],
            "business_rules": ["List of business rules"]
        }
        
        Here is the SRS document:
        """
        
        messages = [{"role": "user", "content": prompt + srs_content}]
        
        # Use the same model infrastructure as requirements extraction
        if len(selected_models) == 1:
            model_id = selected_models[0]
            result = self._extract_requirements_with_model(model_id, messages)
        else:
            model_results = self._run_models_in_parallel(
                selected_models, 
                "extract_context", 
                messages
            )
            aggregator = ResultsAggregator(meta_model_id or "majority")
            result = aggregator.aggregate_extraction_results(model_results)
        
        return result
    
    def detect_missing_requirements(self, requirements, domain_model, selected_models=None, meta_model_id=None, model_weights=None):
        """
        Detect missing requirements using LLMs with a general, domain-agnostic approach
        
        Args:
            requirements (str): The requirements text
            domain_model (dict): The domain model
            selected_models (list): List of model IDs to use
            meta_model_id (str): ID of the meta model to use for aggregation
            model_weights (dict): Custom weights for models when using weighted voting
        
        Returns:
            dict: Missing requirements analysis
        """
        logger.info(f"Detecting missing requirements using {len(selected_models) if selected_models else 0} models")
        
        if not selected_models:
            logger.warning("No models specified, falling back to openai")
            selected_models = ["openai"]
        
        # Create a default response in case all API calls fail
        default_response = {
            "missing_requirements": []
        }
        
        # Get document context if available
        document_context = None
        try:
            with current_app.app_context():
                document_context = session.get('document_context', {})
        except Exception as e:
            logger.warning(f"Could not get document context: {str(e)}")
        
        # Create a clear, general prompt for the LLM
        prompt = """
        You are an expert requirements analyst. Your task is to identify SPECIFIC MISSING REQUIREMENTS by analyzing the provided domain model and requirements, CONSIDER THE REQUIREMENTS AS A WHOLE SET, NOT EACH REQUIREMENT INDVIDUALLY, LEVERAGE THE DOMAIN MODEL AND ITS ENTITIES AND RELATIONSHIPS. DO NOT LIMIT YOURSELF TO JUST GAPS IN DOMAIN MODEL, BUT YOU ALSO NEED TO FIND SPECIFIC MISSING REQUIREMENTS IN THE REQUIREMENTS TEXT SPECIFICALLY FOR THIS SYSTEM

        ## INSTRUCTIONS
        1. Compare the domain model with the requirements and identify requirements that should exist but are missing
        2. Look for missing requirements related to classes, attributes, relationships, and operations in the domain model
        3. Consider industry best practices and common requirements for this type of system
        4. Focus on specific, concrete missing requirements - not just categories of missing requirements
        5. MAKE SURE YOU ARE CONSIDERING THE WHOLE SET AND THE WHOLE DOMAIN WHEN DOING YOUR ANALYSIS
        6. Ensure there are requirements covering the essential behavior and properties of each major element in the `DOMAIN MODEL`. Look for missing CRUD operations (if applicable), state transitions, handling of key attributes, or initialization/termination logic.
        7. Ensure there are requirements covering the entities in the domain model.
        8. Explicitly check for missing NFRs crucial for this type of system. This includes performance, security, usability, scalability, and maintainability aspects that should be defined but are not present in the current requirements.
        9. Explicitly check for functional requirements that are not covered by the existing requirements. This includes checking for missing use cases, scenarios, or specific behaviors that should be defined.
        10. Use the `SYSTEM CONTEXT` (if provided - e.g., stakeholders, business rules, external systems) to identify missing requirements. If a missing requirement stems directly from a specific stakeholder need or business rule mentioned there, explain this linkage clearly in the 'rationale'.
        11. AGAIN FOCUS ON THE SET AS A WHOLE, THIS SYSTEM IS UNIQUE SO YOU NEED TO MAKE SURE YOU THINK AS SPECIFICALLY AS POSSIBLE ABOUT THIS SYSTEM AND ITS DOMAIN MODEL AND REQUIREMENTS AS A WHOLE.
        12. ANOTHER ANALYSIS WOULD BE THINKINING AS BROADLY AS POSSIBLE TO TRY TO CAPTURE WHAT DID THIS SET OF REQUIREMENTS FORGET IN THIS SYSTEM DOMAIN.
        13. IF AN ENTITY OR AN ATTRIBUTE WERE MENTIONED, CHECK IF IT WAS OR WASN'T MENTIONED ELSE WHERE (AGAIN THE SET AS A WHOLE). for example: if we mention 4 attributes for user and how we display them, then later we mention 5 attributes but how we delete them, that means there was a missing requirement for the 5th attribute at the beginning of the set.
        ## OUTPUT FORMAT
        Provide a JSON response with the following structure:
        {
            "missing_requirements": [
                {
                    "id": "MR1",
                    "description": "Specific description of what's missing",
                    "category": "General category",
                    "severity": "CRITICAL|HIGH|MEDIUM|LOW",
                    "suggested_requirement": "Complete, specific requirement text that should be added",
                    "affected_model_elements": ["Class1", "Class2.attribute"],
                    "rationale": "Explanation of why this requirement should exist"
                }
            ]
        }
        """
        
        # Add document context if available
        if document_context:
            context_info = "\n## SYSTEM CONTEXT\n"
            if document_context.get('system_overview'):
                context_info += f"System Overview: {document_context.get('system_overview')}\n"
            if document_context.get('stakeholders'):
                context_info += f"Stakeholders: {', '.join(document_context.get('stakeholders'))}\n"
            if document_context.get('business_rules'):
                context_info += "Business Rules:\n"
                for rule in document_context.get('business_rules'):
                    context_info += f"- {rule}\n"
            if document_context.get('external_systems'):
                context_info += "External Systems:\n"
                for system in document_context.get('external_systems'):
                    context_info += f"- {system}\n"
            prompt += context_info
        
        # Add domain model
        prompt += "\n## DOMAIN MODEL\n"
        domain_model_text = json.dumps(domain_model, indent=2)
        prompt += domain_model_text
        
        # Add requirements
        prompt += "\n\n## REQUIREMENTS\n"
        prompt += requirements
        
        # Create messages for the LLM
        messages = [{"role": "user", "content": prompt}]
        
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
        
        # Aggregate results
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
                
                # # Save raw response for debugging
                # with open(f"log/raw_missing_requirements_{model_id}_{attempt}.txt", "w", encoding="utf-8") as f:
                #     f.write(result_json)
                
                # Parse and validate the JSON
                try:
                    result = extract_json_from_response(result_json, expected_top_level_keys=["missing_requirements"])
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
        Analyze completeness of individual requirements in a general way
        
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
            logger.warning("No models specified, falling back to openai")
            selected_models = ["openai"]
        
        # Create a default response in case all API calls fail
        default_response = {
            "requirement_completeness": []
        }
        
        # Create a general prompt
        prompt = """
                You are an expert requirements analyst. Analyze the completeness of each individual requirement.
                
                For each requirement, check if it is complete by ensuring it contains all necessary elements. Make sure to consider the requirements format first which may be in various formats, including but not limited to:
                - Scenario: A description of a specific situation or interaction.
                - Use Case Step: A step within a larger use case description.
                - "Shall" Statement: A formal requirement statement using the word "shall."
                - User Story: A short description from the user's perspective ("As a [user], I want [goal] so that [benefit]").
                - Free-form Text: A general description of desired system behavior.
                - Other declarative statement: Any other way of expressing a system requirement.
                
                Check for the following elements of a software requirement, ONLY IF the values are explicitly stated or can be DIRECTLY and LOGICALLY inferred from the provided requirement statement:
                - unique identifier: A unique ID for the requirement.
                - priority: The priority of the requirement (e.g., High, Medium, Low).
                - rationale: The reason or justification for the requirement.
                - source: The origin of the requirement (e.g., stakeholder name, document).
                - status: The current status of the requirement (e.g., Draft, Approved, Implemented).
                - acceptance criteria: Specific, testable conditions that must be met for the requirement to be considered complete.
                - dependencies: Other requirements that this requirement depends on.
                - version: The version number of the requirement.
                - type: The type of requirement (e.g., Functional, Non-functional, Security).
                - actor: The person or system that initiates the action or interacts with the system.
                - action: The action performed by the actor or the system.
                - object: The thing that the action is performed on.
                - condition: The circumstances or context under which the action occurs.
                - quality attributes: Non-functional characteristics of the system related to the requirement (e.g., performance, security, usability).
                
                ## FORMAT YOUR RESPONSE AS JSON
                {
                    "requirement_completeness": [
                        {
                            "requirement_id": "ID",
                            "requirement_text": "text",
                            "completeness_score": 0-100,
                            "missing_elements": ["element1", "element2"],
                            "suggested_improvement": "Suggested improved text",
                            "rationale": "Why this improvement is needed"
                        }
                    ]
                }
                """
        
        # Add domain model
        prompt += "\n\n## DOMAIN MODEL\n"
        domain_model_text = json.dumps(domain_model, indent=2)
        prompt += domain_model_text
        
        # Add requirements
        prompt += "\n\n## REQUIREMENTS\n"
        prompt += requirements
        
        # Create messages for the LLM
        messages = [{"role": "user", "content": prompt}]
        
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
        
        # Aggregate results
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
        Analyze requirement completeness using a specific LLM with improved error handling
        
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
                
                # Parse and validate the JSON with improved error handling
                try:
                    from utils.json_utils import extract_json_from_response
                    from utils.json_debug_utils import save_problematic_json, find_json_error_location, suggest_json_fixes
                    
                    result = extract_json_from_response(result_json)
                    if not result:
                        # Enhanced debugging for JSON parsing failures
                        error_context = f"{model_id}_requirement_completeness_attempt_{attempt+1}"
                        debug_file = save_problematic_json(result_json, "Failed to extract valid JSON", error_context)
                        
                        logger.error(f"Failed to extract valid JSON from {model_id} response")
                        logger.error(f"Debug file saved: {debug_file}")
                        
                        raise ValueError("Failed to extract valid JSON from response")
                    
                    # Ensure requirement_completeness key exists
                    if "requirement_completeness" not in result:
                        logger.warning("Missing 'requirement_completeness' key in response")
                        result["requirement_completeness"] = []
                    
                    logger.info(f"Analyzed completeness of {len(result.get('requirement_completeness', []))} requirements")
                    
                    # Add model ID
                    result["model_id"] = model_id
                    
                    return result
                    
                except json.JSONDecodeError as json_error:
                    # Enhanced JSON error handling
                    logger.error(f"JSON parsing error in {model_id} requirement completeness response: {str(json_error)}")
                    
                    # Save debug information
                    error_context = f"{model_id}_requirement_completeness_json_error_attempt_{attempt+1}"
                    debug_file = save_problematic_json(result_json, str(json_error), error_context)
                    
                    # Get detailed error analysis
                    from utils.json_debug_utils import find_json_error_location, suggest_json_fixes
                    error_info = find_json_error_location(result_json, str(json_error))
                    if error_info:
                        logger.error(f"JSON error location: {error_info}")
                        suggestions = suggest_json_fixes(result_json, error_info)
                        logger.error(f"Suggested fixes: {suggestions}")
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying requirement completeness analysis after JSON error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"All JSON parsing attempts failed for {model_id} requirement completeness")
                        
                except Exception as e:
                    logger.warning(f"Error processing requirement completeness: {str(e)}")
                    
                    # Save debug information for any processing error
                    error_context = f"{model_id}_requirement_completeness_processing_error_attempt_{attempt+1}"
                    debug_file = save_problematic_json(result_json, str(e), error_context)
                    
                    # If not the last attempt, retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying requirement completeness analysis after processing error (attempt {attempt+1})")
                        time.sleep(self.retry_delay)
                        continue
                
            except Exception as e:
                logger.error(f"Requirement completeness API request error to {model_id} (attempt {attempt+1}): {str(e)}")
                logger.error(traceback.format_exc())
                
                # If not the last attempt, retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying requirement completeness analysis after API error (waiting {self.retry_delay} seconds)")
                    time.sleep(self.retry_delay)
                    continue
        
        # If we've exhausted all retries, return a default response
        logger.error(f"All requirement completeness analysis attempts with {model_id} failed, returning default")
        default_response["model_id"] = model_id
        return default_response
    
    def analyze_requirements_completeness(self, requirements, domain_model, selected_models=None, meta_model_id=None, model_weights=None):
        """
        Analyze requirements for completeness against the domain model with a general approach
        
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
            logger.warning("No models specified, falling back to openai")
            selected_models = ["openai"]
        
        # Step 1: Get missing requirements
        missing_requirements_result = self.detect_missing_requirements(
            requirements, domain_model, selected_models, meta_model_id, model_weights
        )
        
        # Step 2: Analyze individual requirement completeness
        requirement_completeness_result = self.analyze_requirement_completeness(
            requirements, domain_model, selected_models, meta_model_id, model_weights
        )
        
        # Step 3: Analyze requirement issues and domain model issues
        prompt = """
            You are an expert requirements analyst and software architect conducting a thorough analysis of software requirements and their domain model.
            
            ## TASK
            Analyze the provided requirements and domain model to identify:
            1. Issues within the requirements themselves (ambiguity, conflicts, inconsistencies)
            2. Issues with how the domain model represents the requirements
            
            ## DETAILED INSTRUCTIONS FOR REQUIREMENT ISSUES
            For each requirement, identify:
            - Ambiguity: Unclear or vague statements that could be interpreted multiple ways
            - Conflicts: Requirements that contradict each other
            - Inconsistencies: Requirements that use different terminology for the same concept
            - Missing context: Requirements that lack necessary details
            - Testability issues: Requirements that cannot be verified
            - Implementation concerns: Requirements that are technically infeasible
            
            ## DETAILED INSTRUCTIONS FOR DOMAIN MODEL ISSUES
            Examine the domain model for:
            - Missing classes that should exist based on requirements
            - Incorrect relationships between classes
            - Missing or incorrect attributes
            - Architectural concerns
            
            ## OUTPUT FORMAT
            Return a JSON object with this structure:
            {
                "requirement_issues": [
                    {
                        "requirement_id": "REQ-001",
                        "requirement_text": "The actual requirement text",
                        "issues": [
                            {
                                "issue_type": "Ambiguity|Conflict|Inconsistency|Missing Context|Testability|Implementation",
                                "severity": "MUST FIX|SHOULD FIX|SUGGESTION",
                                "description": "Detailed description of the issue",
                                "suggested_fix": "Specific suggestion to fix the issue",
                                "affected_model_elements": ["Class1", "Class2"]
                            }
                        ]
                    }
                ],
                "domain_model_issues": [
                    {
                        "element_type": "Class|Relationship|Attribute|Method|General",
                        "element_name": "Name of the specific element with issue",
                        "issue_type": "Missing|Incomplete|Incorrect|Inconsistent|Misaligned",
                        "severity": "MUST FIX|SHOULD FIX|SUGGESTION",
                        "description": "Detailed description of the issue",
                        "suggested_fix": "Specific suggestion to resolve the issue",
                        "affected_requirements": ["REQ-001", "REQ-002"]
                    }
                ]
            }
            
            ## REQUIREMENTS TO ANALYZE:
            """
        
        # Add domain model
        prompt += "\n\n## DOMAIN MODEL\n"
        domain_model_text = json.dumps(domain_model, indent=2)
        prompt += domain_model_text
        
        # Add requirements
        prompt += "\n\n## REQUIREMENTS\n"
        prompt += requirements
        
        # Create messages for the LLM
        messages = [{"role": "user", "content": prompt}]
        
        # Process requirement issues analysis
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
                
                # # Save raw analysis response for debugging
                # with open(f"log/raw_analysis_response_{model_id}_{attempt}.txt", "w", encoding="utf-8") as f:
                #     f.write(analysis_json)
                
                # Attempt to parse and validate the JSON
                try:
                    analysis = extract_json_from_response(analysis_json, expected_top_level_keys=["requirement_issues", "domain_model_issues"])
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
            # Use openai as fallback for backward compatibility
            logger.warning("No models specified, falling back to openai")
            selected_models = ["openai"]
        
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
                elif operation_type == "extract_context":
                    return self._extract_context_with_model(model_id, messages)
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
    
    def _save_domain_model_debug(self, model_id, attempt, messages, response_content, domain_model=None, error=None):
        """
        Save debug information for domain model creation
        
        Args:
            model_id (str): ID of the model used
            attempt (int): Attempt number
            messages (list): Messages sent to the LLM
            response_content (str): Full response from the LLM
            domain_model (dict): Parsed domain model (if successful)
            error (str): Error message (if failed)
        """
        try:
            # Create debug directory if it doesn't exist
            debug_dir = "log/domain_model_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create debug data structure
            debug_data = {
                "model_id": model_id,
                "attempt": attempt,
                "timestamp": timestamp,
                "messages_sent": messages,
                "full_response": response_content,
                "parsed_domain_model": domain_model,
                "error": error,
                "content_length": len(response_content) if response_content else 0
            }
            
            # Save to file
            filename = f"{debug_dir}/domain_model_{model_id}_attempt_{attempt}_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Domain model debug data saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save domain model debug data: {str(e)}")