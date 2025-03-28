import logging
import json
import traceback
from collections import Counter
from utils.json_utils import extract_json_from_response, validate_domain_model
from services.llm_adapters import get_adapter
from config import get_available_meta_models

logger = logging.getLogger(__name__)

class ResultsAggregator:
    """
    Aggregates results from multiple LLMs using various strategies
    """
    
    def __init__(self, meta_model_id="majority", custom_weights=None):
        """
        Initialize with the specified meta model
        
        Args:
            meta_model_id (str): ID of the meta model to use for aggregation
            custom_weights (dict): Custom weights for models when using weighted voting
        """
        self.meta_model_id = meta_model_id
        self.meta_models = get_available_meta_models()
        self.custom_weights = custom_weights or {}
        
        normalized_id = self._normalize_meta_model_id(meta_model_id)
        
        # Then check if it exists in available models
        if normalized_id not in self.meta_models:
            logger.warning(f"Unknown meta model ID: {meta_model_id} (normalized: {normalized_id}), falling back to majority vote")
            self.meta_model_id = "majority"
        else:
            self.meta_model_id = normalized_id

    def _get_model_weights(self):
        """Get model weights, using custom weights if provided or default weights otherwise"""
        # Default weights for different models
        default_weights = {
            "deepseek": 1.2,  # DeepSeek slightly higher weight for domain modeling
            "openai": 1.0,
            "claude": 1.1
        }
        
        # If custom weights are provided, use them with fallback to default weights
        if self.custom_weights:
            weights = {}
            for model_id, default_weight in default_weights.items():
                weights[model_id] = self.custom_weights.get(model_id, default_weight)
            return weights
        
        # Otherwise use default weights
        return default_weights
    
    def _get_model_weights_for_analysis(self):
        """Get model weights for analysis, using custom weights if provided or default weights otherwise"""
        # Default weights for analysis might be different
        default_weights = {
            "deepseek": 1.0,
            "openai": 1.1,   # OpenAI slightly higher weight for analysis
            "claude": 1.2    # Claude highest weight for analysis
        }
        
        # If custom weights are provided, use them with fallback to default weights
        if self.custom_weights:
            weights = {}
            for model_id, default_weight in default_weights.items():
                weights[model_id] = self.custom_weights.get(model_id, default_weight)
            return weights
        
        # Otherwise use default weights
        return default_weights

    def _normalize_meta_model_id(self, meta_model_id):
        """
        Normalize meta model ID to handle both with and without _meta suffix
        """
        # Define mappings for meta model IDs
        meta_model_mappings = {
            # Standard to standard
            "majority": "majority", 
            "weighted": "weighted",
            
            # OpenAI variants
            "openai": "openai",
            "openai_meta": "openai",
            
            # DeepSeek variants
            "deepseek": "deepseek",
            "deepseek_meta": "deepseek",
            
            # Claude variants
            "claude": "claude",
            "claude_meta": "claude"
        }
        
        # Return the normalized ID or the original if not found
        return meta_model_mappings.get(meta_model_id, "majority")
    def aggregate_extracted_requirements(self, model_results):
        """
        Aggregate requirements extracted from multiple LLMs
        
        Args:
            model_results (list): List of results from different models, each containing
                                extracted requirements
        
        Returns:
            dict: Aggregated requirements and metadata
        """
        logger.info(f"Aggregating extracted requirements using {self.meta_model_id} strategy")
        
        if len(model_results) == 0:
            logger.error("No model results to aggregate")
            return {
                "extracted_requirements": "",
                "requirements_count": 0,
                "requirements_list": [],
                "reasoning": "No model results available for aggregation"
            }
        
        if len(model_results) == 1:
            logger.info("Only one model result, no aggregation needed")
            return model_results[0]
        
        # Extract requirements lists from results
        requirements_data = []
        for result in model_results:
            if result and "extracted_requirements" in result and result["extracted_requirements"]:
                requirements_data.append({
                    "model_id": result.get("model_id", "unknown"),
                    "requirements_list": result.get("requirements_list", [])
                })
        
        if len(requirements_data) == 0:
            logger.error("No valid requirements found in results")
            # Log more details about the model results to help diagnose the issue
            for i, result in enumerate(model_results):
                logger.debug(f"Model result {i+1} keys: {result.keys() if result else 'None'}")
                if result and "error" in result:
                    logger.debug(f"Model result {i+1} error: {result['error']}")
            
            return {
                "extracted_requirements": "",
                "requirements_count": 0,
                "requirements_list": [],
                "reasoning": "No valid requirements found in results"
            }
        
        if len(requirements_data) == 1:
            logger.info("Only one valid requirements result, no aggregation needed")
            for result in model_results:
                if "extracted_requirements" in result and result["extracted_requirements"]:
                    return result
        
        # Use the appropriate aggregation strategy
        normalized_meta_id = self._normalize_meta_model_id(self.meta_model_id)
    
        if normalized_meta_id == "majority":
            return self._majority_vote_requirements(requirements_data, model_results)
        elif normalized_meta_id == "weighted":
            return self._weighted_vote_requirements(requirements_data, model_results)
        elif normalized_meta_id in ["openai", "deepseek", "claude"]:
            return self._llm_based_requirements_aggregation(requirements_data, model_results, normalized_meta_id)
        else:
            logger.warning(f"Unknown aggregation strategy: {self.meta_model_id}, falling back to majority vote")
            return self._majority_vote_requirements(requirements_data, model_results)

    def _majority_vote_requirements(self, requirements_data, original_results):
        """
        Aggregate requirements using majority voting
        
        This method:
        1. Normalizes requirement text to handle minor differences
        2. Counts occurrences of similar requirements across models
        3. Includes requirements that appear in majority of models
        """
        logger.info("Using majority vote to aggregate requirements")
        
        # Process all requirements and create normalized versions for comparison
        all_requirements = []
        for data in requirements_data:
            for req in data["requirements_list"]:
                # Normalize the requirement for better comparison
                # Extract the ID and text
                parts = req.split(':', 1)
                req_id = parts[0].strip() if len(parts) > 1 else ""
                req_text = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                
                # Normalize the text (remove extra spaces, lowercase)
                normalized_text = ' '.join(req_text.lower().split())
                
                all_requirements.append({
                    "model_id": data["model_id"],
                    "requirement": req,
                    "req_id": req_id,
                    "req_text": req_text,
                    "normalized_text": normalized_text
                })
        
        # Group similar requirements
        requirement_groups = {}
        for req in all_requirements:
            normalized = req["normalized_text"]
            
            # Check if this requirement is similar to any existing group
            matched = False
            for group_key in list(requirement_groups.keys()):
                # Simple similarity check - can be improved with better algorithms
                if self._text_similarity(normalized, group_key) > 0.8:
                    requirement_groups[group_key].append(req)
                    matched = True
                    break
            
            # If no similar requirement found, create a new group
            if not matched:
                requirement_groups[normalized] = [req]
        
        # Calculate majority threshold
        majority_threshold = len(requirements_data) // 2 + 1  # Simple majority
        
        # Select requirements that appear in majority of models
        selected_requirements = []
        
        for group_key, reqs in requirement_groups.items():
            # Count distinct models
            models = set(req["model_id"] for req in reqs)
            
            if len(models) >= majority_threshold:
                # Find the most detailed requirement in this group
                best_req = max(reqs, key=lambda x: len(x["requirement"]))
                selected_requirements.append(best_req["requirement"])
        
        # Sort requirements by ID if possible
        selected_requirements.sort(key=lambda x: x.split(':')[0] if ':' in x else x)
        
        # Combine the requirements into a single string
        extracted_requirements = '\n'.join(selected_requirements)
        
        # Create the result with reasoning
        result = {
            "extracted_requirements": extracted_requirements,
            "requirements_count": len(selected_requirements),
            "requirements_list": selected_requirements,
            "reasoning": f"These requirements were aggregated from {len(requirements_data)} models " \
                        f"using majority voting. Selected {len(selected_requirements)} requirements that appeared in at least " \
                        f"{majority_threshold} models.",
            "aggregation_info": {
                "strategy": "majority_vote",
                "model_count": len(requirements_data),
                "majority_threshold": majority_threshold,
                "contributing_models": [data["model_id"] for data in requirements_data]
            }
        }
        
        return result

    def _weighted_vote_requirements(self, requirements_data, original_results):
        """
        Aggregate requirements using weighted voting
        
        Similar to majority voting but assigns weights to different models.
        """
        logger.info("Using weighted vote to aggregate requirements")
        
        # Get weights for different models
        model_weights = self._get_model_weights()
        
        # Default weight for any model not in the weights dictionary
        default_weight = 1.0
        
        # Total weight of all models for calculating threshold
        total_weight = 0
        
        # Process all requirements and create normalized versions for comparison
        all_requirements = []
        for data in requirements_data:
            model_id = data["model_id"]
            weight = model_weights.get(model_id, default_weight)
            total_weight += weight
            
            for req in data["requirements_list"]:
                # Normalize the requirement for better comparison
                # Extract the ID and text
                parts = req.split(':', 1)
                req_id = parts[0].strip() if len(parts) > 1 else ""
                req_text = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                
                # Normalize the text (remove extra spaces, lowercase)
                normalized_text = ' '.join(req_text.lower().split())
                
                all_requirements.append({
                    "model_id": model_id,
                    "weight": weight,
                    "requirement": req,
                    "req_id": req_id,
                    "req_text": req_text,
                    "normalized_text": normalized_text
                })
        
        # Group similar requirements
        requirement_groups = {}
        for req in all_requirements:
            normalized = req["normalized_text"]
            
            # Check if this requirement is similar to any existing group
            matched = False
            for group_key in list(requirement_groups.keys()):
                # Simple similarity check - can be improved with better algorithms
                if self._text_similarity(normalized, group_key) > 0.8:
                    requirement_groups[group_key].append(req)
                    matched = True
                    break
            
            # If no similar requirement found, create a new group
            if not matched:
                requirement_groups[normalized] = [req]
        
        # Calculate weighted threshold (50% of total weight)
        weighted_threshold = total_weight / 2
        
        # Select requirements that exceed the weighted threshold
        selected_requirements = []
        
        for group_key, reqs in requirement_groups.items():
            # Calculate total weight for this requirement
            req_weight = sum(req["weight"] for req in reqs)
            
            if req_weight >= weighted_threshold:
                # Find the most detailed requirement in this group
                best_req = max(reqs, key=lambda x: len(x["requirement"]))
                selected_requirements.append(best_req["requirement"])
        
        # Sort requirements by ID if possible
        selected_requirements.sort(key=lambda x: x.split(':')[0] if ':' in x else x)
        
        # Combine the requirements into a single string
        extracted_requirements = '\n'.join(selected_requirements)
        
        # Create the result with reasoning
        result = {
            "extracted_requirements": extracted_requirements,
            "requirements_count": len(selected_requirements),
            "requirements_list": selected_requirements,
            "reasoning": f"These requirements were aggregated from {len(requirements_data)} models " \
                        f"using weighted voting. Selected {len(selected_requirements)} requirements that exceeded the weighted threshold of {weighted_threshold:.1f}.",
            "aggregation_info": {
                "strategy": "weighted_vote",
                "model_count": len(requirements_data),
                "weighted_threshold": weighted_threshold,
                "contributing_models": [f"{data['model_id']} (weight: {model_weights.get(data['model_id'], default_weight)})" 
                                    for data in requirements_data]
            }
        }
        
        return result

    def _llm_based_requirements_aggregation(self, requirements_data, original_results, meta_model_id):
        """
        Use an LLM to aggregate requirements from multiple sources
        
        Args:
            requirements_data (list): List of requirements from different LLMs
            original_results (list): Original results from LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
        
        Returns:
            dict: Aggregated requirements and reasoning
        """
        logger.info(f"Using {meta_model_id} LLM to aggregate requirements")
        
        try:
            # Prepare the prompt for the LLM
            prompt = """You are an expert in software requirements engineering. You need to analyze multiple sets of extracted requirements from different LLMs and create a consolidated set that:

    1. Includes all important requirements without duplication
    2. Uses consistent ID numbering
    3. Maintains clear and concise language
    4. Resolves any conflicts between different extractions
    5. Ensures all requirements are well-formatted

    OUTPUT FORMAT:
    Output ONLY the consolidated requirements in this format, with ONE requirement per line:

    REQ-ID: Requirement description

    For example:
    REQ-001: The system shall allow users to register an account with email and password.
    REQ-002: The system shall validate all user inputs for security purposes.

    DO NOT include any other text, explanations, or commentary in your response. 
    ONLY return the numbered requirements list.

    SOURCE REQUIREMENTS:
    """
            
            # Add each source requirements list
            for i, data in enumerate(requirements_data):
                requirements_text = '\n'.join(data["requirements_list"])
                prompt += f"\n\nMODEL {i+1} ({data['model_id']}):\n{requirements_text}\n"
            
            # Create the messages for the LLM
            messages = [
                {"role": "system", "content": "You are an expert in software requirements engineering specializing in requirements consolidation."},
                {"role": "user", "content": prompt}
            ]
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            response = adapter.generate_response(messages)
            
            # Extract the response content
            extracted_text = response["content"]
            if not extracted_text:
                raise ValueError("Empty response content")
            
            # Process the extracted requirements
            requirements_list = []
            for line in extracted_text.strip().split('\n'):
                line = line.strip()
                if line and (':' in line):
                    requirements_list.append(line)
            
            logger.info(f"Consolidated {len(requirements_list)} requirements using {meta_model_id}")
            
            # Combine the requirements into a single string
            extracted_requirements = '\n'.join(requirements_list)
            
            # Create the result
            result = {
                "extracted_requirements": extracted_requirements,
                "requirements_count": len(requirements_list),
                "requirements_list": requirements_list,
                "reasoning": f"These requirements were consolidated from {len(requirements_data)} models using {meta_model_id} as a meta-model.",
                "aggregation_info": {
                    "strategy": f"llm_based_{meta_model_id}",
                    "model_count": len(requirements_data),
                    "contributing_models": [data["model_id"] for data in requirements_data],
                    "meta_model_id": meta_model_id
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM-based requirements aggregation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to majority vote
            logger.info("Falling back to majority vote due to LLM aggregation error")
            return self._majority_vote_requirements(requirements_data, original_results)

    def _text_similarity(self, text1, text2):
        """
        Calculate similarity between two texts (simple version)
        Returns a value between 0 (completely different) and 1 (identical)
        """
        # Simple Jaccard similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Handle empty sets
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def aggregate_domain_models(self, model_results):
        """
        Aggregate domain models from multiple LLMs
        
        Args:
            model_results (list): List of results from different models, each containing
                                 a domain model and model metadata
        
        Returns:
            dict: Aggregated domain model and reasoning
        """
        logger.info(f"Aggregating domain models using {self.meta_model_id} strategy")
        
        if len(model_results) == 0:
            logger.error("No model results to aggregate")
            return {
                "domain_model": {
                    "classes": [],
                    "relationships": [],
                    "plantuml": "@startuml\n@enduml"
                },
                "reasoning": "No model results available for aggregation"
            }
        
        if len(model_results) == 1:
            logger.info("Only one model result, no aggregation needed")
            return model_results[0]
        
        # Extract domain models from results
        domain_models = []
        for result in model_results:
            if "domain_model" in result and result["domain_model"]:
                domain_models.append({
                    "model_id": result.get("model_id", "unknown"),
                    "domain_model": result["domain_model"]
                })
        
        if len(domain_models) == 0:
            logger.error("No valid domain models found in results")
            return {
                "domain_model": {
                    "classes": [],
                    "relationships": [],
                    "plantuml": "@startuml\n@enduml"
                },
                "reasoning": "No valid domain models found in results"
            }
        
        if len(domain_models) == 1:
            logger.info("Only one valid domain model, no aggregation needed")
            for result in model_results:
                if "domain_model" in result and result["domain_model"]:
                    return result
        
        # Use the appropriate aggregation strategy
        if self.meta_model_id == "majority":
            return self._majority_vote_domain_models(domain_models, model_results)
        elif self.meta_model_id == "weighted":
            return self._weighted_vote_domain_models(domain_models, model_results)
        elif self.meta_model_id in ["openai", "deepseek", "claude"]:
            return self._llm_based_aggregation(domain_models, model_results, self.meta_model_id)
        else:
            logger.warning(f"Unknown aggregation strategy: {self.meta_model_id}, falling back to majority vote")
            return self._majority_vote_domain_models(domain_models, model_results)
    
    def aggregate_analysis_results(self, model_results):
        """
        Aggregate analysis results from multiple LLMs
        
        Args:
            model_results (list): List of results from different models, each containing
                                 analysis results and model metadata
        
        Returns:
            dict: Aggregated analysis results and reasoning
        """
        logger.info(f"Aggregating analysis results using {self.meta_model_id} strategy")
        
        if len(model_results) == 0:
            logger.error("No analysis results to aggregate")
            return {
                "analysis": {
                    "requirement_issues": [],
                    "missing_requirements": [],
                    "domain_model_issues": [],
                    "requirement_completeness": []
                },
                "reasoning": "No analysis results available for aggregation"
            }
        
        if len(model_results) == 1:
            logger.info("Only one model result, no aggregation needed")
            return model_results[0]
        
        # Extract analysis results from models
        analysis_results = []
        for result in model_results:
            if "analysis" in result and result["analysis"]:
                analysis_results.append({
                    "model_id": result.get("model_id", "unknown"),
                    "analysis": result["analysis"]
                })
        
        if len(analysis_results) == 0:
            logger.error("No valid analysis results found in results")
            return {
                "analysis": {
                    "requirement_issues": [],
                    "missing_requirements": [],
                    "domain_model_issues": [],
                    "requirement_completeness": []
                },
                "reasoning": "No valid analysis results found"
            }
        
        if len(analysis_results) == 1:
            logger.info("Only one valid analysis result, no aggregation needed")
            for result in model_results:
                if "analysis" in result and result["analysis"]:
                    return result
        
        # Use the appropriate aggregation strategy
        if self.meta_model_id == "majority":
            return self._majority_vote_analysis(analysis_results, model_results)
        elif self.meta_model_id == "weighted":
            return self._weighted_vote_analysis(analysis_results, model_results)
        elif self.meta_model_id in ["openai", "deepseek", "claude"]:
            return self._llm_based_analysis_aggregation(analysis_results, model_results, self.meta_model_id)
        else:
            logger.warning(f"Unknown aggregation strategy: {self.meta_model_id}, falling back to majority vote")
            return self._majority_vote_analysis(analysis_results, model_results)
    
    def _majority_vote_domain_models(self, domain_models, original_results):
        """
        Aggregate domain models using majority voting
        
        This method:
        1. Collects all classes and relationships across models
        2. Counts occurrences of classes and relationships
        3. Includes elements that appear in majority of models
        4. Aggregates plantuml diagrams using the most complete one
        """
        logger.info("Using majority vote to aggregate domain models")
        
        # Extract all classes and relationships
        all_classes = []
        all_relationships = []
        
        for dm in domain_models:
            model_classes = dm["domain_model"].get("classes", [])
            model_relationships = dm["domain_model"].get("relationships", [])
            
            # Add model ID to track source
            for cls in model_classes:
                all_classes.append({
                    "model_id": dm["model_id"],
                    "class": cls
                })
            
            for rel in model_relationships:
                all_relationships.append({
                    "model_id": dm["model_id"],
                    "relationship": rel
                })
        
        # Group classes by name to find consensus
        class_groups = {}
        for cls_entry in all_classes:
            cls = cls_entry["class"]
            cls_name = cls["name"]
            
            if cls_name not in class_groups:
                class_groups[cls_name] = []
            
            class_groups[cls_name].append(cls_entry)
        
        # Select classes that appear in majority of models
        majority_threshold = len(domain_models) // 2 + 1  # Simple majority
        
        # Choose representative classes
        selected_classes = []
        for cls_name, cls_entries in class_groups.items():
            if len(cls_entries) >= majority_threshold:
                # Select the most detailed class definition (highest attribute count)
                most_detailed = max(
                    cls_entries, 
                    key=lambda x: len(x["class"].get("attributes", [])) + len(x["class"].get("methods", []))
                )
                selected_classes.append(most_detailed["class"])
        
        # Group relationships by source-target-type to find consensus
        rel_groups = {}
        for rel_entry in all_relationships:
            rel = rel_entry["relationship"]
            rel_key = f"{rel['source']}-{rel['target']}-{rel['type']}"
            
            if rel_key not in rel_groups:
                rel_groups[rel_key] = []
            
            rel_groups[rel_key].append(rel_entry)
        
        # Select relationships that appear in majority of models
        selected_relationships = []
        for rel_key, rel_entries in rel_groups.items():
            if len(rel_entries) >= majority_threshold:
                # Select the most detailed relationship definition
                most_detailed = max(
                    rel_entries, 
                    key=lambda x: len(x["relationship"].get("description", ""))
                )
                selected_relationships.append(most_detailed["relationship"])
        
        # Choose the most complete PlantUML diagram
        plantuml_code = "@startuml\n@enduml"
        max_uml_length = 0
        
        for dm in domain_models:
            uml_code = dm["domain_model"].get("plantuml", "")
            if len(uml_code) > max_uml_length:
                plantuml_code = uml_code
                max_uml_length = len(uml_code)
        
        # If we have classes but no proper PlantUML, generate a basic one
        if max_uml_length <= 20 and selected_classes:
            plantuml_code = "@startuml\n"
            for cls in selected_classes:
                plantuml_code += f"class {cls['name']} {{\n"
                for attr in cls.get("attributes", []):
                    plantuml_code += f"  +{attr['name']}: {attr['type']}\n"
                for method in cls.get("methods", []):
                    plantuml_code += f"  +{method['name']}(): {method.get('returnType', 'void')}\n"
                plantuml_code += "}\n\n"
            
            for rel in selected_relationships:
                rel_type = rel["type"].lower()
                if rel_type == "inheritance":
                    plantuml_code += f"{rel['target']} <|-- {rel['source']}\n"
                elif rel_type == "composition":
                    plantuml_code += f"{rel['source']} *-- {rel['target']}\n"
                elif rel_type == "aggregation":
                    plantuml_code += f"{rel['source']} o-- {rel['target']}\n"
                else:  # association or other
                    plantuml_code += f"{rel['source']} -- {rel['target']}\n"
            
            plantuml_code += "@enduml"
        
        # Create aggregated domain model
        aggregated_model = {
            "classes": selected_classes,
            "relationships": selected_relationships,
            "plantuml": plantuml_code
        }
        
        # Create the result with reasoning
        result = {
            "domain_model": aggregated_model,
            "reasoning": f"This domain model was created by aggregating {len(domain_models)} models " \
                         f"using majority voting. Selected {len(selected_classes)} classes and " \
                         f"{len(selected_relationships)} relationships that appeared in at least " \
                         f"{majority_threshold} models.",
            "aggregation_info": {
                "strategy": "majority_vote",
                "model_count": len(domain_models),
                "majority_threshold": majority_threshold,
                "contributing_models": [dm["model_id"] for dm in domain_models]
            }
        }
        
        return result
    
    def _weighted_vote_domain_models(self, domain_models, original_results):
        """
        Aggregate domain models using weighted voting
        
        Similar to majority voting but assigns weights to different models.
        """
        logger.info("Using weighted vote to aggregate domain models")
        
        # Get weights for different models
        model_weights = self._get_model_weights()
        
        # Default weight for any model not in the weights dictionary
        default_weight = 1.0
        
        # Extract all classes and relationships
        all_classes = []
        all_relationships = []
        
        # Total weight of all models for calculating threshold
        total_weight = 0
        
        for dm in domain_models:
            model_id = dm["model_id"]
            model_classes = dm["domain_model"].get("classes", [])
            model_relationships = dm["domain_model"].get("relationships", [])
            
            # Get weight for this model
            weight = model_weights.get(model_id, default_weight)
            total_weight += weight
            
            # Add model ID and weight to track source
            for cls in model_classes:
                all_classes.append({
                    "model_id": model_id,
                    "class": cls,
                    "weight": weight
                })
            
            for rel in model_relationships:
                all_relationships.append({
                    "model_id": model_id,
                    "relationship": rel,
                    "weight": weight
                })
        
        # Group classes by name to find consensus
        class_groups = {}
        for cls_entry in all_classes:
            cls = cls_entry["class"]
            cls_name = cls["name"]
            
            if cls_name not in class_groups:
                class_groups[cls_name] = []
            
            class_groups[cls_name].append(cls_entry)
        
        # Calculate weighted threshold (50% of total weight)
        weighted_threshold = total_weight / 2
        
        # Choose representative classes
        selected_classes = []
        for cls_name, cls_entries in class_groups.items():
            # Calculate total weight for this class
            class_weight = sum(entry["weight"] for entry in cls_entries)
            
            if class_weight >= weighted_threshold:
                # Select the most detailed class definition (highest attribute count)
                most_detailed = max(
                    cls_entries, 
                    key=lambda x: len(x["class"].get("attributes", [])) + len(x["class"].get("methods", []))
                )
                selected_classes.append(most_detailed["class"])
        
        # Group relationships by source-target-type to find consensus
        rel_groups = {}
        for rel_entry in all_relationships:
            rel = rel_entry["relationship"]
            rel_key = f"{rel['source']}-{rel['target']}-{rel['type']}"
            
            if rel_key not in rel_groups:
                rel_groups[rel_key] = []
            
            rel_groups[rel_key].append(rel_entry)
        
        # Select relationships that exceed weighted threshold
        selected_relationships = []
        for rel_key, rel_entries in rel_groups.items():
            # Calculate total weight for this relationship
            rel_weight = sum(entry["weight"] for entry in rel_entries)
            
            if rel_weight >= weighted_threshold:
                # Select the most detailed relationship definition
                most_detailed = max(
                    rel_entries, 
                    key=lambda x: len(x["relationship"].get("description", ""))
                )
                selected_relationships.append(most_detailed["relationship"])
        
        # Choose the most complete PlantUML diagram, with preference to higher-weighted models
        plantuml_code = "@startuml\n@enduml"
        max_uml_score = 0
        
        for dm in domain_models:
            uml_code = dm["domain_model"].get("plantuml", "")
            model_id = dm["model_id"]
            weight = model_weights.get(model_id, default_weight)
            
            # Score is a combination of length and model weight
            uml_score = len(uml_code) * weight
            
            if uml_score > max_uml_score:
                plantuml_code = uml_code
                max_uml_score = uml_score
        
        # If we have classes but no proper PlantUML, generate a basic one
        if max_uml_score <= 20 * max(model_weights.values()) and selected_classes:
            plantuml_code = "@startuml\n"
            for cls in selected_classes:
                plantuml_code += f"class {cls['name']} {{\n"
                for attr in cls.get("attributes", []):
                    plantuml_code += f"  +{attr['name']}: {attr['type']}\n"
                for method in cls.get("methods", []):
                    plantuml_code += f"  +{method['name']}(): {method.get('returnType', 'void')}\n"
                plantuml_code += "}\n\n"
            
            for rel in selected_relationships:
                rel_type = rel["type"].lower()
                if rel_type == "inheritance":
                    plantuml_code += f"{rel['target']} <|-- {rel['source']}\n"
                elif rel_type == "composition":
                    plantuml_code += f"{rel['source']} *-- {rel['target']}\n"
                elif rel_type == "aggregation":
                    plantuml_code += f"{rel['source']} o-- {rel['target']}\n"
                else:  # association or other
                    plantuml_code += f"{rel['source']} -- {rel['target']}\n"
            
            plantuml_code += "@enduml"
        
        # Create aggregated domain model
        aggregated_model = {
            "classes": selected_classes,
            "relationships": selected_relationships,
            "plantuml": plantuml_code
        }
        
        # Create the result with reasoning
        result = {
            "domain_model": aggregated_model,
            "reasoning": f"This domain model was created by aggregating {len(domain_models)} models " \
                        f"using weighted voting. Selected {len(selected_classes)} classes and " \
                        f"{len(selected_relationships)} relationships that exceeded the weighted threshold.",
            "aggregation_info": {
                "strategy": "weighted_vote",
                "model_count": len(domain_models),
                "weighted_threshold": weighted_threshold,
                "contributing_models": [f"{dm['model_id']} (weight: {model_weights.get(dm['model_id'], default_weight)})" 
                                    for dm in domain_models]
            }
        }
        
        return result
    
    def _majority_vote_analysis(self, analysis_results, original_results):
        """
        Aggregate analysis results using majority voting
        
        This method combines analysis results with a focus on:
        1. Requirement issues that appear in multiple models
        2. Missing requirements identified by multiple models
        3. Domain model issues identified by multiple models
        4. Requirement completeness assessments
        """
        logger.info("Using majority vote to aggregate analysis results")
        
        # Extract all requirement issues
        all_req_issues = []
        all_missing_reqs = []
        all_model_issues = []
        all_completeness = []
        
        for ar in analysis_results:
            model_id = ar["model_id"]
            analysis = ar["analysis"]
            
            # Process requirement issues
            for req_issue in analysis.get("requirement_issues", []):
                req_id = req_issue.get("requirement_id", "")
                req_text = req_issue.get("requirement_text", "")
                
                for issue in req_issue.get("issues", []):
                    all_req_issues.append({
                        "model_id": model_id,
                        "requirement_id": req_id,
                        "requirement_text": req_text,
                        "issue_type": issue.get("issue_type", ""),
                        "severity": issue.get("severity", ""),
                        "description": issue.get("description", ""),
                        "suggested_fix": issue.get("suggested_fix", ""),
                        "affected_model_elements": issue.get("affected_model_elements", [])
                    })
            
            # Process missing requirements
            for missing_req in analysis.get("missing_requirements", []):
                all_missing_reqs.append({
                    "model_id": model_id,
                    "id": missing_req.get("id", ""),
                    "description": missing_req.get("description", ""),
                    "category": missing_req.get("category", ""),
                    "severity": missing_req.get("severity", ""),
                    "suggested_requirement": missing_req.get("suggested_requirement", ""),
                    "affected_model_elements": missing_req.get("affected_model_elements", []),
                    "rationale": missing_req.get("rationale", "")
                })
            
            # Process domain model issues
            for model_issue in analysis.get("domain_model_issues", []):
                all_model_issues.append({
                    "model_id": model_id,
                    "element_type": model_issue.get("element_type", ""),
                    "element_name": model_issue.get("element_name", ""),
                    "issue_type": model_issue.get("issue_type", ""),
                    "severity": model_issue.get("severity", ""),
                    "description": model_issue.get("description", ""),
                    "suggested_fix": model_issue.get("suggested_fix", ""),
                    "affected_requirements": model_issue.get("affected_requirements", [])
                })
            
            # Process requirement completeness
            for completeness in analysis.get("requirement_completeness", []):
                all_completeness.append({
                    "model_id": model_id,
                    "requirement_id": completeness.get("requirement_id", ""),
                    "requirement_text": completeness.get("requirement_text", ""),
                    "completeness_score": completeness.get("completeness_score", 0),
                    "missing_elements": completeness.get("missing_elements", []),
                    "suggested_improvement": completeness.get("suggested_improvement", ""),
                    "rationale": completeness.get("rationale", "")
                })
        
        # Group requirement issues by requirement ID and issue type
        req_issue_groups = {}
        for issue in all_req_issues:
            key = f"{issue['requirement_id']}:{issue['issue_type']}"
            if key not in req_issue_groups:
                req_issue_groups[key] = []
            req_issue_groups[key].append(issue)
        
        # Group missing requirements by description (fuzzy matching would be better)
        missing_req_groups = {}
        for missing_req in all_missing_reqs:
            # Use ID as key if available, otherwise use description
            key = missing_req.get("id", "") or missing_req.get("description", "")
            if key not in missing_req_groups:
                missing_req_groups[key] = []
            missing_req_groups[key].append(missing_req)
        
        # Group model issues by element name and issue type
        model_issue_groups = {}
        for model_issue in all_model_issues:
            key = f"{model_issue['element_name']}:{model_issue['issue_type']}"
            if key not in model_issue_groups:
                model_issue_groups[key] = []
            model_issue_groups[key].append(model_issue)
        
        # Group completeness by requirement ID
        completeness_groups = {}
        for completeness in all_completeness:
            key = completeness.get("requirement_id", "")
            if key not in completeness_groups:
                completeness_groups[key] = []
            completeness_groups[key].append(completeness)
        
        # Calculate majority threshold
        majority_threshold = len(analysis_results) // 2 + 1  # Simple majority
        
        # Select issues that appear in majority of models
        selected_req_issues = {}
        for key, issues in req_issue_groups.items():
            if len(issues) >= majority_threshold:
                # Select the most detailed issue
                most_detailed = max(issues, key=lambda x: len(x.get("description", "")))
                req_id = most_detailed["requirement_id"]
                
                if req_id not in selected_req_issues:
                    selected_req_issues[req_id] = {
                        "requirement_id": req_id,
                        "requirement_text": most_detailed["requirement_text"],
                        "issues": []
                    }
                
                selected_req_issues[req_id]["issues"].append({
                    "issue_type": most_detailed["issue_type"],
                    "severity": most_detailed["severity"],
                    "description": most_detailed["description"],
                    "suggested_fix": most_detailed["suggested_fix"],
                    "affected_model_elements": most_detailed["affected_model_elements"]
                })
        
        # Select missing requirements that appear in majority of models
        selected_missing_reqs = []
        for key, missing_reqs in missing_req_groups.items():
            if len(missing_reqs) >= majority_threshold:
                # Select the most detailed missing requirement
                most_detailed = max(missing_reqs, key=lambda x: len(x.get("description", "")) + len(x.get("rationale", "")))
                selected_missing_reqs.append({
                    "id": most_detailed.get("id", "") or f"MR{len(selected_missing_reqs) + 1}",
                    "description": most_detailed["description"],
                    "category": most_detailed["category"],
                    "severity": most_detailed["severity"],
                    "suggested_requirement": most_detailed["suggested_requirement"],
                    "affected_model_elements": most_detailed["affected_model_elements"],
                    "rationale": most_detailed["rationale"]
                })
        
        # Select model issues that appear in majority of models
        selected_model_issues = []
        for key, model_issues in model_issue_groups.items():
            if len(model_issues) >= majority_threshold:
                # Select the most detailed model issue
                most_detailed = max(model_issues, key=lambda x: len(x.get("description", "")))
                selected_model_issues.append({
                    "element_type": most_detailed["element_type"],
                    "element_name": most_detailed["element_name"],
                    "issue_type": most_detailed["issue_type"],
                    "severity": most_detailed["severity"],
                    "description": most_detailed["description"],
                    "suggested_fix": most_detailed["suggested_fix"],
                    "affected_requirements": most_detailed["affected_requirements"]
                })
        
        # Select requirement completeness entries that appear in majority of models
        selected_completeness = []
        for key, completeness_entries in completeness_groups.items():
            if key and len(completeness_entries) >= majority_threshold:
                # Average the completeness scores
                avg_score = sum(entry["completeness_score"] for entry in completeness_entries) / len(completeness_entries)
                
                # Select the most detailed completeness entry
                most_detailed = max(completeness_entries, key=lambda x: len(x.get("suggested_improvement", "")))
                
                # Combine missing elements from all entries
                all_missing_elements = set()
                for entry in completeness_entries:
                    all_missing_elements.update(entry.get("missing_elements", []))
                
                selected_completeness.append({
                    "requirement_id": most_detailed["requirement_id"],
                    "requirement_text": most_detailed["requirement_text"],
                    "completeness_score": round(avg_score, 1),
                    "missing_elements": list(all_missing_elements),
                    "suggested_improvement": most_detailed["suggested_improvement"],
                    "rationale": most_detailed["rationale"]
                })
        
        # Create aggregated analysis
        aggregated_analysis = {
            "requirement_issues": list(selected_req_issues.values()),
            "missing_requirements": selected_missing_reqs,
            "domain_model_issues": selected_model_issues,
            "requirement_completeness": selected_completeness
        }
        
        # Create the result with reasoning
        result = {
            "analysis": aggregated_analysis,
            "reasoning": f"This analysis was created by aggregating {len(analysis_results)} analyses " \
                         f"using majority voting. Selected issues, missing requirements, and model issues " \
                         f"that appeared in at least {majority_threshold} models.",
            "aggregation_info": {
                "strategy": "majority_vote",
                "model_count": len(analysis_results),
                "majority_threshold": majority_threshold,
                "contributing_models": [ar["model_id"] for ar in analysis_results]
            }
        }
        
        return result
    
    def _weighted_vote_analysis(self, analysis_results, original_results):
        """
        Aggregate analysis results using weighted voting
        """
        logger.info("Using weighted vote to aggregate analysis results")
        
        # Get weights for different models
        model_weights = self._get_model_weights_for_analysis()
        
        # Default weight for any model not in the weights dictionary
        default_weight = 1.0
        
        # Extract all requirement issues
        all_req_issues = []
        all_missing_reqs = []
        all_model_issues = []
        all_completeness = []
        
        # Total weight of all models for calculating threshold
        total_weight = 0
        
        for ar in analysis_results:
            model_id = ar["model_id"]
            analysis = ar["analysis"]
            
            # Get weight for this model
            weight = model_weights.get(model_id, default_weight)
            total_weight += weight
            
            # Process requirement issues
            for req_issue in analysis.get("requirement_issues", []):
                req_id = req_issue.get("requirement_id", "")
                req_text = req_issue.get("requirement_text", "")
                
                for issue in req_issue.get("issues", []):
                    all_req_issues.append({
                        "model_id": model_id,
                        "weight": weight,
                        "requirement_id": req_id,
                        "requirement_text": req_text,
                        "issue_type": issue.get("issue_type", ""),
                        "severity": issue.get("severity", ""),
                        "description": issue.get("description", ""),
                        "suggested_fix": issue.get("suggested_fix", ""),
                        "affected_model_elements": issue.get("affected_model_elements", [])
                    })
            
            # Process missing requirements
            for missing_req in analysis.get("missing_requirements", []):
                all_missing_reqs.append({
                    "model_id": model_id,
                    "weight": weight,
                    "id": missing_req.get("id", ""),
                    "description": missing_req.get("description", ""),
                    "category": missing_req.get("category", ""),
                    "severity": missing_req.get("severity", ""),
                    "suggested_requirement": missing_req.get("suggested_requirement", ""),
                    "affected_model_elements": missing_req.get("affected_model_elements", []),
                    "rationale": missing_req.get("rationale", "")
                })
            
            # Process domain model issues
            for model_issue in analysis.get("domain_model_issues", []):
                all_model_issues.append({
                    "model_id": model_id,
                    "weight": weight,
                    "element_type": model_issue.get("element_type", ""),
                    "element_name": model_issue.get("element_name", ""),
                    "issue_type": model_issue.get("issue_type", ""),
                    "severity": model_issue.get("severity", ""),
                    "description": model_issue.get("description", ""),
                    "suggested_fix": model_issue.get("suggested_fix", ""),
                    "affected_requirements": model_issue.get("affected_requirements", [])
                })
            
            # Process requirement completeness
            for completeness in analysis.get("requirement_completeness", []):
                all_completeness.append({
                    "model_id": model_id,
                    "weight": weight,
                    "requirement_id": completeness.get("requirement_id", ""),
                    "requirement_text": completeness.get("requirement_text", ""),
                    "completeness_score": completeness.get("completeness_score", 0),
                    "missing_elements": completeness.get("missing_elements", []),
                    "suggested_improvement": completeness.get("suggested_improvement", ""),
                    "rationale": completeness.get("rationale", "")
                })
        
        # Group requirement issues by requirement ID and issue type
        req_issue_groups = {}
        for issue in all_req_issues:
            key = f"{issue['requirement_id']}:{issue['issue_type']}"
            if key not in req_issue_groups:
                req_issue_groups[key] = []
            req_issue_groups[key].append(issue)
        
        # Group missing requirements by description (fuzzy matching would be better)
        missing_req_groups = {}
        for missing_req in all_missing_reqs:
            # Use ID as key if available, otherwise use description
            key = missing_req.get("id", "") or missing_req.get("description", "")
            if key not in missing_req_groups:
                missing_req_groups[key] = []
            missing_req_groups[key].append(missing_req)
        
        # Group model issues by element name and issue type
        model_issue_groups = {}
        for model_issue in all_model_issues:
            key = f"{model_issue['element_name']}:{model_issue['issue_type']}"
            if key not in model_issue_groups:
                model_issue_groups[key] = []
            model_issue_groups[key].append(model_issue)
        
        # Group completeness by requirement ID
        completeness_groups = {}
        for completeness in all_completeness:
            key = completeness.get("requirement_id", "")
            if key not in completeness_groups:
                completeness_groups[key] = []
            completeness_groups[key].append(completeness)
        
        # Calculate weighted threshold (50% of total weight)
        weighted_threshold = total_weight / 2
        
        # Select issues that exceed the weighted threshold
        selected_req_issues = {}
        for key, issues in req_issue_groups.items():
            # Calculate total weight for this issue
            issue_weight = sum(issue["weight"] for issue in issues)
            
            if issue_weight >= weighted_threshold:
                # Select the most detailed issue
                most_detailed = max(issues, key=lambda x: len(x.get("description", "")))
                req_id = most_detailed["requirement_id"]
                
                if req_id not in selected_req_issues:
                    selected_req_issues[req_id] = {
                        "requirement_id": req_id,
                        "requirement_text": most_detailed["requirement_text"],
                        "issues": []
                    }
                
                selected_req_issues[req_id]["issues"].append({
                    "issue_type": most_detailed["issue_type"],
                    "severity": most_detailed["severity"],
                    "description": most_detailed["description"],
                    "suggested_fix": most_detailed["suggested_fix"],
                    "affected_model_elements": most_detailed["affected_model_elements"]
                })
        
        # Select missing requirements that exceed the weighted threshold
        selected_missing_reqs = []
        for key, missing_reqs in missing_req_groups.items():
            # Calculate total weight for this missing requirement
            req_weight = sum(req["weight"] for req in missing_reqs)
            
            if req_weight >= weighted_threshold:
                # Select the most detailed missing requirement
                most_detailed = max(missing_reqs, key=lambda x: len(x.get("description", "")) + len(x.get("rationale", "")))
                selected_missing_reqs.append({
                    "id": most_detailed.get("id", "") or f"MR{len(selected_missing_reqs) + 1}",
                    "description": most_detailed["description"],
                    "category": most_detailed["category"],
                    "severity": most_detailed["severity"],
                    "suggested_requirement": most_detailed["suggested_requirement"],
                    "affected_model_elements": most_detailed["affected_model_elements"],
                    "rationale": most_detailed["rationale"]
                })
        
        # Select model issues that exceed the weighted threshold
        selected_model_issues = []
        for key, model_issues in model_issue_groups.items():
            # Calculate total weight for this model issue
            issue_weight = sum(issue["weight"] for issue in model_issues)
            
            if issue_weight >= weighted_threshold:
                # Select the most detailed model issue
                most_detailed = max(model_issues, key=lambda x: len(x.get("description", "")))
                selected_model_issues.append({
                    "element_type": most_detailed["element_type"],
                    "element_name": most_detailed["element_name"],
                    "issue_type": most_detailed["issue_type"],
                    "severity": most_detailed["severity"],
                    "description": most_detailed["description"],
                    "suggested_fix": most_detailed["suggested_fix"],
                    "affected_requirements": most_detailed["affected_requirements"]
                })
        
        # Select requirement completeness entries that exceed the weighted threshold
        selected_completeness = []
        for key, completeness_entries in completeness_groups.items():
            if key and sum(entry["weight"] for entry in completeness_entries) >= weighted_threshold:
                # Calculate weighted average of completeness scores
                total_weighted_score = sum(entry["completeness_score"] * entry["weight"] for entry in completeness_entries)
                total_weights = sum(entry["weight"] for entry in completeness_entries)
                avg_score = total_weighted_score / total_weights if total_weights > 0 else 0
                
                # Select the most detailed completeness entry
                most_detailed = max(completeness_entries, key=lambda x: len(x.get("suggested_improvement", "")))
                
                # Combine missing elements from all entries
                all_missing_elements = set()
                for entry in completeness_entries:
                    all_missing_elements.update(entry.get("missing_elements", []))
                
                selected_completeness.append({
                    "requirement_id": most_detailed["requirement_id"],
                    "requirement_text": most_detailed["requirement_text"],
                    "completeness_score": round(avg_score, 1),
                    "missing_elements": list(all_missing_elements),
                    "suggested_improvement": most_detailed["suggested_improvement"],
                    "rationale": most_detailed["rationale"]
                })
        
        # Create aggregated analysis
        aggregated_analysis = {
            "requirement_issues": list(selected_req_issues.values()),
            "missing_requirements": selected_missing_reqs,
            "domain_model_issues": selected_model_issues,
            "requirement_completeness": selected_completeness
        }
        
        # Create the result with reasoning
        result = {
            "analysis": aggregated_analysis,
            "reasoning": f"This analysis was created by aggregating {len(analysis_results)} analyses " \
                        f"using weighted voting. Selected issues, missing requirements, and model issues " \
                        f"that exceeded the weighted threshold of {weighted_threshold:.1f}.",
            "aggregation_info": {
                "strategy": "weighted_vote",
                "model_count": len(analysis_results),
                "weighted_threshold": weighted_threshold,
                "contributing_models": [f"{ar['model_id']} (weight: {model_weights.get(ar['model_id'], default_weight)})" 
                                    for ar in analysis_results]
            }
        }
        
        return result
    
    def _llm_based_aggregation(self, domain_models, original_results, meta_model_id):
        """
        Use an LLM to aggregate domain models from multiple sources
        
        Args:
            domain_models (list): List of domain models from different LLMs
            original_results (list): Original results from LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
        
        Returns:
            dict: Aggregated domain model and reasoning
        """
        logger.info(f"Using {meta_model_id} LLM to aggregate domain models")
        
        try:
            # Prepare the prompt for the LLM
            prompt = """You are a senior software architect tasked with creating a consensus domain model by combining multiple domain models generated by different LLMs.

Analyze the provided domain models and create a single, coherent domain model that:
1. Includes all important classes from the source models
2. Resolves any conflicts between models
3. Combines the best elements of each model
4. Creates a complete and consistent view of the domain

FORMAT YOUR RESPONSE AS JSON with this structure:
{
    "domain_model": {
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
    },
    "reasoning": "Explanation of how you combined the models and resolved conflicts"
}

SOURCE DOMAIN MODELS:
"""
            
            # Add each source domain model
            for i, dm in enumerate(domain_models):
                model_json = json.dumps(dm["domain_model"], indent=2)
                prompt += f"\n\nMODEL {i+1} (from {dm['model_id']}):\n{model_json}\n"
            
            # Create the messages for the LLM
            messages = [
                {"role": "system", "content": "You are an expert software architect specializing in domain modeling."},
                {"role": "user", "content": prompt}
            ]
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            response = adapter.generate_response(messages)
            
            # Extract and validate the JSON
            result_json = response["content"]
            logger.debug(f"Meta-model response content sample: {result_json[:200]}...")
            
            result = extract_json_from_response(result_json)
            if not result:
                logger.error("Failed to extract valid JSON from meta-model response")
                # Fall back to majority vote
                return self._majority_vote_domain_models(domain_models, original_results)
            
            # Validate the domain model
            if "domain_model" in result:
                result["domain_model"] = validate_domain_model(result["domain_model"])
            else:
                logger.error("Meta-model response missing domain_model key")
                # Fall back to majority vote
                return self._majority_vote_domain_models(domain_models, original_results)
            
            # Add aggregation info
            result["aggregation_info"] = {
                "strategy": f"llm_based_{meta_model_id}",
                "model_count": len(domain_models),
                "contributing_models": [dm["model_id"] for dm in domain_models],
                "meta_model_id": meta_model_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM-based aggregation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to majority vote
            logger.info("Falling back to majority vote due to LLM aggregation error")
            return self._majority_vote_domain_models(domain_models, original_results)
    
    def _llm_based_analysis_aggregation(self, analysis_results, original_results, meta_model_id):
        """
        Use an LLM to aggregate analysis results from multiple sources
        
        Args:
            analysis_results (list): List of analysis results from different LLMs
            original_results (list): Original results from LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
        
        Returns:
            dict: Aggregated analysis results and reasoning
        """
        logger.info(f"Using {meta_model_id} LLM to aggregate analysis results")
        
        try:
            # Prepare the prompt for the LLM
            prompt = """You are a senior requirements analyst tasked with creating a consensus analysis by combining multiple requirement analyses generated by different LLMs.

Analyze the provided analyses and create a single, coherent analysis that:
1. Identifies the most important requirement issues
2. Highlights missing requirements detected across multiple analyses
3. Captures domain model issues that need attention
4. Provides completeness assessment for requirements

FORMAT YOUR RESPONSE AS JSON with this structure:
{
    "analysis": {
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
        ],
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
    },
    "reasoning": "Explanation of how you combined the analyses and resolved conflicts"
}

SOURCE ANALYSES:
"""
            
            # Add each source analysis
            for i, ar in enumerate(analysis_results):
                analysis_json = json.dumps(ar["analysis"], indent=2)
                prompt += f"\n\nANALYSIS {i+1} (from {ar['model_id']}):\n{analysis_json}\n"
            
            # Create the messages for the LLM
            messages = [
                {"role": "system", "content": "You are an expert requirements analyst specializing in requirements engineering."},
                {"role": "user", "content": prompt}
            ]
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            response = adapter.generate_response(messages)
            
            # Extract and validate the JSON
            result_json = response["content"]
            logger.debug(f"Meta-model response content sample: {result_json[:200]}...")
            
            result = extract_json_from_response(result_json)
            if not result:
                logger.error("Failed to extract valid JSON from meta-model response")
                # Fall back to majority vote
                return self._majority_vote_analysis(analysis_results, original_results)
            
            # Ensure analysis key exists
            if "analysis" not in result:
                logger.error("Meta-model response missing analysis key")
                # Fall back to majority vote
                return self._majority_vote_analysis(analysis_results, original_results)
            
            # Add aggregation info
            result["aggregation_info"] = {
                "strategy": f"llm_based_{meta_model_id}",
                "model_count": len(analysis_results),
                "contributing_models": [ar["model_id"] for ar in analysis_results],
                "meta_model_id": meta_model_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM-based analysis aggregation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to majority vote
            logger.info("Falling back to majority vote due to LLM aggregation error")
            return self._majority_vote_analysis(analysis_results, original_results)