import logging
import json
import traceback
import re
from collections import Counter
from utils.json_utils import extract_json_from_response, validate_domain_model, create_default_analysis
from services.llm_adapters import get_adapter
from config import get_available_meta_models

logger = logging.getLogger(__name__)

class ResultsAggregator:
    """
    Aggregates results from multiple LLMs using various strategies
    with improved preservation of unique insights and better error handling
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
        
        # Check if the normalized ID exists in available models
        if normalized_id not in self.meta_models:
            logger.warning(f"Unknown meta model ID: {meta_model_id} (normalized: {normalized_id}), falling back to improved aggregation")
            self.meta_model_id = "improved"
        else:
            self.meta_model_id = normalized_id
            
        logger.info(f"Using aggregation strategy: {self.meta_model_id}")

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
            "improved": "improved",
            
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
        return meta_model_mappings.get(meta_model_id, "improved")
    
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
        normalized_meta_id = self._normalize_meta_model_id(self.meta_model_id)
        
        if normalized_meta_id == "majority":
            return self._majority_vote_domain_models(domain_models, model_results)
        elif normalized_meta_id == "weighted":
            return self._weighted_vote_domain_models(domain_models, model_results)
        elif normalized_meta_id == "improved":
            return self._improved_domain_models(domain_models, model_results)
        elif normalized_meta_id in ["openai", "deepseek", "claude"]:
            return self._llm_based_aggregation(domain_models, model_results, normalized_meta_id)
        else:
            logger.warning(f"Unknown aggregation strategy: {self.meta_model_id}, falling back to improved aggregation")
            return self._improved_domain_models(domain_models, model_results)
    
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
        normalized_meta_id = self._normalize_meta_model_id(self.meta_model_id)
        
        if normalized_meta_id == "majority":
            return self._majority_vote_analysis(analysis_results, model_results)
        elif normalized_meta_id == "weighted":
            return self._weighted_vote_analysis(analysis_results, model_results)
        elif normalized_meta_id == "improved":
            return self._improved_analysis(analysis_results, model_results)
        elif normalized_meta_id in ["openai", "deepseek", "claude"]:
            return self._llm_based_analysis_aggregation(analysis_results, model_results, normalized_meta_id)
        else:
            logger.warning(f"Unknown aggregation strategy: {self.meta_model_id}, falling back to improved analysis")
            return self._improved_analysis(analysis_results, model_results)
    

    def _is_similar_issue(self, issue1, issue2):
        """Check if two issues are similar based on type and description"""
        # Check issue type
        if issue1.get("issue_type") != issue2.get("issue_type"):
            return False
            
        # Compare descriptions
        desc1 = issue1.get("description", "").lower()
        desc2 = issue2.get("description", "").lower()
        
        if desc1 and desc2:
            return self._text_similarity(desc1, desc2) > 0.6
            
        return False
        
    def _group_similar_missing_requirements(self, requirements):
        """
        Group similar missing requirements to avoid duplication
        while preserving unique insights
        """
        # Initialize groups
        groups = []
        
        for req in requirements:
            # Extract key information for similarity comparison
            req_desc = req.get("description", "").lower()
            req_category = req.get("category", "").lower()
            req_elements = [el.lower() for el in req.get("affected_model_elements", [])]
            req_suggestion = req.get("suggested_requirement", "").lower()
            
            # Check if this requirement is similar to any existing group
            matched = False
            for group in groups:
                similarity_score = 0
                
                # Compare with each requirement in the group
                for group_req in group:
                    group_desc = group_req.get("description", "").lower()
                    group_category = group_req.get("category", "").lower()
                    group_elements = [el.lower() for el in group_req.get("affected_model_elements", [])]
                    group_suggestion = group_req.get("suggested_requirement", "").lower()
                    
                    # Calculate similarity based on multiple factors
                    # Description similarity (most important)
                    desc_similarity = self._text_similarity(req_desc, group_desc)
                    if desc_similarity > 0.6:
                        similarity_score += 3
                    
                    # Category similarity
                    if req_category and group_category and req_category == group_category:
                        similarity_score += 1
                    
                    # Affected elements similarity
                    common_elements = set(req_elements).intersection(set(group_elements))
                    if common_elements:
                        similarity_score += len(common_elements) / max(len(req_elements), len(group_elements))
                    
                    # Suggested requirement similarity
                    if req_suggestion and group_suggestion:
                        sugg_similarity = self._text_similarity(req_suggestion, group_suggestion)
                        if sugg_similarity > 0.5:
                            similarity_score += 1
                
                # If similarity score is high enough, add to this group
                if similarity_score >= 3:  # Threshold for considering requirements similar
                    group.append(req)
                    matched = True
                    break
            
            # If no match found, create a new group
            if not matched:
                groups.append([req])
        
        return groups
    
    def _select_best_requirement(self, group):
        """
        Select the best requirement from a group of similar requirements
        """
        if len(group) == 1:
            return group[0].copy()
        
        # Score each requirement based on completeness and clarity
        scored_reqs = []
        for req in group:
            score = 0
            
            # More detailed description (longer but not too long)
            desc_len = len(req.get("description", ""))
            if 30 <= desc_len <= 150:
                score += 1
                
            # More specific suggestion
            sugg_len = len(req.get("suggested_requirement", ""))
            if sugg_len >= 50:
                score += 1
                
            # More detailed rationale
            rationale_len = len(req.get("rationale", ""))
            if rationale_len >= 80:
                score += 1
                
            # Has affected model elements
            if req.get("affected_model_elements", []):
                score += 1
                
            # Higher severity (CRITICAL > HIGH > MEDIUM > LOW)
            severity = req.get("severity", "").upper()
            if severity == "CRITICAL":
                score += 3
            elif severity == "HIGH":
                score += 2
            elif severity == "MEDIUM":
                score += 1
                
            # Prefer requirements from certain models if needed
            # This can be customized based on model performance
            model = req.get("source_model", "").lower()
            if model == "claude":
                score += 0.2
            elif model == "deepseek":
                score += 0.1
                
            scored_reqs.append((score, req))
        
        # Choose the requirement with the highest score
        best_req = max(scored_reqs, key=lambda x: x[0])[1].copy()
        
        # Combine sources if this represents multiple models' findings
        source_models = set(req.get("source_model", "unknown") for req in group)
        if len(source_models) > 1:
            best_req["source_models"] = list(source_models)
        
        # Remove the single source model field
        if "source_model" in best_req:
            del best_req["source_model"]
            
        return best_req
    
    def _improved_domain_models(self, domain_models, original_results):
        """
        Enhanced aggregation method for domain models that preserves unique insights
        """
        logger.info("Using improved aggregation method for domain models")
        
        # Extract all classes and relationships
        all_classes = {}
        all_relationships = {}
        
        for model in domain_models:
            model_id = model.get("model_id", "unknown")
            dm = model.get("domain_model", {})
            
            # Process classes
            for cls in dm.get("classes", []):
                name = cls.get("name")
                if not name:
                    continue
                    
                if name not in all_classes:
                    all_classes[name] = cls
                else:
                    # Merge with existing class, keeping the most detailed version
                    existing = all_classes[name]
                    
                    # Keep the longer description
                    if len(cls.get("description", "")) > len(existing.get("description", "")):
                        existing["description"] = cls.get("description", "")
                    
                    # Merge attributes, avoiding duplicates
                    existing_attrs = {attr.get("name"): attr for attr in existing.get("attributes", [])}
                    for attr in cls.get("attributes", []):
                        attr_name = attr.get("name")
                        if attr_name and attr_name not in existing_attrs:
                            if "attributes" not in existing:
                                existing["attributes"] = []
                            existing["attributes"].append(attr)
                    
                    # Merge methods, avoiding duplicates
                    existing_methods = {method.get("name"): method for method in existing.get("methods", [])}
                    for method in cls.get("methods", []):
                        method_name = method.get("name")
                        if method_name and method_name not in existing_methods:
                            if "methods" not in existing:
                                existing["methods"] = []
                            existing["methods"].append(method)
            
            # Process relationships
            for rel in dm.get("relationships", []):
                source = rel.get("source")
                target = rel.get("target")
                rel_type = rel.get("type")
                
                if not (source and target and rel_type):
                    continue
                    
                key = f"{source}:{target}:{rel_type}"
                
                if key not in all_relationships:
                    all_relationships[key] = rel
                else:
                    # Keep the more detailed relationship
                    existing = all_relationships[key]
                    
                    # Keep source/target multiplicity if present
                    if rel.get("sourceMultiplicity") and not existing.get("sourceMultiplicity"):
                        existing["sourceMultiplicity"] = rel.get("sourceMultiplicity")
                    
                    if rel.get("targetMultiplicity") and not existing.get("targetMultiplicity"):
                        existing["targetMultiplicity"] = rel.get("targetMultiplicity")
                    
                    # Keep the longer description
                    if len(rel.get("description", "")) > len(existing.get("description", "")):
                        existing["description"] = rel.get("description", "")
        
        # Create the aggregated domain model
        aggregated_model = {
            "classes": list(all_classes.values()),
            "relationships": list(all_relationships.values())
        }
        
        # Choose the most complete PlantUML diagram
        plantuml_diagrams = [(len(model["domain_model"].get("plantuml", "")), model["domain_model"].get("plantuml", "")) 
                            for model in domain_models 
                            if "domain_model" in model and model["domain_model"].get("plantuml")]
        
        if plantuml_diagrams:
            # Get the longest diagram
            aggregated_model["plantuml"] = max(plantuml_diagrams, key=lambda x: x[0])[1]
        else:
            # Generate a simple one if none exists
            aggregated_model["plantuml"] = "@startuml\n@enduml"
        
        return {
            "domain_model": aggregated_model,
            "reasoning": "Domain model created using improved aggregation method to preserve unique elements from all models",
            "aggregation_info": {
                "strategy": "improved_aggregation",
                "model_count": len(domain_models),
                "contributing_models": [dm["model_id"] for dm in domain_models]
            }
        }
    
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
    
    def _llm_based_aggregation(self, domain_models, original_results, meta_model_id):
        """
        Use an LLM to aggregate domain models from multiple sources
        with improved error handling and better prompting
        
        Args:
            domain_models (list): List of domain models from different LLMs
            original_results (list): Original results from LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
        
        Returns:
            dict: Aggregated domain model and reasoning
        """
        logger.info(f"Using {meta_model_id} LLM to aggregate domain models")
        
        try:
            # Prepare the prompt for the LLM with clearer instructions
            prompt = """You are a senior software architect tasked with creating a consensus domain model by combining multiple domain models generated by different LLMs.

Analyze the provided domain models and create a single, coherent domain model that:
1. Includes all important classes from the source models
2. Resolves any conflicts between models
3. Combines the best elements of each model
4. Creates a complete and consistent view of the domain

IMPORTANT: Your response MUST be valid JSON with the exact structure shown below.
Do not include any explanatory text outside the JSON.

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
                {"role": "system", "content": "You are an expert software architect specializing in domain modeling and JSON responses. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            response = adapter.generate_response(messages)
            
            # Extract and validate the JSON
            result_json = response["content"]
            logger.debug(f"Meta-model response content sample: {result_json[:200]}...")
            
            # # Save raw response for debugging
            # with open(f"log/meta_model_raw_response_{meta_model_id}.txt", "w", encoding="utf-8") as f:
            #     f.write(result_json)
            
            # Use enhanced JSON extraction
            result = extract_json_from_response(result_json)
            if not result:
                logger.error("Failed to extract valid JSON from meta-model response")
                # Notify the user about the fallback
                logger.warning(f"Falling back to improved aggregation due to JSON parsing issues with {meta_model_id}")
                return self._improved_domain_models(domain_models, original_results)
            
            # Validate the domain model
            if "domain_model" in result:
                result["domain_model"] = validate_domain_model(result["domain_model"])
            else:
                logger.error("Meta-model response missing domain_model key")
                logger.warning(f"Falling back to improved aggregation because {meta_model_id} response lacks domain_model")
                return self._improved_domain_models(domain_models, original_results)
            
            # Add aggregation info
            result["aggregation_info"] = {
                "strategy": f"llm_based_{meta_model_id}",
                "model_count": len(domain_models),
                "contributing_models": [dm["model_id"] for dm in domain_models],
                "meta_model_id": meta_model_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM-based aggregation with {meta_model_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Provide detailed error for debugging
            logger.warning(f"Falling back to improved aggregation due to error with {meta_model_id} meta-analysis: {str(e)}")
            
            # Fall back to improved aggregation
            return self._improved_domain_models(domain_models, original_results)
    
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
            if len(set(issue["model_id"] for issue in issues)) >= majority_threshold:
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
            if len(set(req["model_id"] for req in missing_reqs)) >= majority_threshold:
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
            if len(set(issue["model_id"] for issue in model_issues)) >= majority_threshold:
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
            if key and len(set(entry["model_id"] for entry in completeness_entries)) >= majority_threshold:
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
        
        Similar to majority voting but assigns weights to different models.
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
            # Calculate total weight for this issue group
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
                         f"that exceeded the weighted threshold.",
            "aggregation_info": {
                "strategy": "weighted_vote",
                "model_count": len(analysis_results),
                "weighted_threshold": weighted_threshold,
                "contributing_models": [f"{ar['model_id']} (weight: {model_weights.get(ar['model_id'], default_weight)})" 
                                    for ar in analysis_results]
            }
        }
        
        return result
    
    def _llm_based_analysis_aggregation(self, analysis_results, original_results, meta_model_id):
        """
        Improved version that uses an LLM to aggregate analysis results with better error handling
        and splitting different analysis components into specialized functions for more focused prompting
        
        Args:
            analysis_results (list): List of analysis results from different LLMs
            original_results (list): Original results from LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
        
        Returns:
            dict: Aggregated analysis results and reasoning
        """
        logger.info(f"Using {meta_model_id} LLM to aggregate analysis results")
        
        try:
            # Create the final aggregated analysis container
            final_analysis = {
                "requirement_issues": [],
                "missing_requirements": [],
                "domain_model_issues": [],
                "requirement_completeness": []
            }
            
            # Track reasoning for each component
            component_reasoning = {}
            
            # 1. Process requirement completeness with specialized function
            logger.info("Processing requirement completeness with specialized aggregation")
            req_completeness, req_completeness_reasoning = self._aggregate_requirement_completeness(analysis_results, meta_model_id)
            final_analysis["requirement_completeness"] = req_completeness
            component_reasoning["requirement_completeness"] = req_completeness_reasoning
            
            # 2. Process requirement issues using specialized function
            logger.info("Processing requirement issues with specialized aggregation")
            req_issues, req_issues_reasoning = self._aggregate_requirement_issues(analysis_results, meta_model_id)
            final_analysis["requirement_issues"] = req_issues
            component_reasoning["requirement_issues"] = req_issues_reasoning
            
            # 3. Process missing requirements using specialized function
            logger.info("Processing missing requirements with specialized aggregation")
            missing_reqs, missing_reqs_reasoning = self._aggregate_missing_requirements(analysis_results, meta_model_id)
            final_analysis["missing_requirements"] = missing_reqs
            component_reasoning["missing_requirements"] = missing_reqs_reasoning
            
            # 4. Process domain model issues using specialized function
            logger.info("Processing domain model issues with specialized aggregation")
            model_issues, model_issues_reasoning = self._aggregate_domain_model_issues(analysis_results, meta_model_id)
            final_analysis["domain_model_issues"] = model_issues
            component_reasoning["domain_model_issues"] = model_issues_reasoning
            
            # Create the combined reasoning
            combined_reasoning = f"Analysis aggregated using {meta_model_id} with specialized component processing:\n"
            for component, reasoning in component_reasoning.items():
                combined_reasoning += f"- {component}: {reasoning}\n"
            
            # Create the final result with all components
            result = {
                "analysis": final_analysis,
                "reasoning": combined_reasoning.strip(),
                "aggregation_info": {
                    "strategy": f"llm_based_{meta_model_id}_specialized",
                    "model_count": len(analysis_results),
                    "contributing_models": [ar["model_id"] for ar in analysis_results],
                    "meta_model_id": meta_model_id
                }
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Error in LLM-based analysis aggregation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Use improved aggregation instead of majority vote on failure
            logger.warning(f"Falling back to improved aggregation due to error with {meta_model_id}: {str(e)}")
            return self._improved_analysis(analysis_results, original_results)
    
    def _aggregate_requirement_issues(self, analysis_results, meta_model_id):
        """
        Specialized function to aggregate requirement issues with a focused prompt
        
        Args:
            analysis_results (list): List of analysis results from different LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
            
        Returns:
            tuple: (aggregated requirement issues, reasoning)
        """
        # Extract just requirement issues from all analyses
        all_req_issues = []
        for ar in analysis_results:
            if "analysis" in ar and "requirement_issues" in ar["analysis"]:
                req_issues = ar["analysis"]["requirement_issues"]
                if req_issues:
                    all_req_issues.append({
                        "model_id": ar["model_id"],
                        "requirement_issues": req_issues
                    })
        
        # If no requirement issues found, return empty list
        if not all_req_issues:
            return [], "No requirement issues found in any analysis"
            
        # If only one model provided requirement issues, use them directly
        if len(all_req_issues) == 1:
            return all_req_issues[0]["requirement_issues"], f"Used requirement issues from {all_req_issues[0]['model_id']} (only contributor)"
        
        try:
            # Prepare specialized prompt for requirement issues
            prompt = """You are a senior requirements analyst tasked with creating a consensus view of requirement issues by combining analyses from multiple LLMs.

Analyze the provided requirement issues and create a single, comprehensive list that:
1. Preserves all unique issues from each source analysis
2. Resolves any conflicts or contradictions
3. Combines similar issues with the most detailed information
4. Ensures the final output is consistent and well-structured

IMPORTANT: Your response MUST be valid JSON with the exact structure shown below.
Do not include any explanatory text outside the JSON.

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
    "reasoning": "Your explanation of how you combined the requirement issues"
}

SOURCE REQUIREMENT ISSUES:
"""
            
            # Add each source's requirement issues
            for i, req_issue_data in enumerate(all_req_issues):
                model_id = req_issue_data["model_id"]
                req_issues = req_issue_data["requirement_issues"]
                issues_json = json.dumps(req_issues, indent=2)
                prompt += f"\n\nREQUIREMENT ISSUES {i+1} (from {model_id}):\n{issues_json}\n"
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            messages = [
                {"role": "system", "content": "You are an expert requirements analyst specializing in identifying and combining requirement issues. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            response = adapter.generate_response(messages)
            
            # Extract and validate the JSON
            result_json = response["content"]
            result = extract_json_from_response(result_json)
            
            if not result or "requirement_issues" not in result:
                logger.warning(f"Failed to extract valid requirement issues from {meta_model_id} response")
                # Fall back to improved aggregation for this component
                return self._improved_requirement_issues(analysis_results), f"Used improved aggregation due to issues with {meta_model_id} response"
            
            reasoning = result.get("reasoning", "Successfully aggregated requirement issues using specialized prompt")
            return result["requirement_issues"], reasoning
            
        except Exception as e:
            logger.error(f"Error in requirement issues aggregation: {str(e)}")
            # Fall back to improved aggregation for this component
            return self._improved_requirement_issues(analysis_results), f"Used improved aggregation due to error: {str(e)}"
    
    def _improved_requirement_issues(self, analysis_results):
        """Fallback method to aggregate requirement issues using improved direct method"""
        req_issues_map = {}
        for ar in analysis_results:
            if "analysis" in ar and "requirement_issues" in ar["analysis"]:
                for issue in ar["analysis"]["requirement_issues"]:
                    req_id = issue.get("requirement_id", "")
                    if req_id:
                        if req_id not in req_issues_map:
                            req_issues_map[req_id] = issue
                        else:
                            # Merge issues for the same requirement
                            existing = req_issues_map[req_id].get("issues", [])
                            new = issue.get("issues", [])
                            combined = existing[:]
                            
                            # Add any new issues not already present
                            for new_issue in new:
                                if not any(self._is_similar_issue(new_issue, existing_issue) 
                                        for existing_issue in existing):
                                    combined.append(new_issue)
                            
                            req_issues_map[req_id]["issues"] = combined
        
        return list(req_issues_map.values())
    
    def _aggregate_missing_requirements(self, analysis_results, meta_model_id):
        """
        Specialized function to aggregate missing requirements with a focused prompt
        
        Args:
            analysis_results (list): List of analysis results from different LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
            
        Returns:
            tuple: (aggregated missing requirements, reasoning)
        """
        # Extract just missing requirements from all analyses
        all_missing_reqs = []
        for ar in analysis_results:
            if "analysis" in ar and "missing_requirements" in ar["analysis"]:
                missing_reqs = ar["analysis"]["missing_requirements"]
                if missing_reqs:
                    all_missing_reqs.append({
                        "model_id": ar["model_id"],
                        "missing_requirements": missing_reqs
                    })
        
        # If no missing requirements found, return empty list
        if not all_missing_reqs:
            return [], "No missing requirements found in any analysis"
            
        # If only one model provided missing requirements, use them directly
        if len(all_missing_reqs) == 1:
            return all_missing_reqs[0]["missing_requirements"], f"Used missing requirements from {all_missing_reqs[0]['model_id']} (only contributor)"
        
        try:
            # Prepare specialized prompt for missing requirements
            prompt = """You are a senior requirements analyst tasked with creating a comprehensive list of missing requirements by combining analyses from multiple LLMs.

Analyze the provided missing requirements and create a single, comprehensive list that:
1. Preserves ALL unique missing requirements from each source analysis - this is critical
2. Combines similar requirements with the most detailed information
3. Ensures each requirement has a unique ID (MR1, MR2, etc.)
4. Provides the most detailed rationale for each missing requirement

IMPORTANT: Your response MUST be valid JSON with the exact structure shown below.
Do not include any explanatory text outside the JSON.

{
    "missing_requirements": [
        {
            "id": "MR1",
            "description": "Description of what's missing",
            "category": "Category",
            "severity": "CRITICAL|HIGH|MEDIUM|LOW",
            "suggested_requirement": "Suggested text for the requirement",
            "affected_model_elements": ["Class1", "Relationship2"],
            "rationale": "Why this requirement should exist"
        }
    ],
    "reasoning": "Your explanation of how you combined the missing requirements"
}

SOURCE MISSING REQUIREMENTS:
"""
            
            # Add each source's missing requirements
            for i, missing_req_data in enumerate(all_missing_reqs):
                model_id = missing_req_data["model_id"]
                missing_reqs = missing_req_data["missing_requirements"]
                missing_reqs_json = json.dumps(missing_reqs, indent=2)
                prompt += f"\n\nMISSING REQUIREMENTS {i+1} (from {model_id}):\n{missing_reqs_json}\n"
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            messages = [
                {"role": "system", "content": "You are an expert requirements analyst specializing in identifying missing requirements. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            response = adapter.generate_response(messages)
            
            # Extract and validate the JSON
            result_json = response["content"]
            result = extract_json_from_response(result_json)
            
            if not result or "missing_requirements" not in result:
                logger.warning(f"Failed to extract valid missing requirements from {meta_model_id} response")
                # Fall back to improved aggregation for this component
                return self._improved_missing_requirements(analysis_results), f"Used improved aggregation due to issues with {meta_model_id} response"
            
            reasoning = result.get("reasoning", "Successfully aggregated missing requirements using specialized prompt")
            return result["missing_requirements"], reasoning
            
        except Exception as e:
            logger.error(f"Error in missing requirements aggregation: {str(e)}")
            # Fall back to improved aggregation for this component
            return self._improved_missing_requirements(analysis_results), f"Used improved aggregation due to error: {str(e)}"
    
    def _improved_missing_requirements(self, analysis_results):
        """Fallback method to aggregate missing requirements using improved direct method"""
        all_missing_reqs = []
        for ar in analysis_results:
            if "analysis" in ar and "missing_requirements" in ar["analysis"]:
                model_id = ar.get("model_id", "unknown")
                for req in ar["analysis"]["missing_requirements"]:
                    # Add source tracking
                    req_copy = req.copy()
                    req_copy["source_model"] = model_id
                    all_missing_reqs.append(req_copy)
        
        # Group similar requirements to avoid duplication
        grouped_reqs = self._group_similar_missing_requirements(all_missing_reqs)
        final_missing_reqs = []
        
        # Select the best requirement from each group
        for i, group in enumerate(grouped_reqs):
            best_req = self._select_best_requirement(group)
            best_req["id"] = f"MR{i+1}"
            final_missing_reqs.append(best_req)
            
        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        final_missing_reqs.sort(key=lambda x: severity_order.get(x.get("severity", "").upper(), 4))
        
        return final_missing_reqs
    
    def _aggregate_domain_model_issues(self, analysis_results, meta_model_id):
        """
        Specialized function to aggregate domain model issues with a focused prompt
        
        Args:
            analysis_results (list): List of analysis results from different LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
            
        Returns:
            tuple: (aggregated domain model issues, reasoning)
        """
        # Extract just domain model issues from all analyses
        all_model_issues = []
        for ar in analysis_results:
            if "analysis" in ar and "domain_model_issues" in ar["analysis"]:
                model_issues = ar["analysis"]["domain_model_issues"]
                if model_issues:
                    all_model_issues.append({
                        "model_id": ar["model_id"],
                        "domain_model_issues": model_issues
                    })
        
        # If no domain model issues found, return empty list
        if not all_model_issues:
            return [], "No domain model issues found in any analysis"
            
        # If only one model provided domain model issues, use them directly
        if len(all_model_issues) == 1:
            return all_model_issues[0]["domain_model_issues"], f"Used domain model issues from {all_model_issues[0]['model_id']} (only contributor)"
        
        try:
            # Prepare specialized prompt for domain model issues
            prompt = """You are a senior domain modeler tasked with creating a consensus view of domain model issues by combining analyses from multiple LLMs.

Analyze the provided domain model issues and create a single, comprehensive list that:
1. Preserves all unique issues from each source analysis
2. Combines similar issues with the most detailed information
3. Ensures the final output consistently identifies important model problems
4. Provides clear suggested fixes for each issue

IMPORTANT: Your response MUST be valid JSON with the exact structure shown below.
Do not include any explanatory text outside the JSON.

{
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
    "reasoning": "Your explanation of how you combined the domain model issues"
}

SOURCE DOMAIN MODEL ISSUES:
"""
            
            # Add each source's domain model issues
            for i, model_issue_data in enumerate(all_model_issues):
                model_id = model_issue_data["model_id"]
                model_issues = model_issue_data["domain_model_issues"]
                model_issues_json = json.dumps(model_issues, indent=2)
                prompt += f"\n\nDOMAIN MODEL ISSUES {i+1} (from {model_id}):\n{model_issues_json}\n"
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            messages = [
                {"role": "system", "content": "You are an expert domain modeler specializing in identifying model issues. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            response = adapter.generate_response(messages)
            
            # Extract and validate the JSON
            result_json = response["content"]
            result = extract_json_from_response(result_json)
            
            if not result or "domain_model_issues" not in result:
                logger.warning(f"Failed to extract valid domain model issues from {meta_model_id} response")
                # Fall back to improved aggregation for this component
                return self._improved_domain_model_issues(analysis_results), f"Used improved aggregation due to issues with {meta_model_id} response"
            
            reasoning = result.get("reasoning", "Successfully aggregated domain model issues using specialized prompt")
            return result["domain_model_issues"], reasoning
            
        except Exception as e:
            logger.error(f"Error in domain model issues aggregation: {str(e)}")
            # Fall back to improved aggregation for this component
            return self._improved_domain_model_issues(analysis_results), f"Used improved aggregation due to error: {str(e)}"
    
    def _improved_domain_model_issues(self, analysis_results):
        """Fallback method to aggregate domain model issues using improved direct method"""
        model_issues_map = {}
        for ar in analysis_results:
            if "analysis" in ar and "domain_model_issues" in ar["analysis"]:
                for issue in ar["analysis"]["domain_model_issues"]:
                    element = issue.get("element_name", "")
                    issue_type = issue.get("issue_type", "")
                    if element and issue_type:
                        key = f"{element}:{issue_type}"
                        if key not in model_issues_map:
                            model_issues_map[key] = issue
                        elif len(issue.get("description", "")) > len(model_issues_map[key].get("description", "")):
                            # Keep the most detailed description
                            model_issues_map[key] = issue
        
        return list(model_issues_map.values())
    
    def _improved_analysis(self, analysis_results, original_results):
        """
        Enhanced aggregation method that preserves unique insights from all models
        with specific focus on preserving ALL requirement completeness analyses
        """
        logger.info("Using improved aggregation method to preserve unique insights")
        
        # Initialize consolidated results
        consolidated_analysis = {
            "requirement_issues": [],
            "missing_requirements": [],
            "domain_model_issues": [],
            "requirement_completeness": []
        }
        
        # Process missing requirements - gather all unique ones
        all_missing_reqs = []
        for result in analysis_results:
            if "analysis" in result and "missing_requirements" in result["analysis"]:
                model_id = result.get("model_id", "unknown")
                for req in result["analysis"]["missing_requirements"]:
                    # Add source tracking
                    req["source_model"] = model_id
                    all_missing_reqs.append(req)
        
        # Group similar requirements to avoid duplication
        grouped_reqs = self._group_similar_missing_requirements(all_missing_reqs)
        final_missing_reqs = []
        
        # Select the best requirement from each group
        for i, group in enumerate(grouped_reqs):
            best_req = self._select_best_requirement(group)
            best_req["id"] = f"MR{i+1}"
            final_missing_reqs.append(best_req)
            
        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        final_missing_reqs.sort(key=lambda x: severity_order.get(x.get("severity", "").upper(), 4))
        
        consolidated_analysis["missing_requirements"] = final_missing_reqs
        
        # Similar handling for requirement issues
        req_issues_map = {}
        for result in analysis_results:
            if "analysis" in result and "requirement_issues" in result["analysis"]:
                for issue in result["analysis"]["requirement_issues"]:
                    req_id = issue.get("requirement_id", "")
                    if req_id:
                        if req_id not in req_issues_map:
                            req_issues_map[req_id] = issue
                        else:
                            # Merge issues for the same requirement
                            existing = req_issues_map[req_id].get("issues", [])
                            new = issue.get("issues", [])
                            combined = existing[:]
                            
                            # Add any new issues not already present
                            for new_issue in new:
                                if not any(self._is_similar_issue(new_issue, existing_issue) 
                                        for existing_issue in existing):
                                    combined.append(new_issue)
                            
                            req_issues_map[req_id]["issues"] = combined
        
        consolidated_analysis["requirement_issues"] = list(req_issues_map.values())
        
        # Domain model issues
        model_issues_map = {}
        for result in analysis_results:
            if "analysis" in result and "domain_model_issues" in result["analysis"]:
                for issue in result["analysis"]["domain_model_issues"]:
                    element = issue.get("element_name", "")
                    issue_type = issue.get("issue_type", "")
                    if element and issue_type:
                        key = f"{element}:{issue_type}"
                        if key not in model_issues_map:
                            model_issues_map[key] = issue
                        elif len(issue.get("description", "")) > len(model_issues_map[key].get("description", "")):
                            # Keep the most detailed description
                            model_issues_map[key] = issue
        
        consolidated_analysis["domain_model_issues"] = list(model_issues_map.values())
        
        # IMPROVED: Use the dedicated method for completeness preservation
        consolidated_analysis["requirement_completeness"] = self._merge_all_requirement_completeness(analysis_results)
        
        return {
            "analysis": consolidated_analysis,
            "reasoning": "Analysis combined using improved aggregation method to preserve unique insights from all models",
            "aggregation_info": {
                "strategy": "improved_aggregation",
                "model_count": len(analysis_results),
                "contributing_models": [ar["model_id"] for ar in analysis_results]
            }
        }
    
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
    
    def aggregate_extraction_results(self, model_results):
        """
        Aggregate general extraction results (for extract_context_from_srs)
        with support for meta-model based aggregation
        """
        logger.info(f"Aggregating extraction results using {self.meta_model_id} strategy")
        
        if not model_results:
            logger.error("No extraction results to aggregate")
            return {}
            
        if len(model_results) == 1:
            logger.info("Only one extraction result, no aggregation needed")
            return model_results[0]
        
        # Use the appropriate aggregation strategy based on meta-model ID
        normalized_meta_id = self._normalize_meta_model_id(self.meta_model_id)
        
        if normalized_meta_id in ["openai", "deepseek", "claude"]:
            return self._llm_based_extraction_aggregation(model_results, normalized_meta_id)
        else:
            # For majority/weighted/improved strategies
            # Choose the most detailed extraction result
            logger.info(f"Using heuristic approach for {normalized_meta_id} strategy")
            most_detailed = max(model_results, 
                            key=lambda x: sum(len(str(v)) for v in x.values() if isinstance(v, (str, list, dict))))
            
            # Add aggregation info
            if "aggregation_info" not in most_detailed:
                most_detailed["aggregation_info"] = {
                    "strategy": "most_detailed",
                    "model_count": len(model_results),
                    "selected_model": most_detailed.get("model_id", "unknown")
                }
            
            return most_detailed

    def _llm_based_extraction_aggregation(self, model_results, meta_model_id):
        """
        Use an LLM to aggregate extraction results from multiple sources
        
        Args:
            model_results (list): List of extraction results from different LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
        
        Returns:
            dict: Aggregated extraction results
        """
        logger.info(f"Using {meta_model_id} LLM to aggregate extraction results")
        
        try:
            # Get model IDs for reference
            model_ids = []
            for result in model_results:
                model_id = result.get("model_id", "unknown")
                if model_id not in model_ids:
                    model_ids.append(model_id)
            
            # Prepare the prompt for the LLM
            prompt = """You are a senior requirements analyst tasked with creating a consensus extraction from multiple sources.

Analyze the provided extraction results from different LLMs and create a single, comprehensive result that:
1. Preserves all unique insights from each source
2. Resolves any conflicts or contradictions
3. Combines the best elements of each extraction
4. Creates a complete and consistent view of the extracted information

IMPORTANT: Your response MUST be valid JSON with the exact same structure as the input extractions.
Do not include any explanatory text outside the JSON. Maintain all key names and data types.

SOURCE EXTRACTIONS:
"""
            
            # Add each source extraction
            for i, result in enumerate(model_results):
                # Remove model_id if present to avoid bias
                clean_result = result.copy()
                clean_result.pop("model_id", None)
                clean_result.pop("aggregation_info", None)
                
                extraction_json = json.dumps(clean_result, indent=2)
                prompt += f"\n\nEXTRACTION {i+1} (from {model_ids[i] if i < len(model_ids) else 'unknown'}):\n{extraction_json}\n"
            
            # Create the messages for the LLM
            messages = [
                {"role": "system", "content": "You are an expert requirements analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            response = adapter.generate_response(messages)
            
            # Extract and validate the JSON
            result_json = response["content"]
            logger.debug(f"Meta-model extraction response content sample: {result_json[:200]}...")
            
            # # Save raw response for debugging
            # with open(f"log/meta_model_extraction_raw_response_{meta_model_id}.txt", "w", encoding="utf-8") as f:
            #     f.write(result_json)
            
            # Use enhanced JSON extraction
            result = extract_json_from_response(result_json)
            if not result:
                logger.error("Failed to extract valid JSON from meta-model extraction response")
                logger.warning(f"Falling back to heuristic aggregation due to JSON parsing issues with {meta_model_id}")
                
                # Fall back to heuristic approach
                most_detailed = max(model_results, 
                                key=lambda x: sum(len(str(v)) for v in x.values() if isinstance(v, (str, list, dict))))
                return most_detailed
            
            # Add aggregation info
            result["aggregation_info"] = {
                "strategy": f"llm_based_{meta_model_id}",
                "model_count": len(model_results),
                "contributing_models": model_ids,
                "meta_model_id": meta_model_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM-based extraction aggregation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to heuristic approach
            logger.warning(f"Falling back to heuristic aggregation due to error with {meta_model_id}: {str(e)}")
            most_detailed = max(model_results, 
                            key=lambda x: sum(len(str(v)) for v in x.values() if isinstance(v, (str, list, dict))))
            return most_detailed
    
    def _merge_all_requirement_completeness(self, analysis_results):
        """
        Simply merges all requirement completeness entries across all analyses,
        to preserve all unique insights
        """
        all_completeness = []
        completeness_by_req = {}
        
        for ar in analysis_results:
            if "analysis" in ar and "requirement_completeness" in ar["analysis"]:
                model_id = ar.get("model_id", "unknown")
                for completeness in ar["analysis"]["requirement_completeness"]:
                    req_id = completeness.get("requirement_id", "")
                    if req_id:
                        # Add source tracking
                        completeness_copy = completeness.copy()
                        completeness_copy["source_model"] = model_id
                        
                        if req_id not in completeness_by_req:
                            completeness_by_req[req_id] = []
                        completeness_by_req[req_id].append(completeness_copy)
        
        # For each requirement, choose the completeness entry with the lowest score
        # (most conservative assessment) or combine them
        for req_id, entries in completeness_by_req.items():
            if len(entries) == 1:
                # Just one entry, use it directly
                entry = entries[0].copy()
                if "source_model" in entry:
                    del entry["source_model"]  # Remove internal tracking
                all_completeness.append(entry)
            else:
                # Multiple entries, combine them
                # Use the most conservative completeness score (lowest)
                min_score_entry = min(entries, key=lambda x: x.get("completeness_score", 100))
                combined_entry = min_score_entry.copy()
                
                # Combine missing elements from all entries
                all_missing_elements = set()
                for entry in entries:
                    all_missing_elements.update(entry.get("missing_elements", []))
                combined_entry["missing_elements"] = list(all_missing_elements)
                
                # For suggested improvement, use the one from the entry with the lowest score
                # or the most detailed if scores are equal
                if len(entries) > 1 and any(e.get("completeness_score", 100) == min_score_entry.get("completeness_score", 100) for e in entries if e != min_score_entry):
                    same_score_entries = [e for e in entries if e.get("completeness_score", 100) == min_score_entry.get("completeness_score", 100)]
                    most_detailed = max(same_score_entries, key=lambda x: len(x.get("suggested_improvement", "")))
                    combined_entry["suggested_improvement"] = most_detailed.get("suggested_improvement", "")
                    combined_entry["rationale"] = most_detailed.get("rationale", "")
                
                # Add sources
                source_models = [entry.get("source_model", "unknown") for entry in entries]
                combined_entry["contributing_models"] = source_models
                
                # Remove internal tracking
                if "source_model" in combined_entry:
                    del combined_entry["source_model"]
                
                all_completeness.append(combined_entry)
        
        # Sort by requirement ID
        all_completeness.sort(key=lambda x: x.get("requirement_id", ""))
        
        return all_completeness
    
    def _aggregate_requirement_completeness(self, analysis_results, meta_model_id):
        """
        Specialized function to aggregate requirement completeness with a focused prompt
        
        Args:
            analysis_results (list): List of analysis results from different LLMs
            meta_model_id (str): ID of the LLM to use for aggregation
            
        Returns:
            tuple: (aggregated requirement completeness, reasoning)
        """
        # Extract just requirement completeness from all analyses
        all_completeness = []
        for ar in analysis_results:
            if "analysis" in ar and "requirement_completeness" in ar["analysis"]:
                completeness = ar["analysis"]["requirement_completeness"]
                if completeness:
                    all_completeness.append({
                        "model_id": ar["model_id"],
                        "requirement_completeness": completeness
                    })
        
        # If no requirement completeness found, return empty list
        if not all_completeness:
            return [], "No requirement completeness found in any analysis"
            
        # If only one model provided requirement completeness, use it directly
        if len(all_completeness) == 1:
            return all_completeness[0]["requirement_completeness"], f"Used requirement completeness from {all_completeness[0]['model_id']} (only contributor)"
        
        try:
            # Prepare specialized prompt for requirement completeness
            prompt = """You are a senior requirements analyst tasked with creating a comprehensive assessment of requirement completeness by combining analyses from multiple LLMs.

Analyze the provided requirement completeness assessments and create a single, comprehensive list that:
1. Preserves ALL unique insights about each requirement's completeness
2. Takes a conservative approach to completeness scores (prefer lower scores when in doubt)
3. Combines all identified missing elements from different analyses
4. Provides detailed suggested improvements and rationales

IMPORTANT: Your response MUST be valid JSON with the exact structure shown below.
Do not include any explanatory text outside the JSON.

{
    "requirement_completeness": [
        {
            "requirement_id": "R1",
            "requirement_text": "text of the requirement",
            "completeness_score": 75.0,
            "missing_elements": ["Element1", "Element2"],
            "suggested_improvement": "Text explaining how to improve this requirement",
            "rationale": "Explanation of why the requirement is incomplete"
        }
    ],
    "reasoning": "Your explanation of how you combined the requirement completeness assessments"
}

SOURCE REQUIREMENT COMPLETENESS ASSESSMENTS:
"""
            
            # Add each source's requirement completeness
            for i, completeness_data in enumerate(all_completeness):
                model_id = completeness_data["model_id"]
                completeness = completeness_data["requirement_completeness"]
                completeness_json = json.dumps(completeness, indent=2)
                prompt += f"\n\nREQUIREMENT COMPLETENESS {i+1} (from {model_id}):\n{completeness_json}\n"
            
            # Get the LLM adapter
            adapter = get_adapter(meta_model_id)
            
            # Generate the response
            messages = [
                {"role": "system", "content": "You are an expert requirements analyst specializing in assessing requirement completeness. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            response = adapter.generate_response(messages)
            
            # Extract and validate the JSON
            result_json = response["content"]
            result = extract_json_from_response(result_json)
            
            if not result or "requirement_completeness" not in result:
                logger.warning(f"Failed to extract valid requirement completeness from {meta_model_id} response")
                # Fall back to direct merging for this component
                return self._merge_all_requirement_completeness(analysis_results), f"Used direct merging due to issues with {meta_model_id} response"
            
            reasoning = result.get("reasoning", "Successfully aggregated requirement completeness using specialized prompt")
            return result["requirement_completeness"], reasoning
            
        except Exception as e:
            logger.error(f"Error in requirement completeness aggregation: {str(e)}")
            # Fall back to direct merging for this component
            return self._merge_all_requirement_completeness(analysis_results), f"Used direct merging due to error: {str(e)}"