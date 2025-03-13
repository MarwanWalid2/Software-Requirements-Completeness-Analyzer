import os
import sys
import uuid
import logging
import traceback
from datetime import datetime
import json
import tempfile
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

from config import configure_app
from models.domain_model_analyzer import DomainModelAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r"log\app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
configure_app(app)
Session(app)

# Initialize the analyzer
analyzer = DomainModelAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_requirements():
    try:
        logger.info("Received analyze request")
        requirements = request.json.get('requirements', '')
        
        if not requirements:
            logger.warning("No requirements provided in request")
            return jsonify({"error": "No requirements provided"}), 400
        
        logger.info(f"Processing requirements ({len(requirements)} characters)")
        
        # Generate the domain model
        logger.info("Generating domain model...")
        try:
            domain_model_result = analyzer.create_domain_model(requirements)
        except Exception as e:
            logger.error(f"Error generating domain model: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": f"Domain model generation failed: {str(e)}",
                "traceback": traceback.format_exc()
            }), 500
        
        if "error" in domain_model_result and not domain_model_result.get("domain_model"):
            logger.error(f"Error in domain model result: {domain_model_result['error']}")
            return jsonify(domain_model_result), 500
        
        domain_model = domain_model_result.get("domain_model")
        if not domain_model:
            logger.error("Domain model is empty or invalid")
            # Create a minimal valid model
            domain_model = {
                "classes": [],
                "relationships": [],
                "plantuml": "@startuml\n@enduml"
            }
        
        # Generate UML diagram image
        logger.info("Generating PlantUML diagram...")
        plantuml_code = domain_model.get("plantuml", "")
        if not plantuml_code:
            logger.warning("PlantUML code is empty, using default")
            plantuml_code = "@startuml\n@enduml"
        
        uml_image = analyzer.generate_plantUML_image(plantuml_code)
        if not uml_image:
            logger.warning("Failed to generate UML image, using fallback")
        
        # Analyze completeness (includes both original functionality and enhanced analysis)
        logger.info("Analyzing requirements completeness...")
        try:
            analysis_result = analyzer.analyze_requirements_completeness(requirements, domain_model)
        except Exception as e:
            logger.error(f"Error analyzing requirements: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a minimal valid analysis
            analysis_result = {
                "analysis": {
                    "requirement_issues": [],
                    "missing_requirements": [],
                    "domain_model_issues": [],
                    "requirement_completeness": []
                },
                "error": f"Requirements analysis failed: {str(e)}",
                "reasoning": "Analysis could not be completed due to API errors"
            }
        
        # Store in session
        try:
            session['domain_model'] = domain_model
            session['analysis'] = analysis_result.get("analysis", {})
            session['requirements'] = requirements
        except Exception as e:
            logger.warning(f"Could not store results in session: {str(e)}")
        
        # Prepare response
        response = {
            "domain_model": domain_model,
            "analysis": analysis_result.get("analysis", {}),
            "uml_image": uml_image,
            "reasoning": {
                "domain_model": domain_model_result.get("reasoning", ""),
                "analysis": analysis_result.get("reasoning", "")
            },
            "debug_info": {
                "api_key_present": bool(analyzer.client.api_key),
                "requirements_length": len(requirements),
                "domain_model_present": bool(domain_model),
                "uml_image_present": bool(uml_image),
                "analysis_present": bool(analysis_result.get("analysis"))
            }
        }
        
        # Save the results to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
        try:
            with open(filename, "w") as f:
                json.dump(response, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Could not save results to file: {str(e)}")
        
        logger.info("Analysis completed successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.critical(f"Unhandled exception in analyze endpoint: {str(e)}")
        logger.critical(traceback.format_exc())
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/update', methods=['POST'])
def update_model_and_requirements():
    """API endpoint to accept/reject/edit changes"""
    try:
        logger.info("Received update request")
        
        # Get request data
        data = request.json
        accepted_changes = data.get('accepted_changes', [])
        edited_requirements = data.get('edited_requirements', [])
        
        if not accepted_changes and not edited_requirements:
            logger.warning("No changes provided in update request")
            return jsonify({"error": "No changes provided"}), 400
            
        # Get current domain model and requirements from session
        domain_model = session.get('domain_model')
        requirements = session.get('requirements')
        
        if not domain_model or not requirements:
            logger.error("No domain model or requirements in session")
            return jsonify({"error": "No current analysis session found"}), 400
        
        # Update domain model based on accepted changes
        if accepted_changes:
            logger.info(f"Updating domain model with {len(accepted_changes)} accepted changes")
            updated_domain_model = analyzer.update_domain_model(domain_model, accepted_changes)
        else:
            updated_domain_model = domain_model
            
        # Update requirements text if there are edited requirements
        updated_requirements = requirements
        if edited_requirements:
            logger.info(f"Updating requirements with {len(edited_requirements)} edits")
            
            # Split requirements by line for easier manipulation
            req_lines = requirements.split('\n')
            
            # Apply each edit
            for edit in edited_requirements:
                req_id = edit.get('id')
                new_text = edit.get('text')
                
                # Find the requirement in the text (simple approach, might need refinement)
                for i, line in enumerate(req_lines):
                    if req_id in line:
                        req_lines[i] = new_text
                        break
                        
            # Join back into a single string
            updated_requirements = '\n'.join(req_lines)
            
        # Generate updated UML diagram
        logger.info("Generating updated PlantUML diagram...")
        plantuml_code = updated_domain_model.get("plantuml", "")
        if not plantuml_code:
            plantuml_code = "@startuml\n@enduml"
            
        uml_image = analyzer.generate_plantUML_image(plantuml_code)
        
        # Store updated model and requirements in session
        session['domain_model'] = updated_domain_model
        session['requirements'] = updated_requirements
        
        # Only perform targeted analysis on changes, not a full re-analysis
        logger.info("Performing targeted analysis on updated model...")
        try:
            # Get current analysis and update only what's needed
            current_analysis = session.get('analysis', {})
            
            # Remove items that have been accepted and addressed
            # For example, remove accepted missing requirements from the list
            if 'missing_requirements' in current_analysis:
                accepted_ids = [change['id'] for change in accepted_changes if change['type'] == 'missing_requirement']
                current_analysis['missing_requirements'] = [req for req in current_analysis['missing_requirements'] 
                                                           if req.get('id') not in accepted_ids]
            
            # Similarly handle other accepted changes
            # For requirement improvements
            if 'requirement_completeness' in current_analysis:
                accepted_ids = [change['requirement_id'] for change in accepted_changes if change['type'] == 'requirement_improvement']
                current_analysis['requirement_completeness'] = [req for req in current_analysis['requirement_completeness'] 
                                                                if req.get('requirement_id') not in accepted_ids]
            
            # For model issue fixes
            if 'domain_model_issues' in current_analysis:
                accepted_model_issues = [(change['element_name'], change['issue_type']) for change in accepted_changes 
                                         if change['type'] == 'model_issue_fix']
                current_analysis['domain_model_issues'] = [issue for issue in current_analysis['domain_model_issues'] 
                                                          if (issue.get('element_name'), issue.get('issue_type')) not in accepted_model_issues]
            
            # For requirement issue fixes
            if 'requirement_issues' in current_analysis:
                # This is more complex as we need to remove specific issues from requirements
                for change in accepted_changes:
                    if change['type'] == 'requirement_issue_fix':
                        req_id = change['requirement_id']
                        issue_type = change['issue_type']
                        
                        for req in current_analysis['requirement_issues']:
                            if req.get('requirement_id') == req_id and 'issues' in req:
                                req['issues'] = [issue for issue in req['issues'] if issue.get('issue_type') != issue_type]
            
            # Store updated analysis
            session['analysis'] = current_analysis
            analysis_result = {"analysis": current_analysis}
            
        except Exception as e:
            logger.error(f"Error updating analysis: {str(e)}")
            analysis_result = {
                "analysis": session.get('analysis', {})
            }
            
        # Prepare response
        response = {
            "domain_model": updated_domain_model,
            "requirements": updated_requirements,
            "analysis": analysis_result.get("analysis", {}),
            "uml_image": uml_image,
            "success": True,
            "message": "Model and requirements updated successfully"
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.critical(f"Unhandled exception in update endpoint: {str(e)}")
        logger.critical(traceback.format_exc())
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "traceback": traceback.format_exc(),
            "success": False
        }), 500

if __name__ == '__main__':
    app.run(debug=True)