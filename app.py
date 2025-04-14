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

from config import configure_app, get_available_models, get_available_meta_models
from models.domain_model_analyzer import DomainModelAnalyzer
from werkzeug.utils import secure_filename
import re 

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(os.path.join("log", "app.log")),
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
    """Render the main page with LLM model selection options"""
    # Get available models and meta-models
    available_models = get_available_models()
    available_meta_models = get_available_meta_models()
    
    # Store in session for future use
    session['available_models'] = available_models
    session['available_meta_models'] = available_meta_models
    
    return render_template(
        'index.html', 
        available_models=available_models,
        available_meta_models=available_meta_models
    )

@app.route('/api/available-models', methods=['GET'])
def get_models():
    """Return the available models and meta-models"""
    available_models = get_available_models()
    available_meta_models = get_available_meta_models()
    
    return jsonify({
        "models": available_models,
        "meta_models": available_meta_models
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_requirements():
    try:
        logger.info("Received analyze request")
        data = request.json
        requirements = data.get('requirements', '')
        
        # Get selected models from request or use default
        selected_models = data.get('selected_models', ['deepseek'])
        meta_model_id = data.get('meta_model_id', 'majority')
        model_weights = data.get('model_weights', {}) 

        
        logger.info(f"Selected models: {selected_models}")
        logger.info(f"Meta model: {meta_model_id}")
        
        if not requirements:
            logger.warning("No requirements provided in request")
            return jsonify({"error": "No requirements provided"}), 400
        
        logger.info(f"Processing requirements ({len(requirements)} characters)")
        logger.info("Generating domain model...")
        try:
            domain_model_result = analyzer.create_domain_model(
                requirements, 
                selected_models=selected_models,
                meta_model_id=meta_model_id,
                model_weights=model_weights
            )
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
            analysis_result = analyzer.analyze_requirements_completeness(
                requirements, 
                domain_model,
                selected_models=selected_models,
                meta_model_id=meta_model_id,
                model_weights=model_weights
            )
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
            session['selected_models'] = selected_models
            session['meta_model_id'] = meta_model_id
            session['model_weights'] = model_weights
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
            "aggregation_info": {
                "domain_model": domain_model_result.get("aggregation_info", {}),
                "analysis": analysis_result.get("aggregation_info", {})
            },
            "debug_info": {
                "selected_models": selected_models,
                "meta_model_id": meta_model_id,
                "requirements_length": len(requirements),
                "domain_model_present": bool(domain_model),
                "uml_image_present": bool(uml_image),
                "analysis_present": bool(analysis_result.get("analysis"))
            }
        }
        
        # # Save the results to a JSON file
        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # filename = f"analysis_results_{timestamp}.json"
        # try:
        #     with open(filename, "w") as f:
        #         json.dump(response, f, indent=2)
        #     logger.info(f"Results saved to {filename}")
        # except Exception as e:
        #     logger.error(f"Could not save results to file: {str(e)}")
        
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
        
        # Get selected models from session or request
        selected_models = data.get('selected_models') or session.get('selected_models', ['deepseek'])
        meta_model_id = data.get('meta_model_id') or session.get('meta_model_id', 'majority')
        model_weights = data.get('model_weights') or session.get('model_weights', {})
        
        logger.info(f"Selected models for update: {selected_models}")
        logger.info(f"Meta model for update: {meta_model_id}")
        logger.info(f"Accepted changes: {len(accepted_changes)}")
        
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
            updated_domain_model = analyzer.update_domain_model(
                domain_model, 
                accepted_changes,
                selected_models=selected_models,
                meta_model_id=meta_model_id,
                model_weights=model_weights
            )
        else:
            updated_domain_model = domain_model
        
        # Start with current requirements
        updated_requirements = requirements
        
        # Process all changes to requirements
        if accepted_changes or edited_requirements:
            # Split requirements by line for easier manipulation
            req_lines = requirements.split('\n')
            requirements_updated = False
            
            # First process accepted changes
            for change in accepted_changes:
                change_type = change.get('type')
                logger.info(f"Processing change of type: {change_type}")
                
                if change_type == 'missing_requirement':
                    # Extract existing requirement pattern
                    req_pattern = None
                    for line in req_lines:
                        match = re.match(r'^([A-Za-z]+-?)(\d+):', line)
                        if match:
                            prefix = match.group(1)
                            number = int(match.group(2))
                            req_pattern = (prefix, len(str(number)))
                            break
                    
                    if not req_pattern:
                        # Default pattern if none found
                        new_req_id = f"REQ-{len(req_lines) + 1:03d}"
                    else:
                        # Find highest ID number
                        prefix, padding = req_pattern
                        max_number = 0
                        for line in req_lines:
                            match = re.match(f'^{prefix}(\\d+):', line)
                            if match:
                                max_number = max(max_number, int(match.group(1)))
                        
                        # Generate new ID with same pattern
                        new_req_id = f"{prefix}{(max_number + 1):0{padding}d}"
                    
                    # Get the correct suggested text field
                    suggested_text = change.get('suggested_text', '')
                    
                    # Add the new requirement
                    if suggested_text:
                        if suggested_text.startswith(new_req_id):
                            new_req = suggested_text
                        else:
                            new_req = f"{new_req_id}: {suggested_text}"
                        
                        req_lines.append(new_req)
                        logger.info(f"Added missing requirement: {new_req}")
                        requirements_updated = True
                    
                elif change_type == 'requirement_issue_fix':
                    req_id = change.get('requirement_id')
                    # For issue fixes, the text is in suggested_fix field
                    suggested_text = change.get('suggested_fix', '')
                    
                    if req_id and suggested_text:
                        # Find the requirement in the text
                        for i, line in enumerate(req_lines):
                            if line.startswith(f"{req_id}:") or line.startswith(f"{req_id} "):
                                # Extract id format to maintain consistency
                                id_part = line.split(':', 1)[0] if ':' in line else req_id
                                req_lines[i] = f"{id_part}: {suggested_text}"
                                logger.info(f"Updated requirement {req_id} with fix")
                                requirements_updated = True
                                break
                    
                elif change_type == 'requirement_improvement':
                    req_id = change.get('requirement_id')
                    # For improvements, the text might be in suggested_improvement
                    suggested_text = change.get('suggested_text', '') or change.get('suggested_improvement', '')
                    
                    if req_id and suggested_text:
                        # Find the requirement in the text
                        for i, line in enumerate(req_lines):
                            if line.startswith(f"{req_id}:") or line.startswith(f"{req_id} "):
                                # Extract id format to maintain consistency
                                id_part = line.split(':', 1)[0] if ':' in line else req_id
                                req_lines[i] = f"{id_part}: {suggested_text}"
                                logger.info(f"Updated requirement {req_id} with improvement")
                                requirements_updated = True
                                break
                
                elif change_type == 'model_issue_fix':
                    # Model issue fixes don't affect requirements text
                    pass
            
            # Then process edited requirements
            if edited_requirements:
                logger.info(f"Processing {len(edited_requirements)} edited requirements")
                for edit in edited_requirements:
                    req_id = edit.get('id')
                    new_text = edit.get('text')
                    
                    if req_id and new_text:
                        # Find the requirement in the text
                        for i, line in enumerate(req_lines):
                            if line.startswith(f"{req_id}:") or line.startswith(f"{req_id} ") or req_id in line:
                                req_lines[i] = new_text
                                logger.info(f"Applied manual edit to requirement {req_id}")
                                requirements_updated = True
                                break
            
            # Only join and update if changes were made
            if requirements_updated:
                updated_requirements = '\n'.join(req_lines)
                logger.info("Requirements text was updated")
        
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
            "requirements": updated_requirements,  # Return the updated requirements
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

@app.route('/api/upload-srs', methods=['POST'])
def upload_srs_file():
    """API endpoint to upload and process SRS documents"""
    try:
        logger.info("Received SRS file upload request")
        
        # Check if a file was uploaded
        if 'file' not in request.files:
            logger.warning("No file part in request")
            logger.debug(f"Request files: {request.files}")
            logger.debug(f"Request form: {request.form}")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Check if the file was actually selected
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400
        
        logger.info(f"Processing file: {file.filename}, size: {file.content_length or 'unknown'}, type: {file.content_type}")
        
        # Always extract requirements
        extract_requirements = True
        logger.info(f"Always extracting requirements from document")
        
        # Get selected models from the request
        selected_models = request.form.getlist('selected_models[]')
        if not selected_models:
            selected_models = ['claude']  # Default
        logger.info(f"Selected models for extraction: {selected_models}")
        
        # Get meta model
        meta_model_id = request.form.get('meta_model_id', 'majority')
        logger.info(f"Meta model for extraction: {meta_model_id}")
        
        # Save the file to a temporary location
        temp_filepath = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(temp_filepath)
        logger.info(f"File saved to {temp_filepath}")
        
        # Process the file based on its extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        logger.info(f"File extension: {file_extension}")
        content = ""
        
        if file_extension in ['.txt', '.md']:
            # Plain text files
            logger.info("Processing as text file")
            with open(temp_filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            logger.info(f"Read text file ({len(content)} characters)")
        
        elif file_extension in ['.docx']:
            # Word documents using docx library
            logger.info("Processing as DOCX file")
            try:
                import docx
                doc = docx.Document(temp_filepath)
                paragraphs = [p.text for p in doc.paragraphs]
                content = '\n'.join(paragraphs)
                logger.info(f"Processed DOCX file ({len(content)} characters)")
            except ImportError:
                # If docx is not installed, return error
                logger.error("python-docx library not installed")
                return jsonify({"error": "Cannot process DOCX files. The python-docx library is not installed."}), 500
        
        elif file_extension in ['.pdf']:
            # PDF files using pypdf
            logger.info("Processing as PDF file")
            try:
                import pypdf
                logger.info("Opening PDF file with pypdf")
                pdf_reader = pypdf.PdfReader(temp_filepath)
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                
                content = ""
                for i, page in enumerate(pdf_reader.pages):
                    logger.info(f"Extracting text from page {i+1}/{len(pdf_reader.pages)}")
                    page_text = page.extract_text()
                    logger.info(f"Extracted {len(page_text)} characters from page {i+1}")
                    content += page_text + "\n"
                
                if not content.strip():
                    logger.warning("PDF extraction returned empty content")
                    # If we got empty content, the PDF might be image-based
                    # We could add OCR here in a future enhancement
                    return jsonify({
                        "error": "The PDF appears to be image-based or doesn't contain extractable text. Please convert it to a text-based format or manually copy the requirements."
                    }), 400
                
                logger.info(f"Successfully processed PDF with total {len(content)} characters")
            except ImportError:
                logger.error("pypdf library not installed")
                return jsonify({
                    "error": "Cannot process PDF files. The pypdf library is not installed."
                }), 500
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    "error": f"Error processing PDF: {str(e)}"
                }), 500
        
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return jsonify({"error": f"Unsupported file format: {file_extension}"}), 400
        
        # Clean up the temporary file
        try:
            os.remove(temp_filepath)
            logger.info(f"Removed temporary file: {temp_filepath}")
        except Exception as e:
            logger.warning(f"Could not remove temporary file: {str(e)}")
        
        # Store the original content in the session
        session['uploaded_srs_content'] = content
        logger.info("Stored original content in session")
        
        # If extraction is requested, extract requirements using LLMs
        if extract_requirements:
            logger.info("Extracting requirements using LLMs")
            
            # Extract requirements using the domain model analyzer
            try:
                extraction_result = analyzer.extract_requirements_from_srs(
                    content,
                    selected_models=selected_models,
                    meta_model_id=meta_model_id
                )
                
                extracted_requirements = extraction_result.get("extracted_requirements", "")
                requirements_count = extraction_result.get("requirements_count", 0)
                
                # Extract context information as well
                context_result = analyzer.extract_context_from_srs(
                    content,
                    selected_models=selected_models,
                    meta_model_id=meta_model_id
                )
                
                # Store in session for future use
                session['requirements'] = extracted_requirements
                session['document_context'] = context_result
                
                return jsonify({
                    "success": True,
                    "original_content": content,
                    "extracted_requirements": extracted_requirements,
                    "requirements_count": requirements_count,
                    "context": context_result,
                    "message": f"Successfully extracted {requirements_count} requirements from the document."
                })
                
            except Exception as e:
                logger.error(f"Error extracting requirements: {str(e)}")
                logger.error(traceback.format_exc())
                # Fall back to returning the raw content
                return jsonify({
                    "success": False,
                    "original_content": content,
                    "content": content,
                    "error": f"Could not extract requirements: {str(e)}",
                    "message": "Failed to extract requirements. Returning the original document content."
                })
        
        # If no extraction requested, just return the content
        logger.info("No extraction requested, returning original content")
        session['requirements'] = content
        
        return jsonify({
            "success": True,
            "original_content": content,
            "content": content,
            "message": "File uploaded successfully."
        })
        
    except Exception as e:
        logger.critical(f"Unhandled exception in upload endpoint: {str(e)}")
        logger.critical(traceback.format_exc())
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    if os.environ.get('FLASK_CONFIG') == 'production':
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
    app.run(host='0.0.0.0', port=port)