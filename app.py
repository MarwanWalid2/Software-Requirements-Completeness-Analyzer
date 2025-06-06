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
os.environ["GUNICORN_CMD_ARGS"] = "--timeout=250 --keep-alive=5 --graceful-timeout=120"

# Initialize Flask app
app = Flask(__name__)
configure_app(app)
Session(app)

# Initialize the analyzer
analyzer = DomainModelAnalyzer()

import threading
import uuid
from collections import defaultdict

# Add a job storage (replace with Redis in production)
job_store = {
    'status': {},      # job_id -> 'pending', 'processing', 'completed', 'error'
    'results': {},     # job_id -> result data
    'errors': {},      # job_id -> error message
    'progress': defaultdict(int)  # job_id -> progress percentage
}

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
        
        # Create job ID and set initial status
        job_id = str(uuid.uuid4())
        job_store['status'][job_id] = 'pending'
        job_store['progress'][job_id] = 0
        
        # Start a background thread for analysis
        thread = threading.Thread(
            target=analyze_requirements_async,
            args=(job_id, requirements, selected_models, meta_model_id, model_weights)
        )
        thread.daemon = True
        thread.start()
        
        # Return immediately with the job ID
        return jsonify({
            "success": True,
            "message": "Analysis started in background",
            "job_id": job_id
        })
        
    except Exception as e:
        logger.critical(f"Unhandled exception in analyze endpoint: {str(e)}")
        logger.critical(traceback.format_exc())
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

# Function to perform analysis asynchronously
def analyze_requirements_async(job_id, requirements, selected_models, meta_model_id, model_weights):
    try:
        # Update job status
        job_store['status'][job_id] = 'processing'
        job_store['progress'][job_id] = 10
        
        # Generate domain model
        logger.info("Generating domain model...")
        try:
            job_store['progress'][job_id] = 20
            domain_model_result = analyzer.create_domain_model(
                requirements, 
                selected_models=selected_models,
                meta_model_id=meta_model_id,
                model_weights=model_weights
            )
        except Exception as e:
            logger.error(f"Error generating domain model: {str(e)}")
            logger.error(traceback.format_exc())
            job_store['status'][job_id] = 'error'
            job_store['errors'][job_id] = f"Domain model generation failed: {str(e)}"
            return
        
        if "error" in domain_model_result and not domain_model_result.get("domain_model"):
            logger.error(f"Error in domain model result: {domain_model_result['error']}")
            job_store['status'][job_id] = 'error'
            job_store['errors'][job_id] = domain_model_result['error']
            return
        
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
        job_store['progress'][job_id] = 40
        logger.info("Generating PlantUML diagram...")
        plantuml_code = domain_model.get("plantuml", "")
        if not plantuml_code:
            logger.warning("PlantUML code is empty, using default")
            plantuml_code = "@startuml\n@enduml"
        
        uml_image = analyzer.generate_plantUML_image(plantuml_code)
        if not uml_image:
            logger.warning("Failed to generate UML image, using fallback")
        
        # Analyze completeness
        job_store['progress'][job_id] = 60
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
        
        # Removed session storage for async processing
        
        # Prepare response
        job_store['progress'][job_id] = 90
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
        
        # Store results
        job_store['results'][job_id] = response
        job_store['status'][job_id] = 'completed'
        job_store['progress'][job_id] = 100
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.critical(f"Unhandled exception in async analysis: {str(e)}")
        logger.critical(traceback.format_exc())
        job_store['status'][job_id] = 'error'
        job_store['errors'][job_id] = f"An unexpected error occurred: {str(e)}"

@app.route('/api/update', methods=['POST'])
def update_model_and_requirements():
    """API endpoint to accept/reject/edit changes - async version"""
    try:
        logger.info("Received update request")
        
        # Get request data
        data = request.json
        accepted_changes = data.get('accepted_changes', [])
        edited_requirements = data.get('edited_requirements', [])
        
        # Get selected models from request
        selected_models = data.get('selected_models', ['deepseek'])
        meta_model_id = data.get('meta_model_id', 'majority')
        model_weights = data.get('model_weights', {})
        
        # Retrieve requirements and domain model
        # For async version, we need to pass these in the request
        domain_model = data.get('domain_model')
        requirements = data.get('requirements')
        
        if not domain_model and not requirements:
            logger.error("No domain model or requirements provided")
            return jsonify({"error": "No domain model or requirements provided"}), 400
            
        logger.info(f"Selected models for update: {selected_models}")
        logger.info(f"Meta model for update: {meta_model_id}")
        logger.info(f"Accepted changes: {len(accepted_changes)}")
        
        if not accepted_changes and not edited_requirements:
            logger.warning("No changes provided in update request")
            return jsonify({"error": "No changes provided"}), 400
        
        # Create job ID and set initial status
        job_id = str(uuid.uuid4())
        job_store['status'][job_id] = 'pending'
        job_store['progress'][job_id] = 0
        
        # Start background thread for update
        thread = threading.Thread(
            target=update_model_and_requirements_async,
            args=(job_id, domain_model, requirements, accepted_changes, edited_requirements, 
                  selected_models, meta_model_id, model_weights)
        )
        thread.daemon = True
        thread.start()
        
        # Return immediately with the job ID
        return jsonify({
            "success": True,
            "message": "Update started in background",
            "job_id": job_id
        })
        
    except Exception as e:
        logger.critical(f"Unhandled exception in update endpoint: {str(e)}")
        logger.critical(traceback.format_exc())
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "traceback": traceback.format_exc(),
            "success": False
        }), 500

# Function to perform update asynchronously
def update_model_and_requirements_async(job_id, domain_model, requirements, accepted_changes, 
                                        edited_requirements, selected_models, meta_model_id, model_weights):
    try:
        # Update job status
        job_store['status'][job_id] = 'processing'
        job_store['progress'][job_id] = 10
        
        # Update domain model based on accepted changes
        if accepted_changes:
            logger.info(f"Updating domain model with {len(accepted_changes)} accepted changes")
            job_store['progress'][job_id] = 20
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
        job_store['progress'][job_id] = 40
        if accepted_changes or edited_requirements:
            # Split requirements by line for easier manipulation
            req_lines = requirements.split('\n')
            requirements_updated = False
            
            # Process accepted changes - this is the same code as in your existing update function
            # I'll include it here for completeness but won't modify it
            
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
        job_store['progress'][job_id] = 60
        logger.info("Generating updated PlantUML diagram...")
        plantuml_code = updated_domain_model.get("plantuml", "")
        if not plantuml_code:
            plantuml_code = "@startuml\n@enduml"
            
        uml_image = analyzer.generate_plantUML_image(plantuml_code)
        
        # Analyze changes (simplified for async version)
        job_store['progress'][job_id] = 80
        
        # Prepare response
        response = {
            "domain_model": updated_domain_model,
            "requirements": updated_requirements,
            "uml_image": uml_image,
            "success": True,
            "message": "Model and requirements updated successfully"
        }
        
        # Store results
        job_store['results'][job_id] = response
        job_store['status'][job_id] = 'completed'
        job_store['progress'][job_id] = 100
        logger.info("Update completed successfully")
        
    except Exception as e:
        logger.critical(f"Unhandled exception in async update: {str(e)}")
        logger.critical(traceback.format_exc())
        job_store['status'][job_id] = 'error'
        job_store['errors'][job_id] = f"An unexpected error occurred: {str(e)}"

@app.route('/api/upload-srs', methods=['POST'])
def upload_srs_file():
    """API endpoint to upload SRS documents with async processing"""
    try:
        logger.info("Received SRS file upload request")
        
        # Check if a file was uploaded
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Check if the file was actually selected
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400
        
        logger.info(f"Processing file: {file.filename}, size: {file.content_length or 'unknown'}, type: {file.content_type}")
        
        # Save the file to a temporary location
        temp_filepath = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(temp_filepath)
        logger.info(f"File saved to {temp_filepath}")
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Get extraction parameters
        extract_requirements = True
        
        # Get selected models from the request
        selected_models = request.form.getlist('selected_models[]')
        if not selected_models:
            selected_models = ['claude']  # Default
        
        # Get meta model
        meta_model_id = request.form.get('meta_model_id', 'majority')
        
        # Set initial job status
        job_store['status'][job_id] = 'pending'
        job_store['progress'][job_id] = 0
        
        # Start a background thread to process the file
        thread = threading.Thread(
            target=process_srs_file_async,
            args=(job_id, temp_filepath, file.filename, extract_requirements, selected_models, meta_model_id)
        )
        thread.daemon = True
        thread.start()
        
        # Return immediately with the job ID
        return jsonify({
            "success": True,
            "message": "File upload started. Processing in background.",
            "job_id": job_id
        })
        
    except Exception as e:
        logger.critical(f"Unhandled exception in upload endpoint: {str(e)}")
        logger.critical(traceback.format_exc())
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500
    


# a new function to process the file asynchronously
def process_srs_file_async(job_id, temp_filepath, filename, extract_requirements, selected_models, meta_model_id):
    """Process SRS file in a background thread"""
    try:
        # Update job status
        job_store['status'][job_id] = 'processing'
        job_store['progress'][job_id] = 10
        
        # Process the file based on its extension
        file_extension = os.path.splitext(filename)[1].lower()
        logger.info(f"Processing file with extension: {file_extension}")
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
                error_msg = "Cannot process DOCX files. The python-docx library is not installed."
                logger.error(error_msg)
                job_store['status'][job_id] = 'error'
                job_store['errors'][job_id] = error_msg
                return
        
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
                    error_msg = "The PDF appears to be image-based or doesn't contain extractable text."
                    logger.warning(error_msg)
                    job_store['status'][job_id] = 'error'
                    job_store['errors'][job_id] = error_msg
                    return
                
                logger.info(f"Successfully processed PDF with total {len(content)} characters")
            except ImportError:
                error_msg = "Cannot process PDF files. The pypdf library is not installed."
                logger.error(error_msg)
                job_store['status'][job_id] = 'error'
                job_store['errors'][job_id] = error_msg
                return
            except Exception as e:
                error_msg = f"Error processing PDF: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                job_store['status'][job_id] = 'error'
                job_store['errors'][job_id] = error_msg
                return
        
        else:
            error_msg = f"Unsupported file format: {file_extension}"
            logger.warning(error_msg)
            job_store['status'][job_id] = 'error'
            job_store['errors'][job_id] = error_msg
            return
        
        # Clean up the temporary file
        try:
            os.remove(temp_filepath)
            logger.info(f"Removed temporary file: {temp_filepath}")
        except Exception as e:
            logger.warning(f"Could not remove temporary file: {str(e)}")
        
        # Store the original content
        job_store['progress'][job_id] = 30
        result = {
            "original_content": content,
            "content": content
        }
        
        # If extraction is requested, extract requirements using LLMs
        if extract_requirements:
            job_store['progress'][job_id] = 40
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
                
                job_store['progress'][job_id] = 70
                
                # Extract context information as well
                context_result = analyzer.extract_context_from_srs(
                    content,
                    selected_models=selected_models,
                    meta_model_id=meta_model_id
                )
                
                job_store['progress'][job_id] = 90
                
                # Update the result
                result.update({
                    "extracted_requirements": extracted_requirements,
                    "requirements_count": requirements_count,
                    "context": context_result,
                    "message": f"Successfully extracted {requirements_count} requirements from the document."
                })
                
            except Exception as e:
                logger.error(f"Error extracting requirements: {str(e)}")
                logger.error(traceback.format_exc())
                # Fall back to returning the raw content
                result.update({
                    "error": f"Could not extract requirements: {str(e)}",
                    "message": "Failed to extract requirements. Returning the original document content."
                })
        
        # Store the result and update job status
        job_store['results'][job_id] = result
        job_store['status'][job_id] = 'completed'
        job_store['progress'][job_id] = 100
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in async processing: {str(e)}")
        logger.error(traceback.format_exc())
        job_store['status'][job_id] = 'error'
        job_store['errors'][job_id] = str(e)

# a new endpoint to check job status
@app.route('/api/job-status/<job_id>', methods=['GET'])
def check_job_status(job_id):
    """API endpoint to check the status of an async job"""
    if job_id not in job_store['status']:
        return jsonify({"error": "Job not found"}), 404
    
    status = job_store['status'][job_id]
    progress = job_store['progress'][job_id]
    
    response = {
        "status": status,
        "progress": progress
    }
    
    if status == 'completed':
        response["results"] = job_store['results'][job_id]
    elif status == 'error':
        response["error"] = job_store['errors'].get(job_id, "Unknown error")
    
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    if os.environ.get('FLASK_CONFIG') == 'production':
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
    app.run(host='0.0.0.0', port=port)