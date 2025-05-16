/**
 * File Upload Module
 * Handles SRS document uploads and requirement extraction
 */
class FileUploadHandler {
    constructor() {
        // DOM Elements
        this.fileInput = document.getElementById('srs-file-upload');
        this.extractCheckbox = document.getElementById('extract-requirements-checkbox');
        this.requirementsEditor = document.getElementById('requirements-editor');
        this.fileInfoDisplay = document.getElementById('file-info-display');
        
        // Make sure we have access to these elements
        if (!this.fileInput) console.error("File input element not found!");
        if (!this.extractCheckbox) console.error("Extract checkbox not found!");
        if (!this.requirementsEditor) console.error("Requirements editor not found!");
        
        // State
        this.currentFile = null;
        this.uploadedFileContent = null;
        this.processingFile = false;
        
        // Initialize
        this.init();
    }
    
    /**
     * Initialize the component
     */
    init() {
        console.log("Initializing FileUploadHandler");
        // Add event listeners
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => {
                console.log("File input change event triggered");
                this.handleFileSelection(e);
            });
        }
    }
    
    /**
     * Handle file selection
     */
    handleFileSelection(event) {
        console.log("File selection handler called");
        
        // Get file either from event or directly from input
        let file = null;
        if (event && event.target && event.target.files && event.target.files.length > 0) {
            file = event.target.files[0];
            console.log("File from event:", file.name, file.type, file.size);
        } else if (this.fileInput && this.fileInput.files && this.fileInput.files.length > 0) {
            file = this.fileInput.files[0];
            console.log("File from input element:", file.name, file.type, file.size);
        }
        
        if (!file) {
            console.log("No file selected");
            if (this.fileInfoDisplay) this.fileInfoDisplay.style.display = 'none';
            this.currentFile = null;
            return;
        }
        
        console.log("File selected:", file.name, file.type, file.size);
        this.currentFile = file;
        
        // Update file info display if it exists
        if (this.fileInfoDisplay) {
            const fileName = this.fileInfoDisplay.querySelector('.file-name');
            const fileSize = this.fileInfoDisplay.querySelector('.file-size');
            
            if (fileName) fileName.textContent = file.name;
            if (fileSize) fileSize.textContent = this.formatFileSize(file.size);
            this.fileInfoDisplay.style.display = 'flex';
        }
        
        // Show notification
        this.showNotification(`File selected: ${file.name}. Click "Analyze Requirements" to process.`, 'info');
    }
    
    /**
     * Format file size to human-readable format
     */
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }
    
    /**
 * Process the file during analysis
 * This will be called by the main app before analysis
 */
async processFileIfNeeded() {
    if (!this.currentFile || this.processingFile) {
        return null;
    }
    
    this.processingFile = true;
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', this.currentFile);
        formData.append('extract_requirements', 'true');
        
        // Get selected models from model selector
        if (window.modelSelector) {
            const selectedModels = window.modelSelector.getSelectedModels();
            for (const model of selectedModels) {
                formData.append('selected_models[]', model);
            }
            formData.append('meta_model_id', window.modelSelector.getCurrentMetaModel());
        }
        
        // Show notification
        this.showNotification(`Processing file: ${this.currentFile.name}...`, 'info');
        
        // Upload file and start processing
        const response = await fetch('/api/upload-srs', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to process file');
        }
        
        // Get job ID
        const data = await response.json();
        const jobId = data.job_id;
        
        // Show progress notification
        this.showNotification(`File uploaded. Processing in background...`, 'info');
        
        // Poll for job completion
        const result = await this.pollJobStatus(jobId);
        
        // Store the original content
        this.uploadedFileContent = result.original_content;
        
        // Clear the text box first
        if (this.requirementsEditor) {
            this.requirementsEditor.value = '';
        }
        
        // Update the requirements editor with extracted requirements
        if (result.extracted_requirements) {
            if (this.requirementsEditor) {
                this.requirementsEditor.value = result.extracted_requirements;
            }
            this.showNotification(`Successfully extracted ${result.requirements_count} requirements from ${this.currentFile.name}`, 'success');
            return result.extracted_requirements;
        } else {
            if (this.requirementsEditor) {
                this.requirementsEditor.value = result.content || '';
            }
            this.showNotification(`File processed. Content loaded into editor.`, 'info');
            return result.content;
        }
        
    } catch (error) {
        console.error('Error processing file:', error);
        this.showNotification(`Error: ${error.message}`, 'danger');
        return null;
    } finally {
        this.processingFile = false;
    }
}

// Add a new method to poll for job status
async pollJobStatus(jobId, maxRetries = 60, retryInterval = 2000) {
    let retries = 0;
    
    // Create a progress indicator
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress mb-3';
    progressContainer.style.height = '20px';
    
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated';
    progressBar.style.width = '0%';
    progressBar.setAttribute('role', 'progressbar');
    progressBar.setAttribute('aria-valuenow', '0');
    progressBar.setAttribute('aria-valuemin', '0');
    progressBar.setAttribute('aria-valuemax', '100');
    
    progressContainer.appendChild(progressBar);
    
    // Add to page near the file upload section
    const fileUploadContainer = document.querySelector('.file-upload-container');
    if (fileUploadContainer) {
        fileUploadContainer.appendChild(progressContainer);
    }
    
    while (retries < maxRetries) {
        try {
            const response = await fetch(`/api/job-status/${jobId}`);
            
            if (!response.ok) {
                throw new Error('Failed to check job status');
            }
            
            const statusData = await response.json();
            
            // Update progress bar
            progressBar.style.width = `${statusData.progress}%`;
            progressBar.setAttribute('aria-valuenow', statusData.progress);
            progressBar.textContent = `${statusData.progress}%`;
            
            if (statusData.status === 'completed') {
                // Remove progress bar on completion
                if (fileUploadContainer && progressContainer) {
                    fileUploadContainer.removeChild(progressContainer);
                }
                return statusData.results;
            } else if (statusData.status === 'error') {
                // Remove progress bar on error
                if (fileUploadContainer && progressContainer) {
                    fileUploadContainer.removeChild(progressContainer);
                }
                throw new Error(statusData.error || 'Error processing file');
            }
            
            // Wait before the next poll
            await new Promise(resolve => setTimeout(resolve, retryInterval));
            retries++;
        } catch (error) {
            // Remove progress bar on error
            if (fileUploadContainer && progressContainer) {
                fileUploadContainer.removeChild(progressContainer);
            }
            throw error;
        }
    }
    
    // Remove progress bar on timeout
    if (fileUploadContainer && progressContainer) {
        fileUploadContainer.removeChild(progressContainer);
    }
    throw new Error('Job processing timed out');
}
    
    /**
     * Check if there's a file selected
     */
    hasFileSelected() {
        // Log detailed debugging information
        console.log("Checking if file is selected");
        console.log("Current file state:", this.currentFile);
        console.log("File input element:", this.fileInput);
        
        if (this.fileInput && this.fileInput.files) {
            console.log("Files in input:", this.fileInput.files.length);
            for (let i = 0; i < this.fileInput.files.length; i++) {
                console.log(`File ${i}:`, this.fileInput.files[i].name);
            }
        }
        
        // Check if we have a file in the currentFile property
        if (this.currentFile) {
            console.log("Using currentFile reference:", this.currentFile.name);
            return true;
        }
        
        // As a fallback, also check the file input element directly
        if (this.fileInput && this.fileInput.files && this.fileInput.files.length > 0) {
            // Update our currentFile reference if we find a file here
            this.currentFile = this.fileInput.files[0];
            console.log("Found file in input element:", this.currentFile.name);
            return true;
        }
        
        console.log("No file found");
        return false;
    }
    
    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        console.log(`Notification (${type}): ${message}`);
        
        // Use the app's notification system if available
        if (window.app && typeof window.app.showNotification === 'function') {
            window.app.showNotification(message, type);
        } else {
            console.log(`${type}: ${message}`);
            // Fallback: simple alert for critical messages
            if (type === 'danger' || type === 'warning') {
                alert(message);
            }
        }
    }
    
    /**
     * Get the original uploaded file content
     */
    getOriginalContent() {
        return this.uploadedFileContent;
    }
}

// Initialize the file upload handler and make it available globally
console.log("Creating FileUploadHandler instance");
window.fileUploadHandler = new FileUploadHandler();