/**
 * Main Application Module
 * Coordinates the model selector and results display components
 */
class RequirementsAnalyzer {
    constructor() {
        // Core UI elements
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.requirementsEditor = document.getElementById('requirements-editor');
        this.loader = document.getElementById('loader');
        this.updateModelBtn = document.getElementById('update-model-btn');
        
        // Component references
        this.modelSelector = window.modelSelector; // From modelSelector.js
        this.resultsDisplay = window.resultsDisplay; // From resultsDisplay.js
        if (!this.analyzeBtn) {
            console.error("Analyze button not found!");
        }
        if (!this.loader) {
            console.error("Loader element not found!");
        }
        // Initialize
        this.init();
    }
    
    /**
     * Initialize the application
     */
    init() {
        // Add event listeners
        this.analyzeBtn.addEventListener('click', () => this.handleAnalyze());
        this.updateModelBtn.addEventListener('click', () => this.handleUpdateModel());
        
        // Set up any sample requirements for testing/demo
        this.setupDemoContent();
    }
    
/**
 * Handle Analyze button click
 */
async handleAnalyze() {
    console.log("Analyze button clicked");
    
    // Check for file upload handler
    if (!window.fileUploadHandler) {
        console.error("File upload handler not available!");
    } else {
        console.log("File upload handler found");
        console.log("Has file selected:", window.fileUploadHandler.hasFileSelected());
    }
    
    // Set loading at the beginning
    this.setLoading(true);
    
    // Check if there's a file to process first
    let requirements = "";
    let fileProcessed = false;
    
    try {
        if (window.fileUploadHandler && window.fileUploadHandler.hasFileSelected()) {
            console.log("File selected, processing it...");
            
            // Process the file first
            requirements = await window.fileUploadHandler.processFileIfNeeded();
            console.log("File processing complete, got requirements:", requirements ? "yes" : "no");
            fileProcessed = true;
            
            // If processing succeeded but returned nothing, use the editor content
            if (!requirements) {
                console.log("No requirements from file processing, using editor content");
                requirements = this.requirementsEditor.value.trim();
            }
        } else {
            // No file, just use the editor content
            console.log("No file selected, using editor content");
            requirements = this.requirementsEditor.value.trim();
        }
        
        // Check for empty requirements (only if no file was processed)
        if (!fileProcessed && !requirements) {
            console.log("No requirements provided and no file processed");
            this.showNotification('Please enter some requirements or upload an SRS document.', 'warning');
            this.setLoading(false);
            return;
        }
        
        // Get selected models
        const selectedModels = this.modelSelector.getSelectedModels();
        if (selectedModels.length === 0) {
            console.log("No models selected");
            this.showNotification('Please select at least one LLM model.', 'warning');
            this.setLoading(false);
            return;
        }
        
        console.log("Selected models:", selectedModels);
        
        // Get selected meta-model and weights
        const metaModel = this.modelSelector.getCurrentMetaModel();
        const modelWeights = this.modelSelector.getModelWeights();
        
        console.log("Sending analysis request with requirements length:", requirements.length);
        
        // Send the analysis request
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                requirements,
                selected_models: selectedModels,
                meta_model_id: metaModel,
                model_weights: modelWeights
            })
        });
        
        console.log("Analysis response received:", response.status);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to analyze requirements');
        }
        
        const data = await response.json();
        console.log("Analysis completed successfully");
        
        // Display results
        this.resultsDisplay.displayResults(data);
    } catch (error) {
        console.error('Error in analysis process:', error);
        this.showNotification(`Error: ${error.message}`, 'danger');
    } finally {
        this.setLoading(false);
    }
}
    
   /**
 * Handle Update Model button click
 */
async handleUpdateModel() {
    const acceptedChanges = this.resultsDisplay.getAcceptedChanges();
    const editedRequirements = this.resultsDisplay.getEditedRequirements();
    
    if (acceptedChanges.length === 0 && editedRequirements.length === 0) {
        this.showNotification('No changes to apply.', 'warning');
        return;
    }
    
    // Get selected models
    const selectedModels = this.modelSelector.getSelectedModels();
    if (selectedModels.length === 0) {
        selectedModels.push('deepseek'); // Fallback
    }
    
    // Get selected meta-model and weights
    const metaModel = this.modelSelector.getCurrentMetaModel();
    const modelWeights = this.modelSelector.getModelWeights();
    
    // Show loader
    this.setLoading(true);
    
    try {
        const response = await fetch('/api/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                accepted_changes: acceptedChanges,
                edited_requirements: editedRequirements,
                selected_models: selectedModels,
                meta_model_id: metaModel,
                model_weights: modelWeights
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to update model');
        }
        
        const data = await response.json();
        
        if (data.success) {
            this.showNotification('Domain model updated successfully!', 'success');
            
            // Reset changes tracking
            this.resultsDisplay.reset();
            
            // Update display with new results
            this.resultsDisplay.displayResults(data);
            
            // Update the requirements text box with the updated requirements
            if (data.requirements) {
                this.requirementsEditor.value = data.requirements;
            }
        } else {
            this.showNotification(`Error: ${data.error || 'Failed to update model'}`, 'danger');
        }
    } catch (error) {
        console.error('Error updating model:', error);
        this.showNotification(`Error: ${error.message}`, 'danger');
    } finally {
        this.setLoading(false);
    }
}
    
    /**
     * Set loading state
     */
    setLoading(isLoading) {
        if (isLoading) {
            this.loader.style.display = 'flex';
            this.analyzeBtn.disabled = true;
            this.analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        } else {
            this.loader.style.display = 'none';
            this.analyzeBtn.disabled = false;
            this.analyzeBtn.innerHTML = '<i class="fas fa-brain me-2"></i>Analyze Requirements';
        }
    }
    
    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        // Check if we already have a notification container
        let notificationContainer = document.querySelector('.notification-container');
        
        if (!notificationContainer) {
            // Create a container for notifications
            notificationContainer = document.createElement('div');
            notificationContainer.className = 'notification-container';
            document.body.appendChild(notificationContainer);
            
            // Add some basic styling
            const style = document.createElement('style');
            style.textContent = `
                .notification-container {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 9999;
                    max-width: 350px;
                }
                
                .notification {
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 4px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    color: white;
                    animation: slideIn 0.3s ease-out forwards;
                }
                
                .notification-success {
                    background-color: #28a745;
                }
                
                .notification-info {
                    background-color: #17a2b8;
                }
                
                .notification-warning {
                    background-color: #ffc107;
                    color: #212529;
                }
                
                .notification-danger {
                    background-color: #dc3545;
                }
                
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                
                @keyframes fadeOut {
                    from {
                        opacity: 1;
                    }
                    to {
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        // Create the notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add to container
        notificationContainer.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'fadeOut 0.5s forwards';
            setTimeout(() => {
                notificationContainer.removeChild(notification);
            }, 500);
        }, 5000);
    }
    
    /**
     * Set up demo content (for testing/demonstration purposes)
     */
    setupDemoContent() {
        // Uncomment to add sample requirements for demo
        
//         if (!this.requirementsEditor.value) {
//             this.requirementsEditor.value = `REQ-001: The system shall allow users to register an account with email and password.
// REQ-002: Users shall be able to log in using their email and password.
// REQ-003: The system shall allow users to reset their password via email.
// REQ-004: Authenticated users shall be able to create new projects.
// REQ-005: Each project shall have a title, description, and creation date.
// REQ-006: Users shall be able to add tasks to their projects.
// REQ-007: Tasks shall have a title, description, due date, and status.`;
//         }
        
    }
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Wait a moment to ensure other components are initialized
    setTimeout(() => {
        window.app = new RequirementsAnalyzer();
    }, 100);
});