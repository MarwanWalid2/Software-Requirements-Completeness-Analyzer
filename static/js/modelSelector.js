/**
 * Model Selector Module
 * Handles the selection of LLM models and aggregation methods
 */
class ModelSelector {
    constructor() {
        // DOM Elements
        this.modelSelectionContainer = document.getElementById('model-selection');
        this.metaModelSelect = document.getElementById('meta-model-select');
        this.weightsContainer = document.getElementById('model-weights-container');
        this.weightInputs = document.getElementById('weight-inputs');
        
        // State
        this.availableModels = {};
        this.availableMetaModels = {};
        this.selectedModels = [];
        this.currentMetaModel = 'majority';
        this.modelWeights = {};
        
        // Default weights for different models
        this.defaultWeights = {
            'deepseek': 1.2,
            'openai': 1.0,
            'claude': 1.1
        };
        
        // Bind event listeners
        this.init();
    }
    
    /**
     * Initialize the component
     */
    init() {
        // Fetch available models
        this.fetchAvailableModels();
        
        // Add listener for meta-model selection
        this.metaModelSelect.addEventListener('change', () => {
            this.currentMetaModel = this.metaModelSelect.value;
            this.toggleWeightsContainer();
        });
    }
    
    /**
     * Fetch available models from the server
     */
    async fetchAvailableModels() {
        try {
            const response = await fetch('/api/available-models');
            if (!response.ok) {
                throw new Error('Failed to fetch available models');
            }
            
            const data = await response.json();
            this.availableModels = data.models || {};
            this.availableMetaModels = data.meta_models || {};
            
            this.populateModelSelection();
            this.populateMetaModelSelection();
        } catch (error) {
            console.error('Error fetching available models:', error);
            this.handleFetchError();
        }
    }
    
    /**
     * Handle fetch error
     */
    handleFetchError() {
        this.modelSelectionContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Could not fetch available models. Using default configuration.
            </div>
        `;
        
        // Add default model
        this.addDefaultModelOption();
    }
    
    /**
     * Add default model option
     */
    addDefaultModelOption() {
        this.availableModels = {
            'openai': {
                id: 'openai',
                name: 'GPT-o4-mini',
                enabled: true
            }
        };
        
        this.populateModelSelection();
    }
    
    /**
     * Populate model selection checkboxes
     */
    populateModelSelection() {
        this.modelSelectionContainer.innerHTML = '';
        
        if (!this.availableModels || Object.keys(this.availableModels).length === 0) {
            this.modelSelectionContainer.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    No LLM models available. Check your API keys configuration.
                </div>
            `;
            return;
        }
        
        // Clear selected models
        this.selectedModels = [];
        
        // Create checkbox for each model
        Object.values(this.availableModels).forEach(model => {
            const checkbox = this.createModelCheckbox(model);
            this.modelSelectionContainer.appendChild(checkbox);
            
            // Add to selected models if checked by default
            if (model.id === 'openai') {
                this.selectedModels.push(model.id);
                this.modelWeights[model.id] = this.defaultWeights[model.id] || 1.0;
            }
        });
        
        // Update weights if needed
        if (this.currentMetaModel === 'weighted') {
            this.updateWeightInputs();
        }
    }
    
    /**
     * Create a checkbox for a model
     */
    createModelCheckbox(model) {
        const checkboxDiv = document.createElement('div');
        checkboxDiv.className = 'model-checkbox';
        if (model.id === 'openai') {
            checkboxDiv.classList.add('selected');
        }
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `model-${model.id}`;
        checkbox.value = model.id;
        checkbox.checked = model.id === 'openai';
        
        const label = document.createElement('label');
        label.htmlFor = `model-${model.id}`;
        label.className = 'model-name';
        label.textContent = model.name;
        
        checkboxDiv.appendChild(checkbox);
        checkboxDiv.appendChild(label);
        
        // Add change event listener
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                this.selectedModels.push(model.id);
                checkboxDiv.classList.add('selected');
                
                // Set default weight
                this.modelWeights[model.id] = this.defaultWeights[model.id] || 1.0;
            } else {
                this.selectedModels = this.selectedModels.filter(id => id !== model.id);
                checkboxDiv.classList.remove('selected');
                
                // Remove weight
                delete this.modelWeights[model.id];
            }
            
            // Make sure at least one model is selected
            if (this.selectedModels.length === 0) {
                checkbox.checked = true;
                checkboxDiv.classList.add('selected');
                this.selectedModels.push(model.id);
                this.modelWeights[model.id] = this.defaultWeights[model.id] || 1.0;
                alert('At least one model must be selected');
            }
            
            // Update weights if needed
            if (this.currentMetaModel === 'weighted') {
                this.updateWeightInputs();
            }
        });
        
        return checkboxDiv;
    }
    
    /**
     * Populate meta-model selection dropdown
     */
    populateMetaModelSelection() {
        // Clear existing options except majority and weighted
        const options = this.metaModelSelect.querySelectorAll('option');
        for (let i = options.length - 1; i >= 0; i--) {
            const option = options[i];
            if (option.value !== 'majority' && option.value !== 'weighted') {
                this.metaModelSelect.removeChild(option);
            }
        }
        
        if (!this.availableMetaModels || Object.keys(this.availableMetaModels).length === 0) {
            return;
        }
        
        // Add meta-model options
        Object.values(this.availableMetaModels).forEach(model => {
            // Skip majority and weighted as they're already in the HTML
            if (model.id !== 'majority' && model.id !== 'weighted') {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                
                if (model.description) {
                    option.title = model.description;
                }
                
                this.metaModelSelect.appendChild(option);
            }
        });
        
        // Set default meta-model
        this.metaModelSelect.value = 'majority';
        this.currentMetaModel = 'majority';
    }
    
    /**
     * Toggle weights container visibility based on selected meta-model
     */
    toggleWeightsContainer() {
        if (this.currentMetaModel === 'weighted') {
            this.weightsContainer.style.display = 'block';
            this.updateWeightInputs();
        } else {
            this.weightsContainer.style.display = 'none';
        }
    }
    
    /**
     * Update weight inputs based on selected models
     */
    updateWeightInputs() {
        // Clear existing inputs
        this.weightInputs.innerHTML = '';
        
        // Add weight input for each selected model
        this.selectedModels.forEach(modelId => {
            // Get model info
            const model = this.availableModels[modelId];
            if (!model) return;
            
            // Create input group
            const inputGroup = document.createElement('div');
            inputGroup.className = 'input-group input-group-sm mb-2 weight-input-group';
            inputGroup.style.maxWidth = '250px';
            
            // Label
            const label = document.createElement('span');
            label.className = 'input-group-text';
            label.style.minWidth = '120px';
            label.textContent = model.name;
            
            // Input
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'form-control';
            input.id = `weight-${modelId}`;
            input.min = '0.1';
            input.max = '5.0';
            input.step = '0.1';
            input.value = this.modelWeights[modelId] || this.defaultWeights[modelId] || 1.0;
            
            // Update model weight when input changes
            input.addEventListener('change', () => {
                const value = parseFloat(input.value);
                if (value >= 0.1 && value <= 5.0) {
                    this.modelWeights[modelId] = value;
                } else {
                    input.value = this.defaultWeights[modelId] || 1.0;
                    this.modelWeights[modelId] = parseFloat(input.value);
                }
            });
            
            // Append elements
            inputGroup.appendChild(label);
            inputGroup.appendChild(input);
            this.weightInputs.appendChild(inputGroup);
        });
    }
    
    /**
     * Get selected models
     */
    getSelectedModels() {
        return this.selectedModels;
    }
    
    /**
     * Get current meta-model
     */
    getCurrentMetaModel() {
        return this.currentMetaModel;
    }
    
    /**
     * Get model weights
     */
    getModelWeights() {
        if (this.currentMetaModel === 'weighted') {
            return this.modelWeights;
        }
        return {};
    }
}

// Initialize the model selector and make it globally available
window.modelSelector = new ModelSelector();