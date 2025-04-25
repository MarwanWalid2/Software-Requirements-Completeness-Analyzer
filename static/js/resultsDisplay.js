/**
 * Results Display Module
 * Handles displaying analysis results and domain model
 */
class ResultsDisplay {
    constructor() {
        // DOM Elements - Containers
        this.resultsContainer = document.getElementById('results-container');
        this.requirementsList = document.getElementById('requirements-list');
        this.missingRequirementsContainer = document.getElementById('missing-requirements-container');
        this.modelIssuesContainer = document.getElementById('model-issues-container');
        this.requirementCompletenessContainer = document.getElementById('requirement-completeness-container');
        this.modelElementsList = document.getElementById('model-elements-list');
        this.detailContent = document.getElementById('detail-content');
        this.umlDiagram = document.getElementById('uml-diagram');
        this.requirementsEditor = document.getElementById('requirements-editor');
        
        // DOM Elements - Aggregation Info
        this.domainModelAggregationInfo = document.getElementById('domain-model-aggregation-info');
        this.domainModelAggregationText = document.getElementById('domain-model-aggregation-text');
        this.analysisAggregationInfo = document.getElementById('analysis-aggregation-info');
        this.analysisAggregationText = document.getElementById('analysis-aggregation-text');
        this.modelSourceBadge = document.getElementById('model-source-badge');
        
        // Data state
        this.currentDomainModel = null;
        this.currentAnalysis = null;
        this.acceptedChanges = [];
        this.editedRequirements = [];
        this.processedItems = new Set();

        this.statistics = {
            missingRequirements: { total: 0, accepted: 0 },
            requirementIssues: { total: 0, accepted: 0 },
            modelIssues: { total: 0, accepted: 0 },
            requirementCompleteness: { total: 0, accepted: 0 },
        };
        
        // Modal elements
        this.editModal = new bootstrap.Modal(document.getElementById('editModal'));
        this.editForm = document.getElementById('editForm');
        this.editText = document.getElementById('editText');
        this.editId = document.getElementById('editId');
        this.editType = document.getElementById('editType');
        this.saveEditBtn = document.getElementById('saveEdit');
        
        // Initialize 
        this.init();
    }
    
    /**
     * Initialize the component
     */
    init() {
        // Set up tabs
        this.initializeTabs();
        
        // Set up save edit handler
        this.saveEditBtn.addEventListener('click', () => this.handleSaveEdit());
    }

    /**
     * Initialize statistics tracking
     */
    initializeStatistics() {
        // Reset all counters
        this.statistics = {
            missingRequirements: { total: 0, accepted: 0 },
            requirementIssues: { total: 0, accepted: 0 },
            modelIssues: { total: 0, accepted: 0 },
            requirementCompleteness: { total: 0, accepted: 0 },
        };
    }


    /**
     * Calculate statistics based on current analysis data
     */
    calculateStatistics() {
        if (!this.currentAnalysis) return;
        
        // Count missing requirements
        this.statistics.missingRequirements.total = 
            this.currentAnalysis.missing_requirements ? 
            this.currentAnalysis.missing_requirements.length : 0;
        
        // Count incomplete requirements
        let incompleteReqCount = 0;
        if (this.currentAnalysis.requirement_completeness) {
            incompleteReqCount = this.currentAnalysis.requirement_completeness.filter(
                req => req.completeness_score <= 99
            ).length;
        }
        this.statistics.requirementCompleteness.total = incompleteReqCount;
        
        // Count requirement issues
        let reqIssueCount = 0;
        if (this.currentAnalysis.requirement_issues) {
            this.currentAnalysis.requirement_issues.forEach(req => {
                if (req.issues) {
                    reqIssueCount += req.issues.length;
                }
            });
        }
        this.statistics.requirementIssues.total = reqIssueCount;
        
        // Count model issues
        this.statistics.modelIssues.total = 
            this.currentAnalysis.domain_model_issues ? 
            this.currentAnalysis.domain_model_issues.length : 0;
        
        // Reset accepted counters
        this.statistics.missingRequirements.accepted = 0;
        this.statistics.requirementIssues.accepted = 0;
        this.statistics.modelIssues.accepted = 0;
        this.statistics.requirementCompleteness.accepted = 0;
        
        // Count from accepted changes
        this.acceptedChanges.forEach(change => {
            switch (change.type) {
                case 'missing_requirement':
                    this.statistics.missingRequirements.accepted++;
                    break;
                case 'requirement_issue_fix':
                    this.statistics.requirementIssues.accepted++;
                    break;
                case 'model_issue_fix':
                    this.statistics.modelIssues.accepted++;
                    break;
                case 'requirement_improvement':
                    this.statistics.requirementCompleteness.accepted++;
                    break;
            }
        });
        
        // Update the UI with the new statistics
        this.updateStatisticsUI();
    }

    /**
     * Update the statistics UI elements
     */
    updateStatisticsUI() {
        // Missing requirements
        document.getElementById('total-missing').textContent = this.statistics.missingRequirements.total;
        document.getElementById('accepted-missing').textContent = this.statistics.missingRequirements.accepted;
        const missingProgress = this.statistics.missingRequirements.total > 0 ? 
            (this.statistics.missingRequirements.accepted / this.statistics.missingRequirements.total * 100) : 0;
        document.getElementById('missing-progress').style.width = `${missingProgress}%`;
        
        // Requirement issues
        document.getElementById('total-req-issues').textContent = this.statistics.requirementIssues.total;
        document.getElementById('accepted-req-issues').textContent = this.statistics.requirementIssues.accepted;
        const reqIssuesProgress = this.statistics.requirementIssues.total > 0 ? 
            (this.statistics.requirementIssues.accepted / this.statistics.requirementIssues.total * 100) : 0;
        document.getElementById('req-issues-progress').style.width = `${reqIssuesProgress}%`;
        
        // Model issues
        document.getElementById('total-model-issues').textContent = this.statistics.modelIssues.total;
        document.getElementById('accepted-model-issues').textContent = this.statistics.modelIssues.accepted;
        const modelIssuesProgress = this.statistics.modelIssues.total > 0 ? 
            (this.statistics.modelIssues.accepted / this.statistics.modelIssues.total * 100) : 0;
        document.getElementById('model-issues-progress').style.width = `${modelIssuesProgress}%`;
        
        // Requirement completeness
        document.getElementById('total-completeness').textContent = this.statistics.requirementCompleteness.total;
        document.getElementById('accepted-completeness').textContent = this.statistics.requirementCompleteness.accepted;
        const completenessProgress = this.statistics.requirementCompleteness.total > 0 ? 
            (this.statistics.requirementCompleteness.accepted / this.statistics.requirementCompleteness.total * 100) : 0;
        document.getElementById('completeness-progress').style.width = `${completenessProgress}%`;
        
        // Total statistics
        const totalIssues = this.statistics.missingRequirements.total + 
                            this.statistics.requirementIssues.total + 
                            this.statistics.modelIssues.total + 
                            this.statistics.requirementCompleteness.total;
                            
        const totalAccepted = this.statistics.missingRequirements.accepted + 
                            this.statistics.requirementIssues.accepted + 
                            this.statistics.modelIssues.accepted + 
                            this.statistics.requirementCompleteness.accepted;
                            
        document.getElementById('total-issues').textContent = totalIssues;
        document.getElementById('total-accepted').textContent = totalAccepted;
        
        const overallProgress = totalIssues > 0 ? Math.round(totalAccepted / totalIssues * 100) : 0;
        document.getElementById('overall-progress').textContent = `${overallProgress}%`;
    }
        
    /**
     * Initialize tabs navigation
     */
    initializeTabs() {
        document.querySelectorAll('.sidebar-nav .nav-link').forEach(tabEl => {
            tabEl.addEventListener('click', e => {
                e.preventDefault();
                
                const targetTabId = tabEl.getAttribute('data-bs-target');
                const parentNav = tabEl.closest('.sidebar-nav');
                
                // Deactivate all tabs in this sidebar
                parentNav.querySelectorAll('.nav-link').forEach(tab => {
                    tab.classList.remove('active');
                    tab.setAttribute('aria-selected', 'false');
                });
                
                // Activate clicked tab
                tabEl.classList.add('active');
                tabEl.setAttribute('aria-selected', 'true');
                
                // Hide all panels in this sidebar
                const sidebarContent = parentNav.nextElementSibling;
                sidebarContent.querySelectorAll('.analysis-panel').forEach(panel => {
                    panel.classList.remove('active');
                });
                
                // Show target panel
                document.querySelector(targetTabId).classList.add('active');
            });
        });
    }
    

/**
 * Display results from analysis
 */
displayResults(data) {
    // Store current data
    this.currentDomainModel = data.domain_model;
    this.currentAnalysis = data.analysis;
    
    // Reset lists of changes
    this.acceptedChanges = [];
    this.editedRequirements = [];
    this.processedItems.clear();
    
    // Initialize statistics
    this.initializeStatistics();
    
    // Show results container
    this.resultsContainer.style.display = 'flex';
    
    // Reset lists of changes
    this.acceptedChanges = [];
    this.editedRequirements = [];
    this.processedItems.clear();
    
    // Show results container
    this.resultsContainer.style.display = 'flex';
    
    // Display UML diagram
    if (data.uml_image) {
        this.umlDiagram.src = 'data:image/png;base64,' + data.uml_image;
    }
    
    // Display domain model elements
    this.displayDomainModelElements(data.domain_model);
    
    // Display requirement issues
    this.displayRequirementIssues(data.analysis);
    
    // Display missing requirements
    this.displayMissingRequirements(data.analysis);
    
    // Display model issues
    this.displayModelIssues(data.analysis);
    
    // Display requirement completeness
    this.displayRequirementCompleteness(data.analysis);
    
    // Display aggregation info
    this.displayAggregationInfo(data);
    
    // Calculate and display statistics
    this.calculateStatistics();
    
    // Update requirements in the text editor if they're provided
    if (data.requirements && this.requirementsEditor) {
        this.updateRequirementsEditor(data.requirements);
    }
}

/**
 * Update the requirements editor with new requirements
 */
updateRequirementsEditor(requirements) {
    // Reference to the requirements editor
    const requirementsEditor = document.getElementById('requirements-editor');
    if (requirementsEditor) {
        // Clear the current content
        requirementsEditor.value = '';
        
        // Add the new content with a small delay to ensure smooth UI update
        setTimeout(() => {
            requirementsEditor.value = requirements;
            
            // Trigger a change event to ensure any listeners are notified
            const event = new Event('change', { bubbles: true });
            requirementsEditor.dispatchEvent(event);
            
            // Scroll to the top of the editor
            requirementsEditor.scrollTop = 0;
        }, 50);
    }
}
    
    /**
     * Display aggregation information
     */
    displayAggregationInfo(data) {
        // Display domain model aggregation info
        this.updateSourceBadge(data);
        
        if (data.aggregation_info && data.aggregation_info.domain_model) {
            const dmInfo = data.aggregation_info.domain_model;
            if (dmInfo.strategy && dmInfo.contributing_models && dmInfo.contributing_models.length > 1) {
                let infoText = `Generated using ${this.formatAggregationStrategy(dmInfo.strategy)} from ${dmInfo.contributing_models.length} models: ${dmInfo.contributing_models.join(', ')}`;
                this.domainModelAggregationText.textContent = infoText;
                this.domainModelAggregationInfo.style.display = 'block';
            } else {
                this.domainModelAggregationInfo.style.display = 'none';
            }
        } else {
            this.domainModelAggregationInfo.style.display = 'none';
        }
        
        // Display analysis aggregation info
        if (data.aggregation_info && data.aggregation_info.analysis) {
            const analysisInfo = data.aggregation_info.analysis;
            if (analysisInfo.strategy && analysisInfo.contributing_models && analysisInfo.contributing_models.length > 1) {
                let infoText = `Analysis combined using ${this.formatAggregationStrategy(analysisInfo.strategy)} from ${analysisInfo.contributing_models.length} models: ${analysisInfo.contributing_models.join(', ')}`;
                this.analysisAggregationText.textContent = infoText;
                this.analysisAggregationInfo.style.display = 'block';
            } else {
                this.analysisAggregationInfo.style.display = 'none';
            }
        } else {
            this.analysisAggregationInfo.style.display = 'none';
        }
    }
    
    /**
     * Update the source badge based on aggregation info
     */
    updateSourceBadge(data) {
        if (!this.modelSourceBadge) return;
        
        let html = '';
        
        if (data.aggregation_info && data.aggregation_info.domain_model) {
            const info = data.aggregation_info.domain_model;
            if (info.strategy) {
                // Multiple models
                html = `
                    <i class="fas fa-layer-group me-1"></i>
                    <span>Multi-LLM (${this.formatAggregationStrategy(info.strategy)})</span>
                `;
            } else if (info.contributing_models && info.contributing_models.length === 1) {
                // Single model
                html = `
                    <i class="fas fa-robot me-1"></i>
                    <span>${info.contributing_models[0]}</span>
                `;
            }
        }
        
        if (!html) {
            // Default if no info
            html = `
                <i class="fas fa-robot me-1"></i>
                <span>Domain Model</span>
            `;
        }
        
        this.modelSourceBadge.innerHTML = html;
    }
    
    /**
     * Format aggregation strategy for display
     */
    formatAggregationStrategy(strategy) {
        switch (strategy) {
            case 'majority_vote':
                return 'majority voting';
            case 'weighted_vote':
                return 'weighted voting';
            case 'llm_based_openai':
                return 'OpenAI meta-analysis';
            case 'llm_based_deepseek':
                return 'DeepSeek meta-analysis';
            case 'llm_based_claude':
                return 'Claude meta-analysis';
            default:
                return strategy;
        }
    }
    
    /**
     * Display domain model elements
     */
    displayDomainModelElements(domainModel) {
        this.modelElementsList.innerHTML = '';
        
        if (!domainModel || !domainModel.classes) {
            this.modelElementsList.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    No domain model elements found
                </div>
            `;
            return;
        }
        
        // Add class elements
        const classesHeader = document.createElement('h6');
        classesHeader.className = 'mt-3 mb-2';
        classesHeader.innerHTML = '<i class="fas fa-cube me-2"></i>Classes';
        this.modelElementsList.appendChild(classesHeader);
        
        domainModel.classes.forEach(cls => {
            const element = document.createElement('div');
            element.className = 'requirement-item';
            element.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <span class="fw-bold">${cls.name}</span>
                </div>
                <p class="mb-0 small text-muted">${cls.description || 'No description'}</p>
            `;
            
            element.addEventListener('click', () => {
                // Show class details in the detail panel
                this.showClassDetails(cls);
                
                // Toggle active state
                document.querySelectorAll('#model-elements-list .requirement-item').forEach(el => {
                    el.classList.remove('active');
                });
                element.classList.add('active');
            });
            
            this.modelElementsList.appendChild(element);
        });
        
        // Add relationship elements
        if (domainModel.relationships && domainModel.relationships.length > 0) {
            const relationshipsHeader = document.createElement('h6');
            relationshipsHeader.className = 'mt-4 mb-2';
            relationshipsHeader.innerHTML = '<i class="fas fa-link me-2"></i>Relationships';
            this.modelElementsList.appendChild(relationshipsHeader);
            
            domainModel.relationships.forEach(rel => {
                const element = document.createElement('div');
                element.className = 'requirement-item';
                element.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="fw-bold">${rel.source} → ${rel.target}</span>
                    </div>
                    <p class="mb-0 small text-muted">Type: ${rel.type}</p>
                `;
                
                element.addEventListener('click', () => {
                    // Show relationship details in the detail panel
                    this.showRelationshipDetails(rel);
                    
                    // Toggle active state
                    document.querySelectorAll('#model-elements-list .requirement-item').forEach(el => {
                        el.classList.remove('active');
                    });
                    element.classList.add('active');
                });
                
                this.modelElementsList.appendChild(element);
            });
        }
    }
    
    /**
     * Display requirement issues
     */
    displayRequirementIssues(analysis) {
        this.requirementsList.innerHTML = '';
        
        if (!analysis || !analysis.requirement_issues || analysis.requirement_issues.length === 0) {
            this.requirementsList.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    No requirement issues found
                </div>
            `;
            return;
        }
        
        analysis.requirement_issues.forEach((req, index) => {
            const reqId = req.requirement_id || `R${index + 1}`;
            const hasIssues = req.issues && req.issues.length > 0;
            
            const element = document.createElement('div');
            element.className = 'requirement-item' + (hasIssues ? ' has-issues' : '');
            
            element.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <span class="fw-bold">${reqId}</span>
                        ${hasIssues ? `<span class="badge bg-danger ms-2">${req.issues.length}</span>` : ''}
                    </div>
                    <i class="fas ${hasIssues ? 'fa-exclamation-circle text-danger' : 'fa-check-circle text-success'}"></i>
                </div>
                <p class="mb-0 text-truncate">${req.requirement_text}</p>
            `;
            
            element.addEventListener('click', () => {
                // Show requirement issues in the detail panel
                this.showRequirementIssues(req);
                
                // Toggle active state
                document.querySelectorAll('#requirements-list .requirement-item').forEach(el => {
                    el.classList.remove('active');
                });
                element.classList.add('active');
            });
            
            this.requirementsList.appendChild(element);
        });
        
        // Select the first requirement by default
        if (analysis.requirement_issues.length > 0) {
            const firstRequirement = this.requirementsList.querySelector('.requirement-item');
            if (firstRequirement) {
                firstRequirement.classList.add('active');
                this.showRequirementIssues(analysis.requirement_issues[0]);
            }
        }
    }
    
    /**
     * Show requirement issues in the detail panel
     */
    showRequirementIssues(requirement) {
        this.detailContent.innerHTML = '';
        
        // Display the full requirement text
        const reqText = document.createElement('div');
        reqText.className = 'alert alert-secondary';
        reqText.innerHTML = `<strong>Requirement ${requirement.requirement_id || ''}:</strong> ${requirement.requirement_text}`;
        this.detailContent.appendChild(reqText);
        
        if (!requirement.issues || requirement.issues.length === 0) {
            const noIssues = document.createElement('div');
            noIssues.className = 'alert alert-success';
            noIssues.innerHTML = `<i class="fas fa-check-circle me-2"></i>No issues found for this requirement.`;
            this.detailContent.appendChild(noIssues);
            return;
        }
        
        // Display each issue
        requirement.issues.forEach(issue => {
            let severityClass = '';
            let severityBadge = '';
            
            switch (issue.severity) {
                case 'MUST FIX':
                    severityClass = 'must-fix';
                    severityBadge = '<span class="badge bg-danger">MUST FIX</span>';
                    break;
                case 'SHOULD FIX':
                    severityClass = 'should-fix';
                    severityBadge = '<span class="badge bg-warning">SHOULD FIX</span>';
                    break;
                case 'SUGGESTION':
                    severityClass = 'suggestion';
                    severityBadge = '<span class="badge bg-success">SUGGESTION</span>';
                    break;
            }
            
            const issueCard = document.createElement('div');
            issueCard.className = `card issue-card ${severityClass} mb-3`;
            
            issueCard.innerHTML = `
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="card-title mb-0">${issue.issue_type}</h6>
                        ${severityBadge}
                    </div>
                    <p class="card-text">${issue.description}</p>
                    <div class="alert alert-light">
                        <strong>Suggested Fix:</strong> ${issue.suggested_fix}
                    </div>
                    ${issue.affected_model_elements && issue.affected_model_elements.length > 0 ? `
                        <div class="mt-2 mb-3">
                            <strong>Affected Model Elements:</strong>
                            <ul class="mb-0">
                                ${issue.affected_model_elements.map(element => `<li>${element}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    <div class="action-buttons">
                        <button class="btn btn-sm btn-accept" data-req-id="${requirement.requirement_id || ''}" data-issue-type="${issue.issue_type}">
                            <i class="fas fa-check me-1"></i> Accept Fix
                        </button>
                        <button class="btn btn-sm btn-edit" data-req-id="${requirement.requirement_id || ''}" data-issue-type="${issue.issue_type}">
                            <i class="fas fa-edit me-1"></i> Edit & Accept
                        </button>
                        <button class="btn btn-sm btn-decline" data-req-id="${requirement.requirement_id || ''}" data-issue-type="${issue.issue_type}">
                            <i class="fas fa-times me-1"></i> Decline
                        </button>
                    </div>
                </div>
            `;
            
            // Add event listeners for action buttons
            issueCard.querySelector('.btn-accept').addEventListener('click', () => {
                this.acceptIssueFixAction(requirement, issue, false);
            });
            
            issueCard.querySelector('.btn-edit').addEventListener('click', () => {
                this.showEditModal(requirement, issue, 'requirement_issue');
            });
            
            issueCard.querySelector('.btn-decline').addEventListener('click', () => {
                this.declineIssueFixAction(requirement, issue);
            });
            
            this.detailContent.appendChild(issueCard);
        });
    }
    
    /**
     * Display missing requirements
     */
    displayMissingRequirements(analysis) {
        this.missingRequirementsContainer.innerHTML = '';
        
        if (!analysis || !analysis.missing_requirements || analysis.missing_requirements.length === 0) {
            this.missingRequirementsContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    No missing requirements detected
                </div>
            `;
            return;
        }
        
        // Group by severity or category
        const grouped = {};
        
        analysis.missing_requirements.forEach(req => {
            const category = req.category || 'Other';
            if (!grouped[category]) {
                grouped[category] = [];
            }
            grouped[category].push(req);
        });
        
        // Display each group
        Object.keys(grouped).forEach(category => {
            const categoryHeader = document.createElement('h6');
            categoryHeader.className = 'mt-3 mb-2';
            categoryHeader.textContent = category;
            this.missingRequirementsContainer.appendChild(categoryHeader);
            
            // Display each missing requirement
            grouped[category].forEach(req => {
                let severityClass = '';
                
                switch (req.severity) {
                    case 'CRITICAL':
                        severityClass = 'critical';
                        break;
                    case 'HIGH':
                        severityClass = 'high';
                        break;
                    case 'MEDIUM':
                        severityClass = 'medium';
                        break;
                    case 'LOW':
                        severityClass = 'low';
                        break;
                }
                
                const reqCard = document.createElement('div');
                reqCard.className = `card issue-card ${severityClass} mb-3`;
                
                reqCard.innerHTML = `
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="card-title mb-0">${req.description}</h6>
                            <span class="badge bg-${severityClass === 'critical' ? 'danger' : 
                                                severityClass === 'high' ? 'warning' : 
                                                severityClass === 'medium' ? 'success' : 'info'}">${req.severity}</span>
                        </div>
                        <div class="alert alert-light">
                            <strong>Suggested Requirement:</strong> ${req.suggested_requirement}
                        </div>
                        <p class="small text-muted mb-2"><strong>Rationale:</strong> ${req.rationale || 'No rationale provided'}</p>
                        
                        ${req.affected_model_elements && req.affected_model_elements.length > 0 ? `
                            <div class="mt-2 mb-3">
                                <strong>Affected Model Elements:</strong>
                                <ul class="mb-0">
                                    ${req.affected_model_elements.map(element => `<li>${element}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        
                        <div class="action-buttons">
                            <button class="btn btn-sm btn-accept" data-missing-id="${req.id}">
                                <i class="fas fa-check me-1"></i> Accept
                            </button>
                            <button class="btn btn-sm btn-edit" data-missing-id="${req.id}">
                                <i class="fas fa-edit me-1"></i> Edit & Accept
                            </button>
                            <button class="btn btn-sm btn-decline" data-missing-id="${req.id}">
                                <i class="fas fa-times me-1"></i> Decline
                            </button>
                        </div>
                    </div>
                `;
                
                // Add event listeners for action buttons
                reqCard.querySelector('.btn-accept').addEventListener('click', () => {
                    this.acceptMissingRequirementAction(req, false);
                });
                
                reqCard.querySelector('.btn-edit').addEventListener('click', () => {
                    this.showEditModal(req, null, 'missing_requirement');
                });
                
                reqCard.querySelector('.btn-decline').addEventListener('click', () => {
                    this.declineMissingRequirementAction(req);
                });
                
                this.missingRequirementsContainer.appendChild(reqCard);
            });
        });
    }
    
    /**
     * Display requirement completeness
     */
    displayRequirementCompleteness(analysis) {
        this.requirementCompletenessContainer.innerHTML = '';
        
        if (!analysis || !analysis.requirement_completeness || analysis.requirement_completeness.length === 0) {
            this.requirementCompletenessContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    No requirement completeness analysis available
                </div>
            `;
            return;
        }
        
        // Sort by completeness score (ascending)
        const sortedRequirements = [...analysis.requirement_completeness].sort((a, b) => a.completeness_score - b.completeness_score);
        
        sortedRequirements.forEach(req => {
            let completenessClass = '';
            
            if (req.completeness_score < 50) {
                completenessClass = 'critical';
            } else if (req.completeness_score < 75) {
                completenessClass = 'high';
            } else if (req.completeness_score < 90) {
                completenessClass = 'medium';
            } else {
                completenessClass = 'low';
            }
            
            const reqCard = document.createElement('div');
            reqCard.className = `card issue-card ${completenessClass} mb-3`;
            
            reqCard.innerHTML = `
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h6 class="card-title mb-0">${req.requirement_id || ''}</h6>
                        <div>
                            <span class="badge bg-${completenessClass === 'critical' ? 'danger' : 
                                                completenessClass === 'high' ? 'warning' : 
                                                completenessClass === 'medium' ? 'success' : 'info'}">${req.completeness_score}%</span>
                        </div>
                    </div>
                    <p class="card-text">${req.requirement_text}</p>
                    
                    ${req.missing_elements && req.missing_elements.length > 0 ? `
                        <div class="mt-2 mb-2">
                            <strong>Missing Elements:</strong>
                            <ul class="mb-0">
                                ${req.missing_elements.map(element => `<li>${element}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    <div class="alert alert-light">
                        <strong>Suggested Improvement:</strong> ${req.suggested_improvement}
                    </div>
                    <p class="small text-muted mb-3"><strong>Rationale:</strong> ${req.rationale || 'No rationale provided'}</p>
                    
                    <div class="action-buttons">
                        <button class="btn btn-sm btn-accept" data-completeness-id="${req.requirement_id}">
                            <i class="fas fa-check me-1"></i> Accept
                        </button>
                        <button class="btn btn-sm btn-edit" data-completeness-id="${req.requirement_id}">
                            <i class="fas fa-edit me-1"></i> Edit & Accept
                        </button>
                        <button class="btn btn-sm btn-decline" data-completeness-id="${req.requirement_id}">
                            <i class="fas fa-times me-1"></i> Decline
                        </button>
                    </div>
                </div>
            `;
            
            // Add event listeners for action buttons
            reqCard.querySelector('.btn-accept').addEventListener('click', () => {
                this.acceptCompletenessImprovementAction(req, false);
            });
            
            reqCard.querySelector('.btn-edit').addEventListener('click', () => {
                this.showEditModal(req, null, 'requirement_completeness');
            });
            
            reqCard.querySelector('.btn-decline').addEventListener('click', () => {
                this.declineCompletenessImprovementAction(req);
            });
            
            this.requirementCompletenessContainer.appendChild(reqCard);
        });
    }
    
    /**
     * Display model issues
     */
    displayModelIssues(analysis) {
        this.modelIssuesContainer.innerHTML = '';
        
        if (!analysis || !analysis.domain_model_issues || analysis.domain_model_issues.length === 0) {
            this.modelIssuesContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    No domain model issues detected
                </div>
            `;
            return;
        }
        
        // Group by element type
        const grouped = {};
        
        analysis.domain_model_issues.forEach(issue => {
            const type = issue.element_type || 'Other';
            if (!grouped[type]) {
                grouped[type] = [];
            }
            grouped[type].push(issue);
        });
        
        // Display each group
        Object.keys(grouped).forEach(type => {
            const typeHeader = document.createElement('h6');
            typeHeader.className = 'mt-3 mb-2';
            typeHeader.innerHTML = `<i class="fas fa-${type === 'Class' ? 'cube' : 
                                                type === 'Relationship' ? 'link' : 
                                                type === 'Attribute' ? 'tag' : 
                                                type === 'Method' ? 'cog' : 'question'} me-2"></i>${type} Issues`;
            this.modelIssuesContainer.appendChild(typeHeader);
            
            // Display each issue
            grouped[type].forEach(issue => {
                let severityClass = '';
                let severityBadge = '';
                
                switch (issue.severity) {
                    case 'MUST FIX':
                        severityClass = 'must-fix';
                        severityBadge = '<span class="badge bg-danger">MUST FIX</span>';
                        break;
                    case 'SHOULD FIX':
                        severityClass = 'should-fix';
                        severityBadge = '<span class="badge bg-warning">SHOULD FIX</span>';
                        break;
                    case 'SUGGESTION':
                        severityClass = 'suggestion';
                        severityBadge = '<span class="badge bg-success">SUGGESTION</span>';
                        break;
                }
                
                const issueCard = document.createElement('div');
                issueCard.className = `card issue-card ${severityClass} mb-3`;
                
                issueCard.innerHTML = `
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="card-title mb-0">${issue.element_name}</h6>
                            ${severityBadge}
                        </div>
                        <p class="card-text">
                            <span class="badge bg-secondary me-2">${issue.issue_type}</span>
                            ${issue.description}
                        </p>
                        <div class="alert alert-light">
                            <strong>Suggested Fix:</strong> ${issue.suggested_fix}
                        </div>
                        ${issue.affected_requirements && issue.affected_requirements.length > 0 ? `
                            <div class="mt-2 mb-3">
                                <strong>Affected Requirements:</strong>
                                <ul class="mb-0">
                                    ${issue.affected_requirements.map(req => `<li>${req}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        
                        <div class="action-buttons">
                            <button class="btn btn-sm btn-accept" data-model-issue-element="${issue.element_name}" data-model-issue-type="${issue.issue_type}">
                                <i class="fas fa-check me-1"></i> Accept Fix
                            </button>
                            <button class="btn btn-sm btn-edit" data-model-issue-element="${issue.element_name}" data-model-issue-type="${issue.issue_type}">
                                <i class="fas fa-edit me-1"></i> Edit & Accept
                            </button>
                            <button class="btn btn-sm btn-decline" data-model-issue-element="${issue.element_name}" data-model-issue-type="${issue.issue_type}">
                                <i class="fas fa-times me-1"></i> Decline
                            </button>
                        </div>
                    </div>
                `;
                
                // Add event listeners for action buttons
                issueCard.querySelector('.btn-accept').addEventListener('click', () => {
                    this.acceptModelIssueFixAction(issue, false);
                });
                
                issueCard.querySelector('.btn-edit').addEventListener('click', () => {
                    this.showEditModal(issue, null, 'model_issue');
                });
                
                issueCard.querySelector('.btn-decline').addEventListener('click', () => {
                    this.declineModelIssueFixAction(issue);
                });
                
                this.modelIssuesContainer.appendChild(issueCard);
            });
        });
    }
    
    /**
     * Show class details
     */
    showClassDetails(cls) {
        this.detailContent.innerHTML = '';
        
        const header = document.createElement('h5');
        header.className = 'mb-3';
        header.innerHTML = `<i class="fas fa-cube me-2"></i>${cls.name}`;
        this.detailContent.appendChild(header);
        
        const description = document.createElement('div');
        description.className = 'alert alert-secondary mb-3';
        description.innerHTML = `<strong>Description:</strong> ${cls.description || 'No description provided'}`;
        this.detailContent.appendChild(description);
        
        // Attributes section
        if (cls.attributes && cls.attributes.length > 0) {
            const attrHeader = document.createElement('h6');
            attrHeader.className = 'mt-4 mb-2';
            attrHeader.innerHTML = '<i class="fas fa-tag me-2"></i>Attributes';
            this.detailContent.appendChild(attrHeader);
            
            const attrTable = document.createElement('div');
            attrTable.className = 'table-responsive';
            attrTable.innerHTML = `
                <table class="table table-sm table-bordered">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${cls.attributes.map(attr => `
                            <tr>
                                <td>${attr.name}</td>
                                <td><code>${attr.type}</code></td>
                                <td>${attr.description || ''}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            this.detailContent.appendChild(attrTable);
        }
        
        // Methods section
        if (cls.methods && cls.methods.length > 0) {
            const methodHeader = document.createElement('h6');
            methodHeader.className = 'mt-4 mb-2';
            methodHeader.innerHTML = '<i class="fas fa-cog me-2"></i>Methods';
            this.detailContent.appendChild(methodHeader);
            
            cls.methods.forEach(method => {
                const methodCard = document.createElement('div');
                methodCard.className = 'card mb-3';
                
                const params = method.parameters ? method.parameters.map(param => `${param.name}: ${param.type}`).join(', ') : '';
                
                methodCard.innerHTML = `
                    <div class="card-body">
                        <h6 class="card-title">${method.name}(${params}): ${method.returnType || 'void'}</h6>
                        <p class="card-text small">${method.description || 'No description'}</p>
                    </div>
                `;
                
                this.detailContent.appendChild(methodCard);
            });
        }
    }
    
    /**
     * Show relationship details
     */
    showRelationshipDetails(rel) {
        this.detailContent.innerHTML = '';
        
        const header = document.createElement('h5');
        header.className = 'mb-3';
        header.innerHTML = `<i class="fas fa-link me-2"></i>Relationship`;
        this.detailContent.appendChild(header);
        
        const relationshipCard = document.createElement('div');
        relationshipCard.className = 'card mb-4';
        relationshipCard.innerHTML = `
            <div class="card-body">
                <div class="d-flex justify-content-between mb-3">
                    <h5 class="card-title">${rel.source}</h5>
                    <h5 class="card-title">${rel.target}</h5>
                </div>
                <div class="text-center mb-3">
                    <span class="badge bg-primary p-2">${rel.type}</span>
                    <i class="fas fa-arrow-right mx-2"></i>
                </div>
                <div class="d-flex justify-content-between mb-3">
                    <div>Source Multiplicity: <strong>${rel.sourceMultiplicity || 'N/A'}</strong></div>
                    <div>Target Multiplicity: <strong>${rel.targetMultiplicity || 'N/A'}</strong></div>
                </div>
                <p class="card-text">${rel.description || 'No description provided'}</p>
            </div>
        `;
        
        this.detailContent.appendChild(relationshipCard);
    }
    
    /**
     * Show edit modal
     */
    showEditModal(item, issue, type) {
        this.editId.value = '';
        this.editType.value = type;
        
        // Set the text based on the item type
        switch (type) {
            case 'requirement_issue':
                this.editId.value = item.requirement_id || '';
                this.editText.value = issue.suggested_fix || '';
                document.getElementById('editModalLabel').textContent = `Edit Fix for ${item.requirement_id}`;
                break;
            case 'missing_requirement':
                this.editId.value = item.id || '';
                this.editText.value = item.suggested_requirement || '';
                document.getElementById('editModalLabel').textContent = 'Edit Missing Requirement';
                break;
            case 'requirement_completeness':
                this.editId.value = item.requirement_id || '';
                this.editText.value = item.suggested_improvement || '';
                document.getElementById('editModalLabel').textContent = `Edit Requirement Improvement`;
                break;
            case 'model_issue':
                this.editId.value = item.element_name || '';
                this.editText.value = item.suggested_fix || '';
                document.getElementById('editModalLabel').textContent = `Edit Model Fix for ${item.element_name}`;
                break;
        }
        
        // Show the modal
        this.editModal.show();
    }
    
    /**
     * Handle save edit
     */
    handleSaveEdit() {
        const id = this.editId.value;
        const type = this.editType.value;
        const text = this.editText.value;
        
        if (!text.trim()) {
            alert('Please enter text.');
            return;
        }
        
        // Process the edit based on type
        switch (type) {
            case 'requirement_issue':
                // Find the requirement and issue
                const req = this.currentAnalysis.requirement_issues.find(r => r.requirement_id === id);
                if (req) {
                    const issue = req.issues[0]; // Simplified - we should find the exact issue
                    issue.suggested_fix = text;
                    this.acceptIssueFixAction(req, issue, true);
                }
                break;
            case 'missing_requirement':
                const missingReq = this.currentAnalysis.missing_requirements.find(r => r.id === id);
                if (missingReq) {
                    missingReq.suggested_requirement = text;
                    this.acceptMissingRequirementAction(missingReq, true);
                }
                break;
            case 'requirement_completeness':
                const completenessReq = this.currentAnalysis.requirement_completeness.find(r => r.requirement_id === id);
                if (completenessReq) {
                    completenessReq.suggested_improvement = text;
                    this.acceptCompletenessImprovementAction(completenessReq, true);
                }
                break;
            case 'model_issue':
                const modelIssue = this.currentAnalysis.domain_model_issues.find(i => i.element_name === id);
                if (modelIssue) {
                    modelIssue.suggested_fix = text;
                    this.acceptModelIssueFixAction(modelIssue, true);
                }
                break;
        }
        
        // Close the modal
        this.editModal.hide();
    }
    
    /**
     * Accept issue fix action
     */
    acceptIssueFixAction(requirement, issue, edited = false) {
        console.log(`Accepting fix for ${requirement.requirement_id}: ${issue.issue_type}`);
        
        // Generate a unique ID for this issue
        const itemId = `req-${requirement.requirement_id}-${issue.issue_type}`;
        
        // Skip if already processed
        if (this.processedItems.has(itemId)) {
            return;
        }
        
        // Mark as processed
        this.processedItems.add(itemId);
        
        // Mark as accepted in UI
        const issueCards = document.querySelectorAll(`.issue-card`);
        issueCards.forEach(card => {
            const reqIdBtn = card.querySelector(`[data-req-id="${requirement.requirement_id}"][data-issue-type="${issue.issue_type}"]`);
            if (reqIdBtn) {
                // Hide or fade out the card
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                
                // Add acceptance message
                card.innerHTML += `<div class="mt-2 text-success"><i class="fas fa-check-circle me-1"></i> Fix accepted</div>`;
                
                // Disable all buttons
                card.querySelectorAll('button').forEach(btn => {
                    btn.disabled = true;
                });
            }
        });
        
        // Add to accepted changes list
        this.acceptedChanges.push({
            type: 'requirement_issue_fix',
            requirement_id: requirement.requirement_id,
            issue_type: issue.issue_type,
            description: issue.description,
            suggested_fix: issue.suggested_fix,
            affected_elements: issue.affected_model_elements || [],
            edited: edited
        });
        
        // Update changes count and show update button
        this.updateChangesCount();
        // Update statistics
        this.calculateStatistics();
    }
    
    /**
     * Decline issue fix action
     */
    declineIssueFixAction(requirement, issue) {
        console.log(`Declining fix for ${requirement.requirement_id}: ${issue.issue_type}`);
        
        // Generate a unique ID for this issue
        const itemId = `req-${requirement.requirement_id}-${issue.issue_type}`;
        
        // Skip if already processed
        if (this.processedItems.has(itemId)) {
            return;
        }
        
        // Mark as processed
        this.processedItems.add(itemId);
        
        // Mark as declined in UI
        const issueCards = document.querySelectorAll(`.issue-card`);
        issueCards.forEach(card => {
            const reqIdBtn = card.querySelector(`[data-req-id="${requirement.requirement_id}"][data-issue-type="${issue.issue_type}"]`);
            if (reqIdBtn) {
                // Hide or fade out the card
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                
                // Add declined message
                card.innerHTML += `<div class="mt-2 text-danger"><i class="fas fa-times-circle me-1"></i> Fix declined</div>`;
                
                // Disable all buttons
                card.querySelectorAll('button').forEach(btn => {
                    btn.disabled = true;
                });
            }
        });
    }
    
    /**
     * Accept missing requirement action
     */
    acceptMissingRequirementAction(missingReq, edited = false) {
        console.log(`Accepting missing requirement: ${missingReq.id}`);
        
        // Generate a unique ID for this requirement
        const itemId = `missing-${missingReq.id}`;
        
        // Skip if already processed
        if (this.processedItems.has(itemId)) {
            return;
        }
        
        // Mark as processed
        this.processedItems.add(itemId);
        
        // Mark as accepted in UI
        const reqCards = document.querySelectorAll(`.issue-card`);
        reqCards.forEach(card => {
            const missingIdBtn = card.querySelector(`[data-missing-id="${missingReq.id}"]`);
            if (missingIdBtn) {
                // Hide or fade out the card
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                
                // Add acceptance message
                card.innerHTML += `<div class="mt-2 text-success"><i class="fas fa-check-circle me-1"></i> Requirement accepted</div>`;
                
                // Disable all buttons
                card.querySelectorAll('button').forEach(btn => {
                    btn.disabled = true;
                });
            }
        });
        
        // Add to accepted changes list
        this.acceptedChanges.push({
            type: 'missing_requirement',
            id: missingReq.id,
            description: missingReq.description,
            suggested_text: missingReq.suggested_requirement,
            affected_elements: missingReq.affected_model_elements || [],
            edited: edited
        });
        
        // Update changes count and show update button
        this.updateChangesCount();
        // Update statistics
        this.calculateStatistics();

    }
    
    /**
     * Decline missing requirement action
     */
    declineMissingRequirementAction(missingReq) {
        console.log(`Declining missing requirement: ${missingReq.id}`);
        
        // Generate a unique ID for this requirement
        const itemId = `missing-${missingReq.id}`;
        
        // Skip if already processed
        if (this.processedItems.has(itemId)) {
            return;
        }
        
        // Mark as processed
        this.processedItems.add(itemId);
        
        // Mark as declined in UI
        const reqCards = document.querySelectorAll(`.issue-card`);
        reqCards.forEach(card => {
            const missingIdBtn = card.querySelector(`[data-missing-id="${missingReq.id}"]`);
            if (missingIdBtn) {
                // Hide or fade out the card
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                
                // Add declined message
                card.innerHTML += `<div class="mt-2 text-danger"><i class="fas fa-times-circle me-1"></i> Requirement declined</div>`;
                
                // Disable all buttons
                card.querySelectorAll('button').forEach(btn => {
                    btn.disabled = true;
                });
            }
        });
    }
    
    /**
     * Accept completeness improvement action
     */
    acceptCompletenessImprovementAction(req, edited = false) {
        console.log(`Accepting completeness improvement for: ${req.requirement_id}`);
        
        // Generate a unique ID for this requirement
        const itemId = `completeness-${req.requirement_id}`;
        
        // Skip if already processed
        if (this.processedItems.has(itemId)) {
            return;
        }
        
        // Mark as processed
        this.processedItems.add(itemId);
        
        // Mark as accepted in UI
        const reqCards = document.querySelectorAll(`.issue-card`);
        reqCards.forEach(card => {
            const completenessIdBtn = card.querySelector(`[data-completeness-id="${req.requirement_id}"]`);
            if (completenessIdBtn) {
                // Hide or fade out the card
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                
                // Add acceptance message
                card.innerHTML += `<div class="mt-2 text-success"><i class="fas fa-check-circle me-1"></i> Improvement accepted</div>`;
                
                // Disable all buttons
                card.querySelectorAll('button').forEach(btn => {
                    btn.disabled = true;
                });
            }
        });
        
        // Add to accepted changes list
        this.acceptedChanges.push({
            type: 'requirement_improvement',
            requirement_id: req.requirement_id,
            description: `Improve completeness of requirement ${req.requirement_id}`,
            suggested_text: req.suggested_improvement,
            missing_elements: req.missing_elements || [],
            edited: edited
        });
        
        // Update changes count and show update button
        this.updateChangesCount();
        // Update statistics
        this.calculateStatistics();
    }
    
    /**
     * Decline completeness improvement action
     */
    declineCompletenessImprovementAction(req) {
        console.log(`Declining completeness improvement for: ${req.requirement_id}`);
        
        // Generate a unique ID for this requirement
        const itemId = `completeness-${req.requirement_id}`;
        
        // Skip if already processed
        if (this.processedItems.has(itemId)) {
            return;
        }
        
        // Mark as processed
        this.processedItems.add(itemId);
        
        // Mark as declined in UI
        const reqCards = document.querySelectorAll(`.issue-card`);
        reqCards.forEach(card => {
            const completenessIdBtn = card.querySelector(`[data-completeness-id="${req.requirement_id}"]`);
            if (completenessIdBtn) {
                // Hide or fade out the card
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                
                // Add declined message
                card.innerHTML += `<div class="mt-2 text-danger"><i class="fas fa-times-circle me-1"></i> Improvement declined</div>`;
                
                // Disable all buttons
                card.querySelectorAll('button').forEach(btn => {
                    btn.disabled = true;
                });
            }
        });
    }
    
    /**
     * Accept model issue fix action
     */
    acceptModelIssueFixAction(issue, edited = false) {
        console.log(`Accepting model issue fix for: ${issue.element_name} - ${issue.issue_type}`);
        
        // Generate a unique ID for this issue
        const itemId = `model-${issue.element_name}-${issue.issue_type}`;
        
        // Skip if already processed
        if (this.processedItems.has(itemId)) {
            return;
        }
        
        // Mark as processed
        this.processedItems.add(itemId);
        
        // Mark as accepted in UI
        const issueCards = document.querySelectorAll(`.issue-card`);
        issueCards.forEach(card => {
            const modelIssueBtn = card.querySelector(`[data-model-issue-element="${issue.element_name}"][data-model-issue-type="${issue.issue_type}"]`);
            if (modelIssueBtn) {
                // Hide or fade out the card
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                
                // Add acceptance message
                card.innerHTML += `<div class="mt-2 text-success"><i class="fas fa-check-circle me-1"></i> Fix accepted</div>`;
                
                // Disable all buttons
                card.querySelectorAll('button').forEach(btn => {
                    btn.disabled = true;
                });
            }
        });
        
        // Add to accepted changes list
        this.acceptedChanges.push({
            type: 'model_issue_fix',
            element_type: issue.element_type,
            element_name: issue.element_name,
            issue_type: issue.issue_type,
            description: issue.description,
            suggested_fix: issue.suggested_fix,
            affected_requirements: issue.affected_requirements || [],
            edited: edited
        });
        
        // Update changes count and show update button
        this.updateChangesCount();
        // Update statistics
        this.calculateStatistics();
    }
    
    /**
     * Decline model issue fix action
     */
    declineModelIssueFixAction(issue) {
        console.log(`Declining model issue fix for: ${issue.element_name} - ${issue.issue_type}`);
        
        // Generate a unique ID for this issue
        const itemId = `model-${issue.element_name}-${issue.issue_type}`;
        
        // Skip if already processed
        if (this.processedItems.has(itemId)) {
            return;
        }
        
        // Mark as processed
        this.processedItems.add(itemId);
        
        // Mark as declined in UI
        const issueCards = document.querySelectorAll(`.issue-card`);
        issueCards.forEach(card => {
            const modelIssueBtn = card.querySelector(`[data-model-issue-element="${issue.element_name}"][data-model-issue-type="${issue.issue_type}"]`);
            if (modelIssueBtn) {
                // Hide or fade out the card
                card.style.opacity = '0.6';
                card.style.pointerEvents = 'none';
                
                // Add declined message
                card.innerHTML += `<div class="mt-2 text-danger"><i class="fas fa-times-circle me-1"></i> Fix declined</div>`;
                
                // Disable all buttons
                card.querySelectorAll('button').forEach(btn => {
                    btn.disabled = true;
                });
            }
        });
    }
    
    /**
     * Update changes count and show update button
     */
    updateChangesCount() {
        const updateModelBtn = document.getElementById('update-model-btn');
        const changesCountEl = document.getElementById('changes-count');
        
        if (changesCountEl) {
            changesCountEl.textContent = this.acceptedChanges.length;
        }
        
        if (updateModelBtn) {
            updateModelBtn.style.display = this.acceptedChanges.length > 0 ? 'block' : 'none';
        }
    }
    
    /**
     * Get accepted changes
     */
    getAcceptedChanges() {
        return this.acceptedChanges;
    }
    
    /**
     * Get edited requirements
     */
    getEditedRequirements() {
        return this.editedRequirements;
    }
    
    /**
     * Reset display state
     */
    reset() {
        this.acceptedChanges = [];
        this.editedRequirements = [];
        this.processedItems.clear();
        this.updateChangesCount();
        this.initializeStatistics();
        this.calculateStatistics();
    }
}

// Initialize the results display
window.resultsDisplay = new ResultsDisplay();