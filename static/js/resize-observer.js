/**
 * Resize Observer to handle content resizing and layout adjustments
 * Add this script to improve the responsiveness of the UI
 */
document.addEventListener('DOMContentLoaded', function() {
    // Elements to observe for size changes
    const resultsContainer = document.getElementById('results-container');
    const sidebarSections = document.querySelectorAll('.sidebar-section');
    const domainModelSection = document.querySelector('.domain-model-section');
    const umlContainer = document.querySelector('.uml-container');
    
    // Function to adjust element heights for better layout
    function adjustHeights() {
        if (!resultsContainer || resultsContainer.style.display === 'none') return;
        
        // Get the viewport height
        const viewportHeight = window.innerHeight;
        
        // Calculate reasonable heights
        const headerHeight = 200; // Approximate height of headers, nav, etc.
        const availableHeight = viewportHeight - headerHeight;
        const minHeight = Math.max(600, availableHeight);
        
        // Apply to main elements
        if (resultsContainer) {
            resultsContainer.style.minHeight = minHeight + 'px';
        }
        
        if (domainModelSection) {
            domainModelSection.style.minHeight = (minHeight - 50) + 'px';
        }
        
        if (umlContainer) {
            umlContainer.style.minHeight = (minHeight - 100) + 'px';
        }
        
        // Apply to sidebar sections
        sidebarSections.forEach(section => {
            section.style.minHeight = (minHeight - 50) + 'px';
            
            // Find the sidebar content within this section
            const sidebarContent = section.querySelector('.sidebar-content');
            if (sidebarContent) {
                sidebarContent.style.height = (minHeight - 150) + 'px';
            }
        });
    }
    
    // Setup resize observer
    if (window.ResizeObserver) {
        const resizeObserver = new ResizeObserver(entries => {
            // When the observed elements change size, adjust heights
            adjustHeights();
        });
        
        // Observe the main results container if it exists
        if (resultsContainer) {
            resizeObserver.observe(resultsContainer);
        }
        
        // Also observe the domain model section
        if (domainModelSection) {
            resizeObserver.observe(domainModelSection);
        }
    }
    
    // Also handle window resize events
    window.addEventListener('resize', adjustHeights);
    
    // Initial adjustment
    adjustHeights();
    
    // Special handling for when results are displayed
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function() {
            // Adjust heights after a slight delay to allow content to render
            setTimeout(adjustHeights, 2000);
        });
    }
    
    // Special handling for UML diagram image loading
    const umlDiagram = document.getElementById('uml-diagram');
    if (umlDiagram) {
        umlDiagram.addEventListener('load', function() {
            // When the image loads, make sure it fits properly
            adjustHeights();
            
            // Make sure the UML container scrolls if needed
            const actualHeight = umlDiagram.naturalHeight;
            const actualWidth = umlDiagram.naturalWidth;
            
            if (umlContainer) {
                if (actualHeight > umlContainer.clientHeight || 
                    actualWidth > umlContainer.clientWidth) {
                    umlContainer.style.overflow = 'auto';
                }
            }
        });
    }
});