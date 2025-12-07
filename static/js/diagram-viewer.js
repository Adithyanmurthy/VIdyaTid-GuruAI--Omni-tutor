/**
 * Diagram Viewer Component for GuruAI
 * 
 * Provides:
 * - Diagram display in chat responses
 * - Zoom and pan functionality
 * - Caption and reference display
 * - Labeled parts explanation
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
 */

class DiagramViewer {
    constructor() {
        this.currentDiagram = null;
        this.scale = 1.0;
        this.minScale = 0.5;
        this.maxScale = 3.0;
        this.isPanning = false;
        this.startX = 0;
        this.startY = 0;
        this.translateX = 0;
        this.translateY = 0;
    }

    /**
     * Display diagram in chat message
     * @param {Object} diagram - Diagram metadata
     * @param {HTMLElement} container - Container element
     */
    displayDiagram(diagram, container) {
        // Ensure diagram data is properly formatted
        const formattedDiagram = this.formatDiagramData(diagram);
        
        const diagramHtml = this.createDiagramHTML(formattedDiagram);
        container.innerHTML += diagramHtml;
        
        // Initialize zoom and pan for this diagram
        const diagramElement = container.querySelector(`#diagram-${formattedDiagram.page_id}`);
        if (diagramElement) {
            this.initializeZoomPan(diagramElement, formattedDiagram.page_id);
        }
    }
    
    /**
     * Format diagram data to ensure all fields are properly typed
     * @param {Object} diagram - Raw diagram data
     * @returns {Object} Formatted diagram data
     */
    formatDiagramData(diagram) {
        const formatted = { ...diagram };
        
        // Ensure figures is an array
        if (typeof formatted.figures === 'string') {
            formatted.figures = formatted.figures.split(',').map(f => f.trim());
        } else if (!Array.isArray(formatted.figures)) {
            formatted.figures = [];
        }
        
        // Ensure captions is an object
        if (typeof formatted.captions === 'string') {
            try {
                formatted.captions = JSON.parse(formatted.captions);
            } catch (e) {
                console.warn('Failed to parse captions:', e);
                formatted.captions = {};
            }
        } else if (!formatted.captions || typeof formatted.captions !== 'object') {
            formatted.captions = {};
        }
        
        return formatted;
    }

    /**
     * Create HTML for diagram display
     * @param {Object} diagram - Diagram metadata
     * @returns {string} HTML string
     */
    createDiagramHTML(diagram) {
        const figuresList = diagram.figures.join(', ');
        const captionsHTML = this.createCaptionsHTML(diagram.captions);
        
        // Create a clear title showing both figure number and chapter context
        const figureTitle = diagram.figures.length === 1 
            ? `Figure ${figuresList}` 
            : `Figures ${figuresList}`;
        
        return `
            <div class="diagram-container" id="diagram-${diagram.page_id}">
                <div class="diagram-header">
                    <h4>${figureTitle}</h4>
                    <span class="diagram-reference">${diagram.reference}</span>
                </div>
                
                <div class="diagram-viewer-wrapper">
                    <div class="diagram-controls">
                        <button class="zoom-btn" onclick="diagramViewer.zoomIn('${diagram.page_id}')">
                            <span>🔍+</span>
                        </button>
                        <button class="zoom-btn" onclick="diagramViewer.zoomOut('${diagram.page_id}')">
                            <span>🔍-</span>
                        </button>
                        <button class="zoom-btn" onclick="diagramViewer.resetZoom('${diagram.page_id}')">
                            <span>↺</span>
                        </button>
                        <button class="zoom-btn" onclick="diagramViewer.showLabeledParts('${diagram.page_id}')">
                            <span>🏷️</span>
                        </button>
                    </div>
                    
                    <div class="diagram-image-container" id="img-container-${diagram.page_id}">
                        <img 
                            src="/api/diagrams/file/${diagram.page_id}" 
                            alt="Figure ${figuresList}"
                            class="diagram-image"
                            id="img-${diagram.page_id}"
                            draggable="false"
                        />
                    </div>
                </div>
                
                ${captionsHTML}
                
                ${diagram.display_note ? `
                <div class="diagram-note">
                    <span class="note-icon">ℹ️</span>
                    <span>${diagram.display_note}</span>
                </div>
                ` : ''}
                
                <div class="labeled-parts-container" id="labeled-parts-${diagram.page_id}" style="display: none;">
                    <div class="loading">Loading labeled parts...</div>
                </div>
            </div>
        `;
    }

    /**
     * Create HTML for captions
     * @param {Object|string} captions - Captions object {figure_number: caption} or JSON string
     * @returns {string} HTML string
     */
    createCaptionsHTML(captions) {
        // Debug logging
        console.log('createCaptionsHTML called with:', typeof captions, captions);
        
        // Handle case where captions might be a JSON string
        let captionsObj = captions;
        if (typeof captions === 'string') {
            try {
                captionsObj = JSON.parse(captions);
                console.log('Parsed captions from string:', captionsObj);
            } catch (e) {
                console.warn('Failed to parse captions JSON:', e, 'Value:', captions);
                return '';
            }
        }
        
        // Validate captionsObj is a proper object (not an array or null)
        if (!captionsObj || 
            typeof captionsObj !== 'object' || 
            Array.isArray(captionsObj) ||
            Object.keys(captionsObj).length === 0) {
            console.log('No valid captions to display');
            return '';
        }

        let html = '<div class="diagram-captions">';
        let captionCount = 0;
        
        for (const [figNum, caption] of Object.entries(captionsObj)) {
            // Skip if caption is empty, invalid, or not a string
            if (!caption || typeof caption !== 'string' || caption.trim().length === 0) {
                console.log(`Skipping invalid caption for figure ${figNum}:`, caption);
                continue;
            }
            
            html += `
                <div class="caption-item">
                    <strong>Figure ${figNum}:</strong> ${caption}
                </div>
            `;
            captionCount++;
        }
        html += '</div>';
        
        console.log(`Generated HTML for ${captionCount} captions`);
        return captionCount > 0 ? html : '';
    }

    /**
     * Initialize zoom and pan functionality
     * @param {HTMLElement} container - Diagram container
     * @param {string} pageId - Page ID
     */
    initializeZoomPan(container, pageId) {
        const imageContainer = container.querySelector(`#img-container-${pageId}`);
        const image = container.querySelector(`#img-${pageId}`);
        
        if (!imageContainer || !image) return;

        // Mouse wheel zoom
        imageContainer.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            this.zoom(pageId, delta);
        });

        // Pan with mouse drag
        image.addEventListener('mousedown', (e) => {
            this.isPanning = true;
            this.startX = e.clientX - this.translateX;
            this.startY = e.clientY - this.translateY;
            image.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!this.isPanning) return;
            
            this.translateX = e.clientX - this.startX;
            this.translateY = e.clientY - this.startY;
            this.updateTransform(pageId);
        });

        document.addEventListener('mouseup', () => {
            if (this.isPanning) {
                this.isPanning = false;
                image.style.cursor = 'grab';
            }
        });

        // Touch support for mobile
        let touchStartX = 0;
        let touchStartY = 0;

        image.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                touchStartX = e.touches[0].clientX - this.translateX;
                touchStartY = e.touches[0].clientY - this.translateY;
            }
        });

        image.addEventListener('touchmove', (e) => {
            if (e.touches.length === 1) {
                e.preventDefault();
                this.translateX = e.touches[0].clientX - touchStartX;
                this.translateY = e.touches[0].clientY - touchStartY;
                this.updateTransform(pageId);
            }
        });

        // Set initial cursor
        image.style.cursor = 'grab';
    }

    /**
     * Zoom in
     * @param {string} pageId - Page ID
     */
    zoomIn(pageId) {
        this.zoom(pageId, 0.2);
    }

    /**
     * Zoom out
     * @param {string} pageId - Page ID
     */
    zoomOut(pageId) {
        this.zoom(pageId, -0.2);
    }

    /**
     * Apply zoom
     * @param {string} pageId - Page ID
     * @param {number} delta - Zoom delta
     */
    zoom(pageId, delta) {
        this.scale = Math.max(this.minScale, Math.min(this.maxScale, this.scale + delta));
        this.updateTransform(pageId);
    }

    /**
     * Reset zoom and pan
     * @param {string} pageId - Page ID
     */
    resetZoom(pageId) {
        this.scale = 1.0;
        this.translateX = 0;
        this.translateY = 0;
        this.updateTransform(pageId);
    }

    /**
     * Update image transform
     * @param {string} pageId - Page ID
     */
    updateTransform(pageId) {
        const image = document.getElementById(`img-${pageId}`);
        if (image) {
            image.style.transform = `translate(${this.translateX}px, ${this.translateY}px) scale(${this.scale})`;
        }
    }

    /**
     * Show labeled parts explanation
     * @param {string} pageId - Page ID
     */
    async showLabeledParts(pageId) {
        const container = document.getElementById(`labeled-parts-${pageId}`);
        if (!container) return;

        // Toggle visibility
        if (container.style.display === 'block') {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';
        container.innerHTML = '<div class="loading">Loading labeled parts...</div>';

        try {
            const response = await fetch(`/api/diagrams/labeled-parts/${pageId}`);
            
            if (!response.ok) {
                throw new Error('Failed to load labeled parts');
            }

            const data = await response.json();
            container.innerHTML = this.createLabeledPartsHTML(data);
            
        } catch (error) {
            console.error('Error loading labeled parts:', error);
            container.innerHTML = '<div class="error">Failed to load labeled parts</div>';
        }
    }

    /**
     * Create HTML for labeled parts
     * @param {Object} data - Labeled parts data
     * @returns {string} HTML string
     */
    createLabeledPartsHTML(data) {
        if (!data.labeled_parts || Object.keys(data.labeled_parts).length === 0) {
            return '<div class="info">No labeled parts found in this diagram.</div>';
        }

        let html = '<div class="labeled-parts-content"><h5>Labeled Parts:</h5>';
        
        for (const [figNum, parts] of Object.entries(data.labeled_parts)) {
            html += `<div class="figure-parts"><strong>Figure ${figNum}:</strong><ul>`;
            
            for (const part of parts) {
                html += `<li><strong>${part.label}:</strong> ${part.description}</li>`;
            }
            
            html += '</ul></div>';
        }
        
        html += '</div>';
        return html;
    }

    /**
     * Display multiple diagrams
     * @param {Array} diagrams - Array of diagram objects
     * @param {HTMLElement} container - Container element
     */
    displayMultipleDiagrams(diagrams, container) {
        if (!diagrams || diagrams.length === 0) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'diagrams-wrapper';
        
        diagrams.forEach(diagram => {
            // displayDiagram will handle formatting
            this.displayDiagram(diagram, wrapper);
        });
        
        container.appendChild(wrapper);
    }

    /**
     * Search and display diagrams for a concept
     * @param {string} concept - Concept to search for
     * @param {string} subject - Optional subject filter
     * @param {HTMLElement} container - Container element
     */
    async searchAndDisplayDiagrams(concept, subject, container) {
        try {
            let url = `/api/diagrams/for-concept?concept=${encodeURIComponent(concept)}`;
            if (subject) {
                url += `&subject=${encodeURIComponent(subject)}`;
            }

            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error('Failed to search diagrams');
            }

            const data = await response.json();
            
            if (data.diagrams && data.diagrams.length > 0) {
                this.displayMultipleDiagrams(data.diagrams, container);
            }
            
        } catch (error) {
            console.error('Error searching diagrams:', error);
        }
    }
}

// Create global instance
const diagramViewer = new DiagramViewer();
