// Search Interface - Search NCERT content
document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    const searchResultsContainer = document.getElementById('search-results-container');
    const searchLoading = document.getElementById('search-loading');
    const noResults = document.getElementById('no-results');
    const resultsCount = document.getElementById('results-count');
    const previewPanel = document.getElementById('preview-panel');
    const previewContent = document.getElementById('preview-content');
    const closePreview = document.getElementById('close-preview');
    const autocompleteSuggestions = document.getElementById('autocomplete-suggestions');
    
    // Filters
    const filterSubject = document.getElementById('filter-subject');
    const filterClass = document.getElementById('filter-class');
    const filterChapter = document.getElementById('filter-chapter');
    const clearFiltersBtn = document.getElementById('clear-filters');

    let searchTimeout = null;
    let currentResults = [];

    // Initialize
    function initialize() {
        setupEventListeners();
        loadChapterOptions();
    }

    initialize();

    // Setup event listeners
    function setupEventListeners() {
        searchForm.addEventListener('submit', handleSearch);
        searchInput.addEventListener('input', handleAutocomplete);
        closePreview.addEventListener('click', hidePreview);
        clearFiltersBtn.addEventListener('click', clearFilters);
        
        filterSubject.addEventListener('change', () => {
            loadChapterOptions();
            if (currentResults.length > 0) {
                filterResults();
            }
        });
        
        filterClass.addEventListener('change', () => {
            loadChapterOptions();
            if (currentResults.length > 0) {
                filterResults();
            }
        });
        
        filterChapter.addEventListener('change', () => {
            if (currentResults.length > 0) {
                filterResults();
            }
        });
    }

    // Handle search
    async function handleSearch(e) {
        e.preventDefault();
        
        const query = searchInput.value.trim();
        if (!query) return;

        showLoading(true);
        hidePreview();
        autocompleteSuggestions.classList.add('hidden');

        try {
            const params = new URLSearchParams({
                q: query,
                subject: filterSubject.value,
                class: filterClass.value,
                chapter: filterChapter.value
            });

            const response = await fetch(`/api/search?${params}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Search failed');
            }

            const data = await response.json();
            currentResults = data.results || [];
            
            // Debug: Log first result to see structure
            if (currentResults.length > 0) {
                console.log('First search result structure:', currentResults[0]);
            }
            
            displayResults(currentResults);
            
            if (data.suggestions && currentResults.length === 0) {
                displaySuggestions(data.suggestions);
            }

        } catch (error) {
            console.error('Search error:', error);
            showError('Search failed. Please try again.');
        } finally {
            showLoading(false);
        }
    }

    // Handle autocomplete
    function handleAutocomplete(e) {
        clearTimeout(searchTimeout);
        
        const query = e.target.value.trim();
        if (query.length < 2) {
            autocompleteSuggestions.classList.add('hidden');
            return;
        }

        searchTimeout = setTimeout(async () => {
            try {
                const response = await fetch(`/api/search/autocomplete?q=${encodeURIComponent(query)}`, {
                    credentials: 'include'
                });

                if (!response.ok) return;

                const data = await response.json();
                displayAutocomplete(data.suggestions || []);

            } catch (error) {
                console.error('Autocomplete error:', error);
            }
        }, 300);
    }

    // Display autocomplete suggestions
    function displayAutocomplete(suggestions) {
        if (suggestions.length === 0) {
            autocompleteSuggestions.classList.add('hidden');
            return;
        }

        autocompleteSuggestions.innerHTML = '';
        suggestions.slice(0, 5).forEach(suggestion => {
            const div = document.createElement('div');
            div.className = 'suggestion-item';
            div.textContent = suggestion;
            div.addEventListener('click', () => {
                searchInput.value = suggestion;
                autocompleteSuggestions.classList.add('hidden');
                handleSearch(new Event('submit'));
            });
            autocompleteSuggestions.appendChild(div);
        });

        autocompleteSuggestions.classList.remove('hidden');
    }

    // Display search results
    function displayResults(results) {
        searchResults.innerHTML = '';
        
        if (results.length === 0) {
            searchResultsContainer.classList.add('hidden');
            noResults.classList.remove('hidden');
            return;
        }

        searchResultsContainer.classList.remove('hidden');
        noResults.classList.add('hidden');
        resultsCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''} found`;

        results.forEach((result, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'search-result-item';
            
            // Extract metadata (handle both nested and flat structures)
            const metadata = result.metadata || {};
            const subject = metadata.subject || result.subject || 'NCERT';
            const classLevel = metadata.class || metadata.class_level || result.class_level || result.class || 'N/A';
            const chapter = metadata.chapter || result.chapter || 'N/A';
            const page = metadata.page || metadata.page_number || result.page_number || result.page || 'N/A';
            
            // Highlight search terms
            const highlightedContent = highlightSearchTerms(result.content, searchInput.value);
            
            // Calculate relevance percentage
            const relevance = result.relevance_score || result.relevance || 0;
            
            resultDiv.innerHTML = `
                <div class="result-header">
                    <h3>${subject} - Class ${classLevel}</h3>
                    <span class="result-meta">Chapter ${chapter} | Page ${page}</span>
                </div>
                <div class="result-content">
                    <p>${highlightedContent}</p>
                </div>
                <div class="result-actions">
                    <button class="preview-btn" data-index="${index}">Preview</button>
                    <span class="relevance-score">Relevance: ${Math.round(relevance * 100)}%</span>
                </div>
            `;
            
            // Add diagram if available
            if (result.diagrams && result.diagrams.length > 0) {
                const diagramDiv = document.createElement('div');
                diagramDiv.className = 'result-diagrams';
                result.diagrams.forEach(diagram => {
                    const img = document.createElement('img');
                    img.src = diagram.path;
                    img.alt = diagram.caption;
                    img.className = 'result-diagram-thumb';
                    diagramDiv.appendChild(img);
                });
                resultDiv.appendChild(diagramDiv);
            }
            
            searchResults.appendChild(resultDiv);
        });

        // Add event listeners to preview buttons
        document.querySelectorAll('.preview-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                showPreview(results[index]);
            });
        });
    }

    // Highlight search terms
    function highlightSearchTerms(text, query) {
        if (!query) return text;
        
        const terms = query.split(' ').filter(t => t.length > 2);
        let highlighted = text;
        
        terms.forEach(term => {
            const regex = new RegExp(`(${term})`, 'gi');
            highlighted = highlighted.replace(regex, '<mark>$1</mark>');
        });
        
        return highlighted;
    }

    // Show preview
    function showPreview(result) {
        // Extract metadata (handle both nested and flat structures)
        const metadata = result.metadata || {};
        const subject = metadata.subject || result.subject || 'NCERT';
        const classLevel = metadata.class || metadata.class_level || result.class_level || result.class || 'N/A';
        const chapter = metadata.chapter || result.chapter || 'N/A';
        const page = metadata.page || metadata.page_number || result.page_number || result.page || 'N/A';
        const section = metadata.section || result.section || '';
        
        previewContent.innerHTML = `
            <div class="preview-header-info">
                <h3>${subject} - Class ${classLevel}</h3>
                <p>Chapter ${chapter}${section ? ', Section ' + section : ''}</p>
                <p>Page ${page}</p>
            </div>
            <div class="preview-text">
                <h4>Content:</h4>
                <p>${result.full_content || result.content}</p>
            </div>
        `;
        
        // Add diagrams if available
        if (result.diagrams && result.diagrams.length > 0) {
            const diagramsDiv = document.createElement('div');
            diagramsDiv.className = 'preview-diagrams';
            diagramsDiv.innerHTML = '<h4>Related Diagrams:</h4>';
            
            result.diagrams.forEach(diagram => {
                const diagramItem = document.createElement('div');
                diagramItem.className = 'preview-diagram-item';
                diagramItem.innerHTML = `
                    <img src="${diagram.path}" alt="${diagram.caption}">
                    <p>${diagram.caption}</p>
                `;
                diagramsDiv.appendChild(diagramItem);
            });
            
            previewContent.appendChild(diagramsDiv);
        }
        
        // Add action buttons
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'preview-actions';
        actionsDiv.innerHTML = `
            <button class="action-btn" onclick="window.location.href='/?query=${encodeURIComponent(result.content.substring(0, 100))}'">
                Ask about this
            </button>
        `;
        previewContent.appendChild(actionsDiv);
        
        previewPanel.classList.remove('hidden');
    }

    // Hide preview
    function hidePreview() {
        previewPanel.classList.add('hidden');
    }

    // Display suggestions for no results
    function displaySuggestions(suggestions) {
        const suggestionsContainer = document.getElementById('suggestions-container');
        if (!suggestionsContainer) return;
        
        suggestionsContainer.innerHTML = '<h4>Try searching for:</h4>';
        
        suggestions.forEach(suggestion => {
            const btn = document.createElement('button');
            btn.className = 'suggestion-btn';
            btn.textContent = suggestion;
            btn.addEventListener('click', () => {
                searchInput.value = suggestion;
                handleSearch(new Event('submit'));
            });
            suggestionsContainer.appendChild(btn);
        });
    }

    // Filter results
    function filterResults() {
        const subject = filterSubject.value;
        const classLevel = filterClass.value;
        const chapter = filterChapter.value;

        const filtered = currentResults.filter(result => {
            // Extract metadata (handle both nested and flat structures)
            const metadata = result.metadata || {};
            const resultSubject = metadata.subject || result.subject || '';
            const resultClass = (metadata.class || metadata.class_level || result.class_level || result.class || '').toString();
            const resultChapter = (metadata.chapter || result.chapter || '').toString();
            
            if (subject && resultSubject !== subject) return false;
            if (classLevel && resultClass !== classLevel) return false;
            if (chapter && resultChapter !== chapter) return false;
            return true;
        });

        displayResults(filtered);
    }

    // Load chapter options based on selected subject and class
    async function loadChapterOptions() {
        const subject = filterSubject.value;
        const classLevel = filterClass.value;

        if (!subject || !classLevel) {
            filterChapter.disabled = true;
            filterChapter.innerHTML = '<option value="">Select subject and class first</option>';
            return;
        }

        try {
            const response = await fetch(`/api/chapters?subject=${subject}&class=${classLevel}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to load chapters');
            }

            const data = await response.json();
            const chapters = data.chapters || [];

            filterChapter.disabled = false;
            filterChapter.innerHTML = '<option value="">All Chapters</option>';
            
            chapters.forEach(chapter => {
                const option = document.createElement('option');
                option.value = chapter.number;
                option.textContent = `Chapter ${chapter.number}: ${chapter.name}`;
                filterChapter.appendChild(option);
            });

        } catch (error) {
            console.error('Error loading chapters:', error);
            filterChapter.disabled = true;
        }
    }

    // Clear filters
    function clearFilters() {
        filterSubject.value = '';
        filterClass.value = '';
        filterChapter.value = '';
        filterChapter.disabled = true;
        filterChapter.innerHTML = '<option value="">Select subject first</option>';
        
        if (currentResults.length > 0) {
            displayResults(currentResults);
        }
    }

    // Show/hide loading
    function showLoading(show) {
        searchLoading.classList.toggle('hidden', !show);
    }

    // Show error
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-banner';
        errorDiv.textContent = message;
        document.querySelector('.search-container').prepend(errorDiv);
        
        setTimeout(() => errorDiv.remove(), 5000);
    }

    // Close autocomplete on outside click
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !autocompleteSuggestions.contains(e.target)) {
            autocompleteSuggestions.classList.add('hidden');
        }
    });
});
