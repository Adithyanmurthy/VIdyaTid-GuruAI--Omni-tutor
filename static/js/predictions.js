/**
 * VidyaTid - Predictions API Client
 * Handles all API communication for NEET prediction features
 */

/**
 * Chart.js Configuration and Setup
 * Dark theme color palette and default options
 */
const ChartConfig = {
    // Dark theme color palette
    colors: {
        primary: '#3b82f6',
        primaryLight: '#60a5fa',
        primaryDark: '#2563eb',
        purple: '#8b5cf6',
        green: '#10b981',
        greenLight: '#34d399',
        yellow: '#f59e0b',
        yellowLight: '#fbbf24',
        red: '#ef4444',
        redLight: '#f87171',
        gray: '#64748b',
        grayLight: '#94a3b8',
        background: 'rgba(15, 23, 42, 0.8)',
        border: 'rgba(59, 130, 246, 0.2)',
        text: '#ffffff',
        textSecondary: '#94a3b8'
    },

    // Gradient colors for charts
    gradients: {
        blue: ['#3b82f6', '#8b5cf6'],
        green: ['#10b981', '#059669'],
        yellow: ['#f59e0b', '#d97706'],
        red: ['#ef4444', '#dc2626'],
        multi: ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
    },

    // Default chart options with responsive configuration
    defaultOptions: {
        responsive: true,
        maintainAspectRatio: false,
        // Responsive resize delay to prevent excessive redraws
        resizeDelay: 100,
        // Device pixel ratio for high DPI displays
        devicePixelRatio: window.devicePixelRatio || 1,
        plugins: {
            legend: {
                display: true,
                position: 'bottom',
                labels: {
                    color: '#94a3b8',
                    font: {
                        family: 'Inter, sans-serif',
                        size: window.innerWidth < 480 ? 10 : (window.innerWidth < 768 ? 11 : 12),
                        weight: 500
                    },
                    padding: window.innerWidth < 480 ? 10 : 15,
                    usePointStyle: true,
                    pointStyle: 'circle',
                    boxWidth: window.innerWidth < 480 ? 8 : 12,
                    boxHeight: window.innerWidth < 480 ? 8 : 12
                }
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                titleColor: '#ffffff',
                bodyColor: '#94a3b8',
                borderColor: 'rgba(59, 130, 246, 0.4)',
                borderWidth: 1,
                padding: window.innerWidth < 480 ? 8 : 12,
                cornerRadius: 8,
                displayColors: true,
                titleFont: {
                    family: 'Inter, sans-serif',
                    size: window.innerWidth < 480 ? 12 : 14,
                    weight: 600
                },
                bodyFont: {
                    family: 'Inter, sans-serif',
                    size: window.innerWidth < 480 ? 11 : 13,
                    weight: 400
                },
                // Touch-friendly tooltip on mobile
                mode: window.innerWidth < 768 ? 'nearest' : 'index',
                intersect: window.innerWidth < 768 ? true : false,
                callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || '';
                        if (label) {
                            label += ': ';
                        }
                        if (context.parsed.y !== null) {
                            label += context.parsed.y.toFixed(1) + '%';
                        }
                        return label;
                    }
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(59, 130, 246, 0.1)',
                    drawBorder: false
                },
                ticks: {
                    color: '#94a3b8',
                    font: {
                        family: 'Inter, sans-serif',
                        size: window.innerWidth < 480 ? 9 : (window.innerWidth < 768 ? 10 : 11)
                    },
                    // Auto-skip labels on small screens
                    autoSkip: true,
                    autoSkipPadding: window.innerWidth < 768 ? 20 : 10,
                    maxRotation: window.innerWidth < 768 ? 45 : 0,
                    minRotation: 0
                }
            },
            y: {
                grid: {
                    color: 'rgba(59, 130, 246, 0.1)',
                    drawBorder: false
                },
                ticks: {
                    color: '#94a3b8',
                    font: {
                        family: 'Inter, sans-serif',
                        size: window.innerWidth < 480 ? 9 : (window.innerWidth < 768 ? 10 : 11)
                    },
                    callback: function(value) {
                        return value + '%';
                    },
                    // Reduce number of ticks on mobile
                    maxTicksLimit: window.innerWidth < 768 ? 5 : 8
                }
            }
        },
        animation: {
            duration: window.innerWidth < 768 ? 400 : 800,
            easing: 'easeInOutQuart'
        },
        // Interaction modes for touch devices
        interaction: {
            mode: window.innerWidth < 768 ? 'nearest' : 'index',
            intersect: window.innerWidth < 768 ? true : false
        }
    },

    /**
     * Create gradient for chart background
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array<string>} colors - Array of color stops
     * @returns {CanvasGradient} - Gradient object
     */
    createGradient(ctx, colors) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        colors.forEach((color, index) => {
            gradient.addColorStop(index / (colors.length - 1), color);
        });
        return gradient;
    },

    /**
     * Get color based on probability/confidence score
     * @param {number} score - Score value (0-1 or 0-100)
     * @returns {string} - Color hex code
     */
    getColorByScore(score) {
        const normalizedScore = score > 1 ? score / 100 : score;
        if (normalizedScore >= 0.75) return this.colors.green;
        if (normalizedScore >= 0.5) return this.colors.yellow;
        return this.colors.red;
    }
};

// Set Chart.js global defaults
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = ChartConfig.colors.textSecondary;
    Chart.defaults.font.family = 'Inter, sans-serif';
    Chart.defaults.plugins.legend.display = true;
}

/**
 * Update chart responsive options based on window size
 * Call this function when window is resized
 */
ChartConfig.updateResponsiveOptions = function() {
    const width = window.innerWidth;
    
    // Update font sizes
    const legendFontSize = width < 480 ? 10 : (width < 768 ? 11 : 12);
    const tooltipTitleSize = width < 480 ? 12 : 14;
    const tooltipBodySize = width < 480 ? 11 : 13;
    const tickFontSize = width < 480 ? 9 : (width < 768 ? 10 : 11);
    
    // Update padding
    const legendPadding = width < 480 ? 10 : 15;
    const tooltipPadding = width < 480 ? 8 : 12;
    
    // Update legend box size
    const boxSize = width < 480 ? 8 : 12;
    
    // Update animation duration
    const animationDuration = width < 768 ? 400 : 800;
    
    // Update interaction mode
    const isMobile = width < 768;
    const interactionMode = isMobile ? 'nearest' : 'index';
    const intersect = isMobile;
    
    // Update default options
    this.defaultOptions.plugins.legend.labels.font.size = legendFontSize;
    this.defaultOptions.plugins.legend.labels.padding = legendPadding;
    this.defaultOptions.plugins.legend.labels.boxWidth = boxSize;
    this.defaultOptions.plugins.legend.labels.boxHeight = boxSize;
    
    this.defaultOptions.plugins.tooltip.titleFont.size = tooltipTitleSize;
    this.defaultOptions.plugins.tooltip.bodyFont.size = tooltipBodySize;
    this.defaultOptions.plugins.tooltip.padding = tooltipPadding;
    this.defaultOptions.plugins.tooltip.mode = interactionMode;
    this.defaultOptions.plugins.tooltip.intersect = intersect;
    
    this.defaultOptions.scales.x.ticks.font.size = tickFontSize;
    this.defaultOptions.scales.x.ticks.autoSkipPadding = isMobile ? 20 : 10;
    this.defaultOptions.scales.x.ticks.maxRotation = isMobile ? 45 : 0;
    
    this.defaultOptions.scales.y.ticks.font.size = tickFontSize;
    this.defaultOptions.scales.y.ticks.maxTicksLimit = isMobile ? 5 : 8;
    
    this.defaultOptions.animation.duration = animationDuration;
    this.defaultOptions.interaction.mode = interactionMode;
    this.defaultOptions.interaction.intersect = intersect;
};

// Store all chart instances for responsive updates
ChartConfig.chartInstances = [];

/**
 * Register a chart instance for responsive updates
 * @param {Chart} chart - Chart.js instance
 */
ChartConfig.registerChart = function(chart) {
    if (chart && !this.chartInstances.includes(chart)) {
        this.chartInstances.push(chart);
    }
};

/**
 * Update all registered charts on resize
 */
ChartConfig.updateAllCharts = function() {
    this.updateResponsiveOptions();
    this.chartInstances.forEach(chart => {
        if (chart && !chart.destroyed) {
            chart.resize();
        }
    });
};

// Debounced resize handler
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        ChartConfig.updateAllCharts();
    }, 150);
});

/**
 * Cache Manager for Predictions
 * Handles sessionStorage caching with expiration
 */
class CacheManager {
    constructor() {
        this.prefix = 'predictions_cache_';
        this.defaultTTL = 30 * 60 * 1000; // 30 minutes in milliseconds
    }

    /**
     * Generate cache key
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Request parameters
     * @returns {string} - Cache key
     */
    generateKey(endpoint, params = {}) {
        const paramString = JSON.stringify(params);
        return `${this.prefix}${endpoint}_${paramString}`;
    }

    /**
     * Set cache entry
     * @param {string} key - Cache key
     * @param {*} data - Data to cache
     * @param {number} ttl - Time to live in milliseconds (optional)
     */
    set(key, data, ttl = this.defaultTTL) {
        try {
            const cacheEntry = {
                data: data,
                timestamp: Date.now(),
                expiry: Date.now() + ttl
            };
            const cacheString = JSON.stringify(cacheEntry);
            
            // Don't cache if data is too large (> 2MB)
            if (cacheString.length > 2 * 1024 * 1024) {
                console.warn('Data too large to cache, skipping cache for:', key);
                return;
            }
            
            sessionStorage.setItem(key, cacheString);
        } catch (error) {
            console.warn('Failed to set cache:', error);
            // If storage is full, clear old entries
            if (error.name === 'QuotaExceededError') {
                this.clearExpired();
                // Try again with smaller data
                try {
                    const cacheEntry = {
                        data: data,
                        timestamp: Date.now(),
                        expiry: Date.now() + ttl
                    };
                    const cacheString = JSON.stringify(cacheEntry);
                    
                    // Skip if still too large
                    if (cacheString.length > 2 * 1024 * 1024) {
                        console.warn('Data too large to cache after clearing');
                        return;
                    }
                    
                    sessionStorage.setItem(key, cacheString);
                } catch (retryError) {
                    console.warn('Failed to set cache after clearing, skipping cache');
                }
            }
        }
    }

    /**
     * Get cache entry
     * @param {string} key - Cache key
     * @returns {*} - Cached data or null if not found/expired
     */
    get(key) {
        try {
            const cached = sessionStorage.getItem(key);
            if (!cached) {
                return null;
            }

            const cacheEntry = JSON.parse(cached);
            
            // Check if expired
            if (Date.now() > cacheEntry.expiry) {
                sessionStorage.removeItem(key);
                return null;
            }

            return cacheEntry.data;
        } catch (error) {
            console.warn('Failed to get cache:', error);
            return null;
        }
    }

    /**
     * Check if cache entry exists and is valid
     * @param {string} key - Cache key
     * @returns {boolean} - True if valid cache exists
     */
    has(key) {
        return this.get(key) !== null;
    }

    /**
     * Remove cache entry
     * @param {string} key - Cache key
     */
    remove(key) {
        try {
            sessionStorage.removeItem(key);
        } catch (error) {
            console.warn('Failed to remove cache:', error);
        }
    }

    /**
     * Clear all prediction caches
     */
    clearAll() {
        try {
            const keys = Object.keys(sessionStorage);
            keys.forEach(key => {
                if (key.startsWith(this.prefix)) {
                    sessionStorage.removeItem(key);
                }
            });
        } catch (error) {
            console.warn('Failed to clear all caches:', error);
        }
    }

    /**
     * Clear expired cache entries
     */
    clearExpired() {
        try {
            const keys = Object.keys(sessionStorage);
            const now = Date.now();
            
            keys.forEach(key => {
                if (key.startsWith(this.prefix)) {
                    try {
                        const cached = sessionStorage.getItem(key);
                        if (cached) {
                            const cacheEntry = JSON.parse(cached);
                            if (now > cacheEntry.expiry) {
                                sessionStorage.removeItem(key);
                            }
                        }
                    } catch (error) {
                        // Remove corrupted entries
                        sessionStorage.removeItem(key);
                    }
                }
            });
        } catch (error) {
            console.warn('Failed to clear expired caches:', error);
        }
    }

    /**
     * Get cache statistics
     * @returns {Object} - Cache stats
     */
    getStats() {
        try {
            const keys = Object.keys(sessionStorage);
            const predictionKeys = keys.filter(key => key.startsWith(this.prefix));
            const now = Date.now();
            
            let validCount = 0;
            let expiredCount = 0;
            let totalSize = 0;

            predictionKeys.forEach(key => {
                try {
                    const cached = sessionStorage.getItem(key);
                    if (cached) {
                        totalSize += cached.length;
                        const cacheEntry = JSON.parse(cached);
                        if (now > cacheEntry.expiry) {
                            expiredCount++;
                        } else {
                            validCount++;
                        }
                    }
                } catch (error) {
                    expiredCount++;
                }
            });

            return {
                total: predictionKeys.length,
                valid: validCount,
                expired: expiredCount,
                sizeBytes: totalSize,
                sizeKB: (totalSize / 1024).toFixed(2)
            };
        } catch (error) {
            console.warn('Failed to get cache stats:', error);
            return {
                total: 0,
                valid: 0,
                expired: 0,
                sizeBytes: 0,
                sizeKB: '0.00'
            };
        }
    }
}

class PredictionAPI {
    constructor() {
        this.baseURL = '/api/prediction';
        this.maxRetries = 3;
        this.retryDelay = 1000; // 1 second
        this.cache = new CacheManager();
        this.pendingRequests = new Map(); // Track pending requests for cancellation
        this.debouncedCalls = new Map(); // Track debounced function calls
    }

    /**
     * Debounce a function call
     * @param {string} key - Unique key for the debounced call
     * @param {Function} func - Function to debounce
     * @param {number} delay - Delay in milliseconds
     * @returns {Promise} - Promise that resolves with function result
     */
    debounce(key, func, delay = 300) {
        return new Promise((resolve, reject) => {
            // Clear existing timeout for this key
            if (this.debouncedCalls.has(key)) {
                const existing = this.debouncedCalls.get(key);
                clearTimeout(existing.timeout);
                // Reject the previous promise
                existing.reject(new Error('DEBOUNCED'));
            }

            // Set new timeout
            const timeout = setTimeout(async () => {
                try {
                    const result = await func();
                    this.debouncedCalls.delete(key);
                    resolve(result);
                } catch (error) {
                    this.debouncedCalls.delete(key);
                    reject(error);
                }
            }, delay);

            // Store timeout and promise resolvers
            this.debouncedCalls.set(key, { timeout, resolve, reject });
        });
    }

    /**
     * Throttle a function call
     * @param {string} key - Unique key for the throttled call
     * @param {Function} func - Function to throttle
     * @param {number} limit - Minimum time between calls in milliseconds
     * @returns {Promise} - Promise that resolves with function result or null if throttled
     */
    async throttle(key, func, limit = 1000) {
        const now = Date.now();
        const lastCall = this.lastThrottleCalls?.get(key) || 0;
        
        if (!this.lastThrottleCalls) {
            this.lastThrottleCalls = new Map();
        }
        
        // Check if enough time has passed
        if (now - lastCall < limit) {
            console.log(`Throttled call for ${key}, waiting ${limit - (now - lastCall)}ms`);
            return null;
        }
        
        // Update last call time
        this.lastThrottleCalls.set(key, now);
        
        // Execute function
        return await func();
    }

    /**
     * Cancel a pending request
     * @param {string} requestKey - Unique key for the request
     */
    cancelRequest(requestKey) {
        if (this.pendingRequests.has(requestKey)) {
            const controller = this.pendingRequests.get(requestKey);
            controller.abort();
            this.pendingRequests.delete(requestKey);
            console.log('Request cancelled:', requestKey);
        }
    }

    /**
     * Cancel all pending requests
     */
    cancelAllRequests() {
        this.pendingRequests.forEach((controller, key) => {
            controller.abort();
            console.log('Request cancelled:', key);
        });
        this.pendingRequests.clear();
    }

    /**
     * Generic error handler wrapper for API calls with retry logic and cancellation support
     * @param {Function} apiCall - The API call function to wrap
     * @param {number} retryCount - Current retry attempt (default: 0)
     * @param {string} requestKey - Unique key for request cancellation (optional)
     * @returns {Promise} - The API response or error
     */
    async handleAPICall(apiCall, retryCount = 0, requestKey = null) {
        try {
            const response = await apiCall();
            
            // Remove from pending requests if successful
            if (requestKey) {
                this.pendingRequests.delete(requestKey);
            }
            
            if (!response.ok) {
                if (response.status === 403) {
                    throw new Error('PERMISSION_DENIED');
                } else if (response.status === 400) {
                    const data = await response.json();
                    throw new Error(data.error || 'Invalid request');
                } else if (response.status === 500) {
                    throw new Error('SERVER_ERROR');
                } else if (response.status === 0 || response.status >= 500) {
                    // Network error or server error - retry
                    throw new Error('NETWORK_ERROR');
                }
                throw new Error(`Request failed with status ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            // Check if request was aborted
            if (error.name === 'AbortError') {
                console.log('Request aborted:', requestKey);
                throw new Error('REQUEST_CANCELLED');
            }
            
            console.error('API Error:', error);
            
            // Remove from pending requests on error
            if (requestKey) {
                this.pendingRequests.delete(requestKey);
            }
            
            // Check if error is retryable
            if (this.isRetryableError(error) && retryCount < this.maxRetries) {
                console.log(`Retrying... Attempt ${retryCount + 1} of ${this.maxRetries}`);
                
                // Wait before retrying (exponential backoff)
                await this.delay(this.retryDelay * Math.pow(2, retryCount));
                
                // Retry the API call
                return this.handleAPICall(apiCall, retryCount + 1, requestKey);
            }
            
            throw error;
        }
    }

    /**
     * Check if an error is retryable
     * @param {Error} error - The error to check
     * @returns {boolean} - True if error is retryable
     */
    isRetryableError(error) {
        // Retry on network errors and server errors
        return error.message === 'NETWORK_ERROR' || 
               error.message === 'SERVER_ERROR' ||
               error.message === 'Failed to fetch' ||
               error.name === 'TypeError'; // Network errors often throw TypeError
    }

    /**
     * Delay helper for retry logic
     * @param {number} ms - Milliseconds to delay
     * @returns {Promise} - Promise that resolves after delay
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Predict paper for a specific subject and year
     * @param {string} subject - Physics, Chemistry, or Biology
     * @param {number} year - Target year (2026-2028)
     * @param {boolean} useAI - Whether to use AI predictions
     * @param {boolean} useCache - Whether to use cache (default: true)
     * @param {boolean} debounce - Whether to debounce the call (default: false)
     * @returns {Promise<Object>} - Predicted paper data
     */
    async predictPaper(subject, year, useAI = true, useCache = true, debounce = false) {
        const cacheKey = this.cache.generateKey('predict-paper', { subject, year, useAI });
        const requestKey = `predict-paper-${subject}-${year}-${useAI}`;
        
        // Check cache first
        if (useCache) {
            const cached = this.cache.get(cacheKey);
            if (cached) {
                console.log('Using cached prediction for', subject, year);
                return cached;
            }
        }
        
        // Cancel any existing request for the same parameters
        this.cancelRequest(requestKey);
        
        // Create abort controller for this request
        const controller = new AbortController();
        this.pendingRequests.set(requestKey, controller);
        
        // Define the API call
        const makeRequest = async () => {
            const result = await this.handleAPICall(async () => {
                return await fetch(`${this.baseURL}/predict-paper`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    credentials: 'include',
                    signal: controller.signal,
                    body: JSON.stringify({
                        subject,
                        year,
                        use_ai: useAI
                    })
                });
            }, 0, requestKey);
            
            // Cache the result
            if (useCache && result) {
                this.cache.set(cacheKey, result);
            }
            
            return result;
        };
        
        // Debounce if requested
        if (debounce) {
            return this.debounce(requestKey, makeRequest, 500);
        }
        
        return makeRequest();
    }

    /**
     * Get prediction insights for a subject
     * @param {string} subject - Physics, Chemistry, or Biology
     * @param {boolean} useCache - Whether to use cache (default: true)
     * @returns {Promise<Object>} - Insights data
     */
    async getInsights(subject, useCache = true) {
        const cacheKey = this.cache.generateKey('insights', { subject });
        
        // Check cache first
        if (useCache) {
            const cached = this.cache.get(cacheKey);
            if (cached) {
                console.log('Using cached insights for', subject);
                return cached;
            }
        }
        
        // Make API call
        const result = await this.handleAPICall(async () => {
            return await fetch(`${this.baseURL}/insights/${subject}`, {
                method: 'GET',
                credentials: 'include'
            });
        });
        
        // Cache the result
        if (useCache && result) {
            this.cache.set(cacheKey, result);
        }
        
        return result;
    }

    /**
     * Generate smart paper based on user's weak areas
     * @param {string} subject - Subject name
     * @param {Array<string>} focusChapters - Chapters to focus on
     * @param {string} difficultyLevel - easy, medium, hard, or mixed
     * @returns {Promise<Object>} - Smart paper data
     */
    async generateSmartPaper(subject, focusChapters, difficultyLevel) {
        return this.handleAPICall(async () => {
            return await fetch(`${this.baseURL}/smart-paper`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({
                    subject,
                    focus_chapters: focusChapters,
                    difficulty_level: difficultyLevel
                })
            });
        });
    }

    /**
     * Get chapter-wise analysis for a subject
     * @param {string} subject - Physics, Chemistry, or Biology
     * @param {boolean} useCache - Whether to use cache (default: true)
     * @param {boolean} debounce - Whether to debounce the call (default: false)
     * @returns {Promise<Object>} - Chapter analysis data
     */
    async getChapterAnalysis(subject, useCache = true, debounce = false) {
        const cacheKey = this.cache.generateKey('chapter-analysis', { subject });
        const requestKey = `chapter-analysis-${subject}`;
        
        // Check cache first
        if (useCache) {
            const cached = this.cache.get(cacheKey);
            if (cached) {
                console.log('Using cached chapter analysis for', subject);
                return cached;
            }
        }
        
        // Cancel any existing request for the same parameters
        this.cancelRequest(requestKey);
        
        // Create abort controller for this request
        const controller = new AbortController();
        this.pendingRequests.set(requestKey, controller);
        
        // Define the API call
        const makeRequest = async () => {
            const result = await this.handleAPICall(async () => {
                return await fetch(`${this.baseURL}/chapter-analysis/${subject}`, {
                    method: 'GET',
                    credentials: 'include',
                    signal: controller.signal
                });
            }, 0, requestKey);
            
            // Cache the result
            if (useCache && result) {
                this.cache.set(cacheKey, result);
            }
            
            return result;
        };
        
        // Debounce if requested
        if (debounce) {
            return this.debounce(requestKey, makeRequest, 300);
        }
        
        return makeRequest();
    }

    /**
     * Get complete NEET prediction for a year
     * @param {number} year - Target year
     * @param {boolean} useCache - Whether to use cache (default: true)
     * @returns {Promise<Object>} - Complete NEET paper (200 questions)
     */
    async getCompleteNEET(year, useCache = true) {
        const cacheKey = this.cache.generateKey('complete-neet', { year });
        
        // Check cache first
        if (useCache) {
            const cached = this.cache.get(cacheKey);
            if (cached) {
                console.log('Using cached complete NEET for', year);
                return cached;
            }
        }
        
        // Make API call
        const result = await this.handleAPICall(async () => {
            return await fetch(`${this.baseURL}/complete-neet/${year}`, {
                method: 'GET',
                credentials: 'include'
            });
        });
        
        // Cache the result
        if (useCache && result) {
            this.cache.set(cacheKey, result);
        }
        
        return result;
    }

    /**
     * Get full prediction with all subjects
     * @param {number} year - Target year
     * @param {boolean} useCache - Whether to use cache (default: true)
     * @returns {Promise<Object>} - Full prediction data
     */
    async getFullPrediction(year, useCache = true) {
        const cacheKey = this.cache.generateKey('full-prediction', { year });
        
        // Check cache first
        if (useCache) {
            const cached = this.cache.get(cacheKey);
            if (cached) {
                console.log('Using cached full prediction for', year);
                return cached;
            }
        }
        
        // Make API call
        const result = await this.handleAPICall(async () => {
            return await fetch(`${this.baseURL}/full-prediction/${year}`, {
                method: 'GET',
                credentials: 'include'
            });
        });
        
        // Cache the result
        if (useCache && result) {
            this.cache.set(cacheKey, result);
        }
        
        return result;
    }

    /**
     * Clear all cached predictions
     */
    clearCache() {
        this.cache.clearAll();
        console.log('All prediction caches cleared');
    }

    /**
     * Clear expired cache entries
     */
    clearExpiredCache() {
        this.cache.clearExpired();
        console.log('Expired caches cleared');
    }

    /**
     * Get cache statistics
     * @returns {Object} - Cache stats
     */
    getCacheStats() {
        return this.cache.getStats();
    }
}

/**
 * Chart Manager for Predictions Page
 * Handles all chart creation and updates with lazy loading
 */
class ChartManager {
    constructor() {
        this.charts = {};
        this.pendingCharts = new Map(); // Store chart configs for lazy loading
        this.setupIntersectionObserver();
    }

    /**
     * Setup Intersection Observer for lazy loading charts
     */
    setupIntersectionObserver() {
        // Check if IntersectionObserver is supported
        if (!('IntersectionObserver' in window)) {
            console.warn('IntersectionObserver not supported, charts will load immediately');
            this.observerSupported = false;
            return;
        }

        this.observerSupported = true;

        // Create observer with options
        const options = {
            root: null, // viewport
            rootMargin: '50px', // Load 50px before entering viewport
            threshold: 0.1 // Trigger when 10% visible
        };

        this.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const canvasId = entry.target.id;
                    
                    // Check if we have a pending chart for this canvas
                    if (this.pendingCharts.has(canvasId)) {
                        const chartConfig = this.pendingCharts.get(canvasId);
                        
                        // Create the chart
                        this.createChartFromConfig(canvasId, chartConfig);
                        
                        // Remove from pending
                        this.pendingCharts.delete(canvasId);
                        
                        // Stop observing this element
                        this.observer.unobserve(entry.target);
                    }
                }
            });
        }, options);
    }

    /**
     * Register a chart for lazy loading
     * @param {string} canvasId - Canvas element ID
     * @param {Object} chartConfig - Chart configuration
     */
    registerLazyChart(canvasId, chartConfig) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas element ${canvasId} not found`);
            return;
        }

        // If observer not supported, create chart immediately
        if (!this.observerSupported) {
            this.createChartFromConfig(canvasId, chartConfig);
            return;
        }

        // Store config for later
        this.pendingCharts.set(canvasId, chartConfig);

        // Add loading placeholder
        const parent = canvas.parentElement;
        if (parent && !parent.querySelector('.chart-lazy-placeholder')) {
            const placeholder = document.createElement('div');
            placeholder.className = 'chart-lazy-placeholder';
            placeholder.innerHTML = '<div class="chart-lazy-spinner"></div>';
            parent.insertBefore(placeholder, canvas);
        }

        // Start observing
        this.observer.observe(canvas);
    }

    /**
     * Create chart from stored configuration
     * @param {string} canvasId - Canvas element ID
     * @param {Object} chartConfig - Chart configuration
     */
    createChartFromConfig(canvasId, chartConfig) {
        const { type, data, method } = chartConfig;

        // Remove placeholder if exists
        const canvas = document.getElementById(canvasId);
        if (canvas && canvas.parentElement) {
            const placeholder = canvas.parentElement.querySelector('.chart-lazy-placeholder');
            if (placeholder) {
                placeholder.remove();
            }
        }

        // Create chart based on type
        switch (type) {
            case 'chapterProbability':
                return this.createChapterProbabilityChart(canvasId, data);
            case 'difficultyDistribution':
                return this.createDifficultyDistributionChart(canvasId, data);
            case 'accuracyTrend':
                return this.createAccuracyTrendChart(canvasId, data);
            case 'topicFrequency':
                return this.createTopicFrequencyChart(canvasId, data.topics, data.sortOrder);
            default:
                console.error('Unknown chart type:', type);
                return null;
        }
    }

    /**
     * Disconnect observer (cleanup)
     */
    disconnectObserver() {
        if (this.observer) {
            this.observer.disconnect();
        }
    }

    /**
     * Create chapter probability horizontal bar chart
     * @param {string} canvasId - Canvas element ID
     * @param {Array<Object>} chapterData - Array of {name, probability}
     * @returns {Chart} - Chart.js instance
     */
    createChapterProbabilityChart(canvasId, chapterData) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas element ${canvasId} not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Sort by probability (highest first)
        const sortedData = [...chapterData].sort((a, b) => b.probability - a.probability);

        // Prepare data
        const labels = sortedData.map(item => item.name);
        const probabilities = sortedData.map(item => item.probability * 100); // Convert to percentage

        // Color code based on probability
        const backgroundColors = probabilities.map(prob => {
            if (prob >= 75) return ChartConfig.colors.green;
            if (prob >= 50) return ChartConfig.colors.yellow;
            return ChartConfig.colors.red;
        });

        const borderColors = probabilities.map(prob => {
            if (prob >= 75) return ChartConfig.colors.greenLight;
            if (prob >= 50) return ChartConfig.colors.yellowLight;
            return ChartConfig.colors.redLight;
        });

        // Create chart
        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability',
                    data: probabilities,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                ...ChartConfig.defaultOptions,
                indexAxis: 'y', // Horizontal bar chart
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    legend: {
                        display: false
                    },
                    tooltip: {
                        ...ChartConfig.defaultOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                return `Probability: ${context.parsed.x.toFixed(1)}%`;
                            },
                            afterLabel: function(context) {
                                const prob = context.parsed.x;
                                if (prob >= 75) return 'High probability';
                                if (prob >= 50) return 'Medium probability';
                                return 'Low probability';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(59, 130, 246, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: ChartConfig.colors.textSecondary,
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11
                            },
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    y: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: ChartConfig.colors.textSecondary,
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11
                            }
                        }
                    }
                }
            }
        });

        // Register chart for responsive updates
        ChartConfig.registerChart(this.charts[canvasId]);

        return this.charts[canvasId];
    }

    /**
     * Create difficulty distribution pie chart
     * @param {string} canvasId - Canvas element ID
     * @param {Object} difficultyData - Object with {easy, medium, hard} counts or percentages
     * @returns {Chart} - Chart.js instance
     */
    createDifficultyDistributionChart(canvasId, difficultyData) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas element ${canvasId} not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Calculate total and percentages
        const total = difficultyData.easy + difficultyData.medium + difficultyData.hard;
        const percentages = {
            easy: total > 0 ? (difficultyData.easy / total) * 100 : 0,
            medium: total > 0 ? (difficultyData.medium / total) * 100 : 0,
            hard: total > 0 ? (difficultyData.hard / total) * 100 : 0
        };

        // Create chart
        this.charts[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Easy', 'Medium', 'Hard'],
                datasets: [{
                    label: 'Difficulty Distribution',
                    data: [percentages.easy, percentages.medium, percentages.hard],
                    backgroundColor: [
                        ChartConfig.colors.green,
                        ChartConfig.colors.yellow,
                        ChartConfig.colors.red
                    ],
                    borderColor: [
                        ChartConfig.colors.greenLight,
                        ChartConfig.colors.yellowLight,
                        ChartConfig.colors.redLight
                    ],
                    borderWidth: 2,
                    hoverOffset: 10
                }]
            },
            options: {
                ...ChartConfig.defaultOptions,
                cutout: '60%', // Doughnut style
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: ChartConfig.colors.textSecondary,
                            font: {
                                family: 'Inter, sans-serif',
                                size: 13,
                                weight: 500
                            },
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            generateLabels: function(chart) {
                                const data = chart.data;
                                if (data.labels.length && data.datasets.length) {
                                    return data.labels.map((label, i) => {
                                        const value = data.datasets[0].data[i];
                                        return {
                                            text: `${label}: ${value.toFixed(1)}%`,
                                            fillStyle: data.datasets[0].backgroundColor[i],
                                            strokeStyle: data.datasets[0].borderColor[i],
                                            lineWidth: 2,
                                            hidden: false,
                                            index: i
                                        };
                                    });
                                }
                                return [];
                            }
                        }
                    },
                    tooltip: {
                        ...ChartConfig.defaultOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                const count = context.dataset.originalData ? context.dataset.originalData[context.dataIndex] : 0;
                                return `${label}: ${value.toFixed(1)}% (${count} questions)`;
                            }
                        }
                    }
                }
            }
        });

        // Store original counts for tooltip
        this.charts[canvasId].data.datasets[0].originalData = [
            difficultyData.easy,
            difficultyData.medium,
            difficultyData.hard
        ];

        // Register chart for responsive updates
        ChartConfig.registerChart(this.charts[canvasId]);

        return this.charts[canvasId];
    }

    /**
     * Create accuracy trend line chart
     * @param {string} canvasId - Canvas element ID
     * @param {Array<Object>} accuracyData - Array of {year, accuracy} objects
     * @returns {Chart} - Chart.js instance
     */
    createAccuracyTrendChart(canvasId, accuracyData) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas element ${canvasId} not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Sort by year
        const sortedData = [...accuracyData].sort((a, b) => a.year - b.year);

        // Prepare data
        const labels = sortedData.map(item => item.year.toString());
        const accuracies = sortedData.map(item => item.accuracy * 100); // Convert to percentage

        // Create gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.4)');
        gradient.addColorStop(1, 'rgba(139, 92, 246, 0.1)');

        // Create chart
        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Prediction Accuracy',
                    data: accuracies,
                    borderColor: ChartConfig.colors.primary,
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4, // Smooth curves
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointBackgroundColor: ChartConfig.colors.primary,
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointHoverBackgroundColor: ChartConfig.colors.primaryLight,
                    pointHoverBorderColor: '#ffffff',
                    pointHoverBorderWidth: 3
                }]
            },
            options: {
                ...ChartConfig.defaultOptions,
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    legend: {
                        display: false
                    },
                    tooltip: {
                        ...ChartConfig.defaultOptions.plugins.tooltip,
                        callbacks: {
                            title: function(context) {
                                return `Year ${context[0].label}`;
                            },
                            label: function(context) {
                                return `Accuracy: ${context.parsed.y.toFixed(1)}%`;
                            },
                            afterLabel: function(context) {
                                const accuracy = context.parsed.y;
                                if (accuracy >= 80) return '✓ Excellent';
                                if (accuracy >= 70) return '✓ Good';
                                if (accuracy >= 60) return '○ Fair';
                                return '○ Needs improvement';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(59, 130, 246, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: ChartConfig.colors.textSecondary,
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11,
                                weight: 500
                            }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(59, 130, 246, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: ChartConfig.colors.textSecondary,
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11
                            },
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });

        // Register chart for responsive updates
        ChartConfig.registerChart(this.charts[canvasId]);

        return this.charts[canvasId];
    }

    /**
     * Create topic frequency bar chart
     * @param {string} canvasId - Canvas element ID
     * @param {Array<Object>} topicData - Array of {topic, frequency} objects
     * @param {string} sortOrder - 'asc' or 'desc' for sorting
     * @returns {Chart} - Chart.js instance
     */
    createTopicFrequencyChart(canvasId, topicData, sortOrder = 'desc') {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas element ${canvasId} not found`);
            return null;
        }

        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        // Sort data based on sortOrder
        const sortedData = [...topicData].sort((a, b) => {
            return sortOrder === 'desc' ? b.frequency - a.frequency : a.frequency - b.frequency;
        });

        // Prepare data
        const labels = sortedData.map(item => item.topic);
        const frequencies = sortedData.map(item => item.frequency);

        // Create gradient colors based on frequency
        const maxFrequency = Math.max(...frequencies);
        const backgroundColors = frequencies.map(freq => {
            const intensity = freq / maxFrequency;
            if (intensity >= 0.7) return ChartConfig.colors.primary;
            if (intensity >= 0.4) return ChartConfig.colors.purple;
            return ChartConfig.colors.gray;
        });

        const borderColors = frequencies.map(freq => {
            const intensity = freq / maxFrequency;
            if (intensity >= 0.7) return ChartConfig.colors.primaryLight;
            if (intensity >= 0.4) return ChartConfig.colors.purple;
            return ChartConfig.colors.grayLight;
        });

        // Create chart
        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Frequency',
                    data: frequencies,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                ...ChartConfig.defaultOptions,
                plugins: {
                    ...ChartConfig.defaultOptions.plugins,
                    legend: {
                        display: false
                    },
                    tooltip: {
                        ...ChartConfig.defaultOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                const freq = context.parsed.y;
                                return `Appeared ${freq} time${freq !== 1 ? 's' : ''}`;
                            },
                            afterLabel: function(context) {
                                const freq = context.parsed.y;
                                const max = Math.max(...context.chart.data.datasets[0].data);
                                const percentage = (freq / max) * 100;
                                return `${percentage.toFixed(0)}% of max frequency`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: ChartConfig.colors.textSecondary,
                            font: {
                                family: 'Inter, sans-serif',
                                size: 10
                            },
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(59, 130, 246, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: ChartConfig.colors.textSecondary,
                            font: {
                                family: 'Inter, sans-serif',
                                size: 11
                            },
                            stepSize: 1,
                            callback: function(value) {
                                return Number.isInteger(value) ? value : '';
                            }
                        }
                    }
                }
            }
        });

        // Store sort order for future updates
        this.charts[canvasId].sortOrder = sortOrder;

        // Register chart for responsive updates
        ChartConfig.registerChart(this.charts[canvasId]);

        return this.charts[canvasId];
    }

    /**
     * Update topic frequency chart with new sort order
     * @param {string} canvasId - Canvas element ID
     * @param {Array<Object>} topicData - Array of {topic, frequency} objects
     * @param {string} sortOrder - 'asc' or 'desc' for sorting
     */
    updateTopicFrequencyChart(canvasId, topicData, sortOrder) {
        return this.createTopicFrequencyChart(canvasId, topicData, sortOrder);
    }

    /**
     * Destroy a specific chart
     * @param {string} canvasId - Canvas element ID
     */
    destroyChart(canvasId) {
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
            delete this.charts[canvasId];
        }
    }

    /**
     * Destroy all charts
     */
    destroyAllCharts() {
        Object.keys(this.charts).forEach(canvasId => {
            this.destroyChart(canvasId);
        });
    }
}

/**
 * UI Controller for Predictions Page
 */
class PredictionsUI {
    constructor() {
        this.api = new PredictionAPI();
        this.chartManager = new ChartManager();
        this.deferredTasks = [];
        this.init();
    }

    init() {
        console.log('Predictions UI initialized');
        
        // Hide all panels by default (show only dashboard)
        this.hideAllPanels();
        
        // Check user subscription and update UI
        this.checkSubscriptionStatus();
        
        this.setupEventListeners();
        this.loadDashboard();
        
        // Defer non-critical tasks
        this.deferNonCriticalTasks();
        
        // Setup cleanup on page unload
        this.setupCleanup();
    }

    /**
     * Hide all prediction panels on page load
     */
    hideAllPanels() {
        const allPanels = document.querySelectorAll('.prediction-panel');
        allPanels.forEach(panel => {
            panel.style.display = 'none';
        });
        console.log(`Hidden ${allPanels.length} prediction panels`);
    }

    /**
     * Check user subscription status and update UI accordingly
     */
    async checkSubscriptionStatus() {
        try {
            // Check if user is authenticated
            if (!window.authCheck || !window.authCheck.isAuthenticated()) {
                console.log('User not authenticated, showing premium locks');
                return; // Keep premium features locked
            }

            // Get user's subscription tier
            const subscription = await window.authCheck.getCurrentSubscription();
            console.log('User subscription tier:', subscription);

            // If user has premium access, unlock premium features
            if (subscription && (subscription === 'premium' || subscription === 'ultimate')) {
                this.unlockPremiumFeatures();
            } else {
                console.log('User does not have premium access');
            }
        } catch (error) {
            console.error('Error checking subscription:', error);
            // Keep features locked on error
        }
    }

    /**
     * Unlock premium features for premium users
     */
    unlockPremiumFeatures() {
        console.log('Unlocking premium features');

        // Remove premium-locked class from feature cards
        const premiumCards = document.querySelectorAll('.feature-card.premium-locked');
        premiumCards.forEach(card => {
            card.classList.remove('premium-locked');
            
            // Remove lock overlay
            const lockOverlay = card.querySelector('.lock-overlay');
            if (lockOverlay) {
                lockOverlay.remove();
            }

            // Update button
            const button = card.querySelector('.feature-btn.premium-btn');
            if (button) {
                button.disabled = false;
                button.classList.remove('premium-btn');
                
                // Get the action from parent card
                const action = card.getAttribute('data-action');
                
                // Update button text based on feature
                if (action === 'complete-neet') {
                    button.innerHTML = 'Generate Full Paper';
                }
                
                // Add click handler
                button.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.handleQuickAction({ currentTarget: card });
                });
            }

            // Make card clickable
            card.style.cursor = 'pointer';
            card.setAttribute('data-action', card.getAttribute('data-action'));
        });
    }

    /**
     * Setup cleanup handlers for page navigation
     */
    setupCleanup() {
        // Cancel pending requests when navigating away
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });

        // Also handle visibility change (tab switching)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Page is hidden, cancel non-critical requests
                console.log('Page hidden, cancelling pending requests');
                this.api.cancelAllRequests();
            }
        });

        // Handle page navigation (for SPAs)
        window.addEventListener('popstate', () => {
            this.cleanup();
        });
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        console.log('Cleaning up predictions UI');
        
        // Cancel all pending API requests
        this.api.cancelAllRequests();
        
        // Disconnect chart observer
        this.chartManager.disconnectObserver();
        
        // Clear any pending debounced calls
        this.api.debouncedCalls.forEach((call) => {
            clearTimeout(call.timeout);
        });
        this.api.debouncedCalls.clear();
    }

    /**
     * Defer non-critical JavaScript execution
     * Uses requestIdleCallback if available, otherwise setTimeout
     */
    deferNonCriticalTasks() {
        // Clear expired cache entries (non-critical)
        this.scheduleIdleTask(() => {
            this.api.clearExpiredCache();
        });

        // Log cache statistics (non-critical)
        this.scheduleIdleTask(() => {
            const stats = this.api.getCacheStats();
            console.log('Cache statistics:', stats);
        });

        // Setup analytics or other non-critical features
        this.scheduleIdleTask(() => {
            this.setupNonCriticalFeatures();
        });
    }

    /**
     * Schedule a task to run when browser is idle
     * @param {Function} task - Task to execute
     * @param {Object} options - Options for requestIdleCallback
     */
    scheduleIdleTask(task, options = { timeout: 2000 }) {
        if ('requestIdleCallback' in window) {
            requestIdleCallback(task, options);
        } else {
            // Fallback to setTimeout
            setTimeout(task, 100);
        }
    }

    /**
     * Setup non-critical features
     */
    setupNonCriticalFeatures() {
        // Add keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Setup tooltips
        this.setupTooltips();
        
        // Log initialization complete
        console.log('Non-critical features initialized');
    }

    /**
     * Setup keyboard shortcuts (non-critical)
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('input[type="search"]');
                if (searchInput) {
                    searchInput.focus();
                }
            }
        });
    }

    /**
     * Setup tooltips (non-critical)
     */
    setupTooltips() {
        // Add tooltip functionality if needed
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                const tooltip = e.target.dataset.tooltip;
                if (tooltip) {
                    // Show tooltip
                    console.log('Tooltip:', tooltip);
                }
            });
        });
    }

    setupEventListeners() {
        // Subject-wise prediction form
        const subjectForm = document.getElementById('subject-prediction-form');
        if (subjectForm) {
            subjectForm.addEventListener('submit', (e) => this.handleSubjectPrediction(e));
        }

        // Complete NEET prediction form
        const completeNeetForm = document.getElementById('complete-neet-form');
        if (completeNeetForm) {
            completeNeetForm.addEventListener('submit', (e) => this.handleCompleteNeetPrediction(e));
        }

        // Chapter analysis subject tabs
        const subjectTabs = document.querySelectorAll('.subject-tabs .tab-btn');
        subjectTabs.forEach(tab => {
            tab.addEventListener('click', (e) => this.handleSubjectTabClick(e));
        });

        // Chapter analysis sort and filter
        const sortSelect = document.getElementById('sort-by');
        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => this.handleChapterSort(e));
        }

        const filterSelect = document.getElementById('filter-class');
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => this.handleChapterFilter(e));
        }

        // Smart paper generation form
        const smartPaperForm = document.getElementById('smart-paper-form');
        if (smartPaperForm) {
            smartPaperForm.addEventListener('submit', (e) => this.handleSmartPaperGeneration(e));
        }

        // Smart paper subject selector - populate chapters when subject changes
        const smartSubjectSelector = document.getElementById('smart-subject-selector');
        if (smartSubjectSelector) {
            smartSubjectSelector.addEventListener('change', (e) => this.handleSmartSubjectChange(e));
        }

        // Quick action buttons
        const quickButtons = document.querySelectorAll('[data-action]');
        quickButtons.forEach(btn => {
            btn.addEventListener('click', (e) => this.handleQuickAction(e));
        });

        // Premium upgrade buttons
        const premiumButtons = document.querySelectorAll('.feature-btn.premium-btn');
        premiumButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.showUpgradeModal('Premium Features');
            });
        });

        console.log('Event listeners setup complete');
    }

    async loadDashboard() {
        try {
            this.showLoading();
            // Dashboard loading logic will be implemented in subsequent tasks
            this.hideLoading();
        } catch (error) {
            this.showError(error.message);
        }
    }

    showLoading() {
        const loader = document.getElementById('predictions-loading');
        if (loader) {
            loader.classList.remove('hidden');
        }
    }

    hideLoading() {
        const loader = document.getElementById('predictions-loading');
        if (loader) {
            loader.classList.add('hidden');
        }
    }

    showError(message, retryCallback = null) {
        this.hideLoading();
        
        // Check if it's a permission error
        if (message === 'PERMISSION_DENIED') {
            this.showUpgradeModal('AI Predictions');
            return;
        }

        // Check if it's a network error
        if (message === 'NETWORK_ERROR' || message === 'Failed to fetch' || message.includes('Network error')) {
            if (retryCallback) {
                this.showNetworkErrorModal(
                    'Unable to connect to the server. Please check your internet connection and try again.',
                    retryCallback
                );
            } else {
                this.showToast('Network error. Please check your connection.', 'error');
            }
            return;
        }

        // Check if it's a server error
        if (message === 'SERVER_ERROR') {
            this.showToast('Server error. Please try again later.', 'error');
            return;
        }

        // Show generic error toast
        this.showToast(message, 'error');
    }

    /**
     * Handle subject-wise prediction form submission
     * @param {Event} e - Form submit event
     */
    async handleSubjectPrediction(e) {
        e.preventDefault();

        const form = e.target;
        const subjectField = form.querySelector('#subject-selector');
        const yearField = form.querySelector('#year-selector');
        const subject = subjectField?.value;
        const year = parseInt(yearField?.value);
        const generateBtn = form.querySelector('.generate-btn');
        const resultsContainer = document.getElementById('subject-prediction-results');

        // Clear previous errors
        this.clearFieldErrors(form);

        // Validate inputs
        let hasErrors = false;

        if (!subject || subject === '') {
            this.showFieldError('subject-selector', 'Please select a subject');
            hasErrors = true;
        }

        if (!year || isNaN(year)) {
            this.showFieldError('year-selector', 'Please select a year');
            hasErrors = true;
        } else if (year < 2026 || year > 2030) {
            this.showFieldError('year-selector', 'Year must be between 2026 and 2030');
            hasErrors = true;
        }

        // Prevent submission if there are errors
        if (hasErrors) {
            this.showToast('Please fix the form errors before submitting', 'warning');
            return;
        }

        try {
            // Show loading state
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<span class="spinner small"></span> Generating...';
            resultsContainer.innerHTML = '<div class="loading-container"><div class="spinner"></div><p>Generating predicted paper...</p></div>';
            resultsContainer.classList.remove('hidden');

            // Call API
            const paperData = await this.api.predictPaper(subject, year, true);

            // Display results
            this.displaySubjectPrediction(paperData, resultsContainer);

            // Show success toast
            this.showToast(`Successfully generated ${subject} prediction for ${year}`, 'success');

            // Reset button
            generateBtn.disabled = false;
            generateBtn.innerHTML = 'Generate Prediction';

        } catch (error) {
            console.error('Prediction error:', error);
            
            // Reset button
            generateBtn.disabled = false;
            generateBtn.innerHTML = 'Generate Prediction';

            // Handle errors
            if (error.message === 'PERMISSION_DENIED') {
                resultsContainer.innerHTML = this.getPermissionErrorHTML();
                this.showUpgradeModal('Subject Predictions');
            } else if (error.message === 'NETWORK_ERROR' || error.message === 'Failed to fetch') {
                resultsContainer.innerHTML = this.getErrorHTML('Network error. Please check your connection.', () => {
                    form.dispatchEvent(new Event('submit'));
                });
                this.showNetworkErrorModal(
                    'Unable to connect to the server. Please check your internet connection and try again.',
                    () => form.dispatchEvent(new Event('submit'))
                );
            } else {
                resultsContainer.innerHTML = this.getErrorHTML(error.message, () => {
                    form.dispatchEvent(new Event('submit'));
                });
                this.showError(error.message, () => form.dispatchEvent(new Event('submit')));
            }
            resultsContainer.classList.remove('hidden');
        }
    }

    /**
     * Display subject-wise prediction results
     * @param {Object} paperData - Predicted paper data
     * @param {HTMLElement} container - Container element
     */
    displaySubjectPrediction(paperData, container) {
        // Handle both API response format and direct paper format
        const paper = paperData.paper || paperData;
        const paper_info = paper.paper_info || {};
        const questions = paper.questions || [];
        
        // Safely extract values with defaults
        const subject = paper_info.subject || 'Unknown';
        const year = paper_info.year || 2026;
        const questionCount = paper_info.question_count || questions.length || 0;
        const toAttempt = paper_info.to_attempt || 0;
        const duration = paper_info.duration_minutes || 60;
        const totalMarks = paper_info.total_marks || 0;
        const confidence = paper_info.prediction_confidence || 0;
        const basedOnYears = paper_info.based_on_years || 5;
        
        const confidenceBadge = this.getConfidenceBadge(confidence);

        // Build comprehensive HTML structure
        let html = `
            <div class="prediction-results">
                <!-- Header Section -->
                <div class="prediction-header">
                    <div class="header-content">
                        <h3>${subject} - ${year} Predicted Paper</h3>
                        <p class="header-subtitle">AI-Powered NEET Prediction</p>
                    </div>
                    ${confidenceBadge}
                </div>
                
                <!-- Statistics Cards -->
                <div class="prediction-stats">
                    <div class="stat-card">
                        <div class="stat-icon">📝</div>
                        <div class="stat-content">
                            <span class="stat-label">Total Questions</span>
                            <span class="stat-value">${questionCount}</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">✅</div>
                        <div class="stat-content">
                            <span class="stat-label">To Attempt</span>
                            <span class="stat-value">${toAttempt}</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">⏱️</div>
                        <div class="stat-content">
                            <span class="stat-label">Duration</span>
                            <span class="stat-value">${duration} min</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">🎯</div>
                        <div class="stat-content">
                            <span class="stat-label">Total Marks</span>
                            <span class="stat-value">${totalMarks}</span>
                        </div>
                    </div>
                </div>

                <!-- Prediction Information -->
                <div class="prediction-info-section">
                    <div class="info-card">
                        <h4>📊 Prediction Details</h4>
                        <div class="info-grid">
                            <div class="info-item">
                                <span class="info-label">Analysis Period:</span>
                                <span class="info-value">${basedOnYears} years of NEET data</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Confidence Level:</span>
                                <span class="info-value">${Math.round(confidence * 100)}%</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Exam Type:</span>
                                <span class="info-value">${paper_info.exam_type || 'NEET_PREDICTED'}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Questions Generated:</span>
                                <span class="info-value">${questions.length} questions</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- High-Probability Topics -->
                <div class="topics-section">
                    <h4>🎯 High-Probability Topics</h4>
                    <p class="section-subtitle">Focus on these chapters for maximum impact</p>
                    <div class="topics-list">
                        ${this.getHighProbabilityTopics(questions)}
                    </div>
                </div>

                <!-- Difficulty Distribution -->
                <div class="difficulty-section">
                    <h4>📈 Difficulty Distribution</h4>
                    <div class="difficulty-breakdown">
                        ${this.getDifficultyBreakdown(questions)}
                    </div>
                </div>

                <!-- Questions Preview -->
                <div class="questions-section">
                    <h4>📋 Questions Preview</h4>
                    <p class="section-subtitle">Showing first 5 questions</p>
                    <div class="questions-preview-list">
                        ${this.getQuestionsPreview(questions, 5)}
                    </div>
                    ${questions.length > 5 ? `<p class="more-questions">+ ${questions.length - 5} more questions</p>` : ''}
                </div>

                <!-- Action Buttons -->
                <div class="prediction-actions">
                    <button class="btn-primary" onclick="window.predictionsUI.viewFullPaper('${subject}', ${year})">
                        <span class="btn-icon">📄</span>
                        View Full Paper (${questionCount} Questions)
                    </button>
                    <button class="btn-secondary" onclick="window.predictionsUI.downloadPaper('${subject}', ${year})">
                        <span class="btn-icon">⬇️</span>
                        Download PDF
                    </button>
                    <button class="btn-secondary" onclick="window.predictionsUI.startPracticeTest('${subject}', ${year})">
                        <span class="btn-icon">🎯</span>
                        Start Practice Test
                    </button>
                </div>
            </div>
        `;

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    /**
     * Get high-probability topics from questions
     * @param {Array} questions - Array of question objects
     * @returns {string} - HTML for topics list
     */
    getHighProbabilityTopics(questions) {
        if (!questions || questions.length === 0) {
            return '<p class="no-data">No topic data available</p>';
        }

        // Group by chapter and calculate average probability
        const chapterMap = {};
        questions.forEach(q => {
            const chapter = q.chapter || q.topic || 'Unknown';
            const score = q.prediction_score || 0.5;
            
            if (!chapterMap[chapter]) {
                chapterMap[chapter] = {
                    count: 0,
                    totalScore: 0
                };
            }
            chapterMap[chapter].count++;
            chapterMap[chapter].totalScore += score;
        });

        // Calculate averages and sort
        const chapters = Object.keys(chapterMap).map(chapter => ({
            name: chapter,
            avgScore: chapterMap[chapter].totalScore / chapterMap[chapter].count,
            count: chapterMap[chapter].count
        })).sort((a, b) => b.avgScore - a.avgScore).slice(0, 8);

        if (chapters.length === 0) {
            return '<p class="no-data">No topic data available</p>';
        }

        // Generate HTML
        return chapters.map((ch, index) => {
            const percentage = Math.round(ch.avgScore * 100);
            const level = ch.avgScore >= 0.75 ? 'high' : ch.avgScore >= 0.5 ? 'medium' : 'low';
            const barWidth = percentage;
            
            return `
                <div class="topic-item">
                    <div class="topic-header">
                        <span class="topic-rank">#${index + 1}</span>
                        <span class="topic-name">${ch.name}</span>
                        <span class="topic-count">${ch.count} Q</span>
                    </div>
                    <div class="topic-probability">
                        <div class="probability-bar">
                            <div class="probability-fill ${level}" style="width: ${barWidth}%"></div>
                        </div>
                        <span class="probability-value">${percentage}%</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    /**
     * Get difficulty breakdown from questions
     * @param {Array} questions - Array of question objects
     * @returns {string} - HTML for difficulty breakdown
     */
    getDifficultyBreakdown(questions) {
        if (!questions || questions.length === 0) {
            return '<p class="no-data">No difficulty data available</p>';
        }

        // Count questions by difficulty
        const difficultyCount = {
            easy: 0,
            medium: 0,
            hard: 0
        };

        questions.forEach(q => {
            const difficulty = (q.difficulty || 'medium').toLowerCase();
            if (difficultyCount.hasOwnProperty(difficulty)) {
                difficultyCount[difficulty]++;
            }
        });

        const total = questions.length;
        
        return `
            <div class="difficulty-item">
                <div class="difficulty-header">
                    <span class="difficulty-label easy">🟢 Easy</span>
                    <span class="difficulty-count">${difficultyCount.easy} questions</span>
                </div>
                <div class="difficulty-bar">
                    <div class="difficulty-fill easy" style="width: ${(difficultyCount.easy / total * 100)}%"></div>
                </div>
                <span class="difficulty-percentage">${Math.round(difficultyCount.easy / total * 100)}%</span>
            </div>
            <div class="difficulty-item">
                <div class="difficulty-header">
                    <span class="difficulty-label medium">🟡 Medium</span>
                    <span class="difficulty-count">${difficultyCount.medium} questions</span>
                </div>
                <div class="difficulty-bar">
                    <div class="difficulty-fill medium" style="width: ${(difficultyCount.medium / total * 100)}%"></div>
                </div>
                <span class="difficulty-percentage">${Math.round(difficultyCount.medium / total * 100)}%</span>
            </div>
            <div class="difficulty-item">
                <div class="difficulty-header">
                    <span class="difficulty-label hard">🔴 Hard</span>
                    <span class="difficulty-count">${difficultyCount.hard} questions</span>
                </div>
                <div class="difficulty-bar">
                    <div class="difficulty-fill hard" style="width: ${(difficultyCount.hard / total * 100)}%"></div>
                </div>
                <span class="difficulty-percentage">${Math.round(difficultyCount.hard / total * 100)}%</span>
            </div>
        `;
    }

    /**
     * Get questions preview
     * @param {Array} questions - Array of question objects
     * @param {number} limit - Number of questions to show
     * @returns {string} - HTML for questions preview
     */
    getQuestionsPreview(questions, limit = 5) {
        if (!questions || questions.length === 0) {
            return '<p class="no-data">No questions available</p>';
        }

        const previewQuestions = questions.slice(0, limit);
        
        return previewQuestions.map(q => {
            const questionNum = q.question_number || '?';
            const questionText = q.question_text || 'Question text not available';
            const chapter = q.chapter || q.topic || 'Unknown';
            const difficulty = q.difficulty || 'medium';
            const marks = q.marks || 4;
            const score = q.prediction_score || 0.5;
            const scorePercent = Math.round(score * 100);
            
            // Truncate long question text
            const displayText = questionText.length > 150 
                ? questionText.substring(0, 150) + '...' 
                : questionText;
            
            return `
                <div class="question-preview-item">
                    <div class="question-preview-header">
                        <span class="question-number">Q${questionNum}</span>
                        <span class="question-chapter">${chapter}</span>
                        <span class="question-difficulty ${difficulty}">${difficulty}</span>
                        <span class="question-marks">${marks} marks</span>
                    </div>
                    <div class="question-preview-text">${displayText}</div>
                    <div class="question-preview-footer">
                        <span class="prediction-score">Prediction Score: ${scorePercent}%</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    /**
     * Handle quick action button clicks
     * @param {Event} e - Click event
     */
    handleQuickAction(e) {
        const target = e.currentTarget;
        const action = target.dataset.action;
        
        // Check if this is a premium locked feature
        if (target.classList.contains('premium-locked')) {
            // Show upgrade modal
            this.showUpgradeModal('Premium Features');
            return;
        }
        
        switch (action) {
            case 'subject-prediction':
                this.showPanel('subject-prediction-panel');
                break;
            case 'chapter-analysis':
                this.showPanel('chapter-analysis-panel');
                // Load default subject (Physics) if not already loaded
                if (!this.currentChapterData) {
                    this.loadChapterAnalysis('Physics');
                }
                break;
            case 'complete-neet':
                this.showPanel('complete-neet-panel');
                break;
            case 'prediction-insights':
                this.showPanel('insights-panel');
                // Load real insights data from API
                this.loadPredictionInsights('Physics');
                break;
            case 'smart-paper':
                this.showPanel('smart-paper-panel');
                break;
            case 'latest-prediction':
                this.loadLatestPrediction();
                break;
            case 'high-probability':
                this.showHighProbabilityTopics();
                break;
            case 'accuracy-report':
                this.showAccuracyReport();
                break;
            default:
                console.log('Unknown action:', action);
        }
    }

    /**
     * Show a specific panel and hide all others
     * @param {string} panelId - Panel element ID to show
     */
    showPanel(panelId) {
        // Hide all panels first
        const allPanels = document.querySelectorAll('.prediction-panel');
        allPanels.forEach(panel => {
            panel.style.display = 'none';
        });

        // Show the requested panel
        const targetPanel = document.getElementById(panelId);
        if (targetPanel) {
            targetPanel.style.display = 'block';
            
            // Scroll to panel smoothly
            targetPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Add a highlight effect
            targetPanel.style.animation = 'highlight-pulse 1s ease-in-out';
            setTimeout(() => {
                targetPanel.style.animation = '';
            }, 1000);
        } else {
            console.error(`Panel ${panelId} not found`);
        }
    }

    /**
     * Scroll to a specific panel
     * @param {string} panelId - Panel element ID
     */
    scrollToPanel(panelId) {
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Add a highlight effect
            panel.style.animation = 'highlight-pulse 1s ease-in-out';
            setTimeout(() => {
                panel.style.animation = '';
            }, 1000);
        }
    }

    /**
     * Show field validation error
     * @param {string} fieldId - Field element ID
     * @param {string} message - Error message
     */
    showFieldError(fieldId, message) {
        const field = document.getElementById(fieldId);
        if (!field) return;

        // Remove existing error
        const existingError = field.parentElement.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }

        // Add error styling
        field.classList.add('error');

        // Add error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.textContent = message;
        field.parentElement.appendChild(errorDiv);

        // Focus the field
        field.focus();

        // Don't auto-remove - let user fix the error
    }

    /**
     * Clear all field errors in a form
     * @param {HTMLFormElement} form - Form element
     */
    clearFieldErrors(form) {
        if (!form) return;

        // Remove error styling from all fields
        const errorFields = form.querySelectorAll('.error');
        errorFields.forEach(field => {
            field.classList.remove('error');
        });

        // Remove all error messages
        const errorMessages = form.querySelectorAll('.field-error');
        errorMessages.forEach(msg => {
            msg.remove();
        });
    }

    /**
     * Validate form fields
     * @param {Object} fields - Object with field IDs and validation rules
     * @returns {boolean} - True if all fields are valid
     */
    validateFields(fields) {
        let isValid = true;

        for (const [fieldId, rules] of Object.entries(fields)) {
            const field = document.getElementById(fieldId);
            if (!field) continue;

            const value = field.value;

            // Required validation
            if (rules.required && (!value || value === '')) {
                this.showFieldError(fieldId, rules.requiredMessage || 'This field is required');
                isValid = false;
                continue;
            }

            // Min/Max validation for numbers
            if (rules.min !== undefined && parseFloat(value) < rules.min) {
                this.showFieldError(fieldId, rules.minMessage || `Value must be at least ${rules.min}`);
                isValid = false;
                continue;
            }

            if (rules.max !== undefined && parseFloat(value) > rules.max) {
                this.showFieldError(fieldId, rules.maxMessage || `Value must be at most ${rules.max}`);
                isValid = false;
                continue;
            }

            // Pattern validation
            if (rules.pattern && !rules.pattern.test(value)) {
                this.showFieldError(fieldId, rules.patternMessage || 'Invalid format');
                isValid = false;
                continue;
            }

            // Custom validation
            if (rules.custom && !rules.custom(value)) {
                this.showFieldError(fieldId, rules.customMessage || 'Invalid value');
                isValid = false;
                continue;
            }
        }

        return isValid;
    }

    /**
     * Get permission error HTML
     * @returns {string} - HTML for permission error
     */
    getPermissionErrorHTML() {
        return `
            <div class="error-message">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                </svg>
                <div class="error-message-content">
                    <div class="error-message-title">Premium Feature</div>
                    <div class="error-message-text">
                        This feature requires a premium subscription. Upgrade your plan to access AI-powered predictions.
                    </div>
                    <div class="error-message-actions">
                        <button class="btn-primary" onclick="window.location.href='/pricing'">
                            Upgrade Now
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Get error HTML with retry option
     * @param {string} message - Error message
     * @param {Function} retryCallback - Retry function
     * @returns {string} - HTML for error
     */
    getErrorHTML(message, retryCallback) {
        const retryId = 'retry-' + Date.now();
        
        // Store callback for later use
        if (!window.retryCallbacks) {
            window.retryCallbacks = {};
        }
        window.retryCallbacks[retryId] = retryCallback;

        return `
            <div class="error-message">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <div class="error-message-content">
                    <div class="error-message-title">Error</div>
                    <div class="error-message-text">${message}</div>
                    <div class="error-message-actions">
                        <button class="error-retry-btn" onclick="window.retryCallbacks['${retryId}']()">
                            Retry
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Start predicted test (placeholder)
     * @param {string} subject - Subject name
     * @param {number} year - Year
     */
    startPredictedTest(subject, year) {
        // This will integrate with existing question paper functionality
        window.location.href = `/question-paper?type=predicted&subject=${subject}&year=${year}`;
    }

    /**
     * View full paper (placeholder)
     * @param {string} subject - Subject name
     * @param {number} year - Year
     */
    viewFullPaper(subject, year) {
        alert(`View full paper for ${subject} ${year} - Feature coming soon!`);
    }

    /**
     * Handle subject tab click for chapter analysis
     * @param {Event} e - Click event
     */
    async handleSubjectTabClick(e) {
        const tab = e.currentTarget;
        const subject = tab.dataset.subject;

        // Update active tab
        document.querySelectorAll('.subject-tabs .tab-btn').forEach(t => {
            t.classList.remove('active');
        });
        tab.classList.add('active');

        // Load chapter analysis for selected subject
        await this.loadChapterAnalysis(subject);
    }

    /**
     * Load chapter analysis for a subject
     * @param {string} subject - Subject name (Physics, Chemistry, Biology)
     */
    async loadChapterAnalysis(subject) {
        const container = document.getElementById('chapter-list-container');
        if (!container) return;

        try {
            // Show skeleton loader
            this.showSkeletonLoader(container, 'list');

            // Call API
            const analysisData = await this.api.getChapterAnalysis(subject);

            // Store current data for sorting/filtering
            this.currentChapterData = {
                subject: subject,
                chapters: analysisData.analysis.chapters,
                totalAnalyzed: analysisData.analysis.total_analyzed
            };

            // Display chapters
            this.displayChapterAnalysis(analysisData, container);

            // Show success toast
            this.showToast(`Loaded ${subject} chapter analysis`, 'success', 3000);

        } catch (error) {
            console.error('Chapter analysis error:', error);
            
            if (error.message === 'PERMISSION_DENIED') {
                container.innerHTML = this.getPermissionErrorHTML();
                this.showUpgradeModal('Chapter Analysis');
            } else if (error.message === 'NETWORK_ERROR' || error.message === 'Failed to fetch') {
                container.innerHTML = this.getErrorHTML('Network error. Please check your connection.', () => {
                    this.loadChapterAnalysis(subject);
                });
                this.showNetworkErrorModal(
                    'Unable to load chapter analysis. Please check your internet connection and try again.',
                    () => this.loadChapterAnalysis(subject)
                );
            } else {
                container.innerHTML = this.getErrorHTML(error.message, () => {
                    this.loadChapterAnalysis(subject);
                });
                this.showError(error.message, () => this.loadChapterAnalysis(subject));
            }
        }
    }

    /**
     * Display chapter analysis with current sort/filter settings
     */
    displayChapterAnalysis() {
        const container = document.getElementById('chapter-list-container');
        if (!container || !this.currentChapterData) return;

        const { subject, chapters, totalAnalyzed } = this.currentChapterData;

        // Apply filters
        let filteredChapters = [...chapters];
        const filterClass = document.getElementById('filter-class')?.value;
        if (filterClass && filterClass !== 'all') {
            // Filter by class (this would need class info in the data)
            // For now, we'll skip this filter
        }

        // Apply sorting
        const sortBy = document.getElementById('sort-by')?.value || 'probability-desc';
        filteredChapters = this.sortChapters(filteredChapters, sortBy);

        // Build HTML
        let html = `
            <div class="chapter-analysis-header">
                <h4>${subject} Chapter Analysis</h4>
                <p class="analysis-note">Based on ${totalAnalyzed} past papers</p>
            </div>
            <div class="chapter-list">
                ${filteredChapters.map(chapter => this.getChapterItemHTML(chapter)).join('')}
            </div>
        `;

        container.innerHTML = html;
    }

    /**
     * Sort chapters based on selected criteria
     * @param {Array} chapters - Array of chapter objects
     * @param {string} sortBy - Sort criteria
     * @returns {Array} - Sorted chapters
     */
    sortChapters(chapters, sortBy) {
        const sorted = [...chapters];

        switch (sortBy) {
            case 'probability-desc':
                return sorted.sort((a, b) => b.probability - a.probability);
            case 'probability-asc':
                return sorted.sort((a, b) => a.probability - b.probability);
            case 'name-asc':
                return sorted.sort((a, b) => a.name.localeCompare(b.name));
            case 'name-desc':
                return sorted.sort((a, b) => b.name.localeCompare(a.name));
            default:
                return sorted;
        }
    }

    /**
     * Get HTML for a single chapter item
     * @param {Object} chapter - Chapter object with name, probability, frequency, recommended
     * @returns {string} - HTML for chapter item
     */
    getChapterItemHTML(chapter) {
        const percentage = Math.round(chapter.probability * 100);
        const level = chapter.probability >= 0.75 ? 'high' : chapter.probability >= 0.5 ? 'medium' : 'low';
        const recommendedBadge = chapter.recommended ? '<span class="recommended-badge">⭐ Recommended</span>' : '';

        return `
            <div class="chapter-item">
                <div class="chapter-info">
                    <div class="chapter-name">${chapter.name}</div>
                    <div class="chapter-meta">
                        <span class="chapter-frequency">Appeared ${chapter.frequency} times</span>
                        ${recommendedBadge}
                    </div>
                </div>
                <div class="chapter-probability">
                    <div class="probability-bar-wrapper">
                        <div class="probability-bar">
                            <div class="probability-fill ${level}" style="width: ${percentage}%"></div>
                        </div>
                        <span class="probability-value">${percentage}%</span>
                    </div>
                    <span class="confidence-badge ${level}">${this.getProbabilityLabel(chapter.probability)}</span>
                </div>
            </div>
        `;
    }

    /**
     * Get probability label based on score
     * @param {number} probability - Probability score (0-1)
     * @returns {string} - Label text
     */
    getProbabilityLabel(probability) {
        if (probability >= 0.75) return 'High';
        if (probability >= 0.5) return 'Medium';
        return 'Low';
    }

    /**
     * Handle chapter sort change
     * @param {Event} e - Change event
     */
    handleChapterSort(e) {
        if (this.currentChapterData) {
            this.displayChapterAnalysis();
        }
    }

    /**
     * Handle chapter filter change
     * @param {Event} e - Change event
     */
    handleChapterFilter(e) {
        if (this.currentChapterData) {
            this.displayChapterAnalysis();
        }
    }

    /**
     * Handle complete NEET prediction form submission
     * @param {Event} e - Form submit event
     */
    async handleCompleteNeetPrediction(e) {
        e.preventDefault();

        const form = e.target;
        const yearField = form.querySelector('#complete-year-selector');
        const year = parseInt(yearField?.value);
        const generateBtn = form.querySelector('.generate-btn');
        const resultsContainer = document.getElementById('complete-neet-results');

        // Clear previous errors
        this.clearFieldErrors(form);

        // Validate year
        let hasErrors = false;

        if (!year || isNaN(year)) {
            this.showFieldError('complete-year-selector', 'Please select a year');
            hasErrors = true;
        } else if (year < 2026 || year > 2030) {
            this.showFieldError('complete-year-selector', 'Year must be between 2026 and 2030');
            hasErrors = true;
        }

        // Prevent submission if there are errors
        if (hasErrors) {
            this.showToast('Please fix the form errors before submitting', 'warning');
            return;
        }

        try {
            // Show loading state
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<span class="spinner small"></span> Generating...';
            const questionCount = year >= 2026 ? 180 : 200;
            resultsContainer.innerHTML = `<div class="loading-container"><div class="spinner"></div><p>Generating complete NEET paper (${questionCount} questions)...</p></div>`;
            resultsContainer.classList.remove('hidden');

            // Call API
            const paperData = await this.api.getCompleteNEET(year);

            // Display results
            this.displayCompleteNeetPrediction(paperData, resultsContainer);

            // Show success toast
            this.showToast(`Successfully generated complete NEET ${year} paper`, 'success');

            // Reset button
            generateBtn.disabled = false;
            generateBtn.innerHTML = 'Generate Complete Paper';

        } catch (error) {
            console.error('Complete NEET prediction error:', error);
            
            // Reset button
            generateBtn.disabled = false;
            generateBtn.innerHTML = 'Generate Complete Paper';

            // Handle errors
            if (error.message === 'PERMISSION_DENIED') {
                resultsContainer.innerHTML = this.getPermissionErrorHTML();
                this.showUpgradeModal('Complete NEET Predictions');
            } else if (error.message === 'NETWORK_ERROR' || error.message === 'Failed to fetch') {
                resultsContainer.innerHTML = this.getErrorHTML('Network error. Please check your connection.', () => {
                    form.dispatchEvent(new Event('submit'));
                });
                this.showNetworkErrorModal(
                    'Unable to connect to the server. Please check your internet connection and try again.',
                    () => form.dispatchEvent(new Event('submit'))
                );
            } else {
                resultsContainer.innerHTML = this.getErrorHTML(error.message, () => {
                    form.dispatchEvent(new Event('submit'));
                });
                this.showError(error.message, () => form.dispatchEvent(new Event('submit')));
            }
            resultsContainer.classList.remove('hidden');
        }
    }

    /**
     * Display complete NEET prediction results
     * @param {Object} paperData - Complete NEET paper data
     * @param {HTMLElement} container - Container element
     */
    displayCompleteNeetPrediction(paperData, container) {
        // Handle both API response format and direct paper format
        const paper = paperData.paper || paperData;
        const paper_info = paper.paper_info || {};
        const questions = paper.questions || [];
        
        const year = paper_info.year || 2026;
        // Use values from paper_info (backend determines pattern based on year)
        const totalQuestions = paper_info.total_questions || (year >= 2026 ? 180 : 200);
        const toAttempt = paper_info.to_attempt || (year >= 2026 ? 180 : 180);
        const duration = paper_info.duration_minutes || (year >= 2026 ? 180 : 200);
        const totalMarks = paper_info.total_marks || 720;
        const overallConfidence = paper_info.prediction_confidence || 0;
        const basedOnYears = paper_info.based_on_years || 5;
        const subjects = paper_info.subjects || {};

        // Build comprehensive HTML with proper spacing
        let html = `
            <div class="complete-neet-results">
                <!-- Header Section -->
                <div class="complete-neet-header">
                    <div class="header-content">
                        <h2 class="paper-title">Complete NEET ${year} Predicted Paper</h2>
                        <p class="paper-subtitle">AI-Generated Full 200-Question Paper</p>
                    </div>
                    <div class="overall-confidence-badge">
                        <div class="confidence-circle-large">
                            <span class="confidence-percentage">${Math.round(overallConfidence * 100)}%</span>
                            <span class="confidence-text">Confidence</span>
                        </div>
                    </div>
                </div>

                <!-- Paper Overview Stats -->
                <div class="paper-overview-section">
                    <h3 class="section-title">📊 Paper Overview</h3>
                    <div class="overview-stats-grid">
                        <div class="overview-stat-card">
                            <div class="stat-icon-large">📝</div>
                            <div class="stat-details">
                                <span class="stat-value-large">${totalQuestions}</span>
                                <span class="stat-label-text">Total Questions</span>
                            </div>
                        </div>
                        <div class="overview-stat-card">
                            <div class="stat-icon-large">✅</div>
                            <div class="stat-details">
                                <span class="stat-value-large">${toAttempt}</span>
                                <span class="stat-label-text">To Attempt</span>
                            </div>
                        </div>
                        <div class="overview-stat-card">
                            <div class="stat-icon-large">⏱️</div>
                            <div class="stat-details">
                                <span class="stat-value-large">${duration}</span>
                                <span class="stat-label-text">Minutes</span>
                            </div>
                        </div>
                        <div class="overview-stat-card">
                            <div class="stat-icon-large">🎯</div>
                            <div class="stat-details">
                                <span class="stat-value-large">${totalMarks}</span>
                                <span class="stat-label-text">Total Marks</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Subject-wise Breakdown -->
                <div class="subject-breakdown-section">
                    <h3 class="section-title">📚 Subject-wise Breakdown</h3>
                    <div class="subjects-grid">
                        ${this.getSubjectBreakdownCard('Physics', subjects.Physics || {}, '⚛️')}
                        ${this.getSubjectBreakdownCard('Chemistry', subjects.Chemistry || {}, '🧪')}
                        ${this.getSubjectBreakdownCard('Biology', subjects.Biology || {}, '🧬')}
                    </div>
                </div>

                <!-- Paper Information -->
                <div class="paper-info-section">
                    <h3 class="section-title">ℹ️ Paper Information</h3>
                    <div class="info-cards-grid">
                        <div class="info-card">
                            <div class="info-icon">📖</div>
                            <div class="info-content">
                                <span class="info-label">Analysis Based On</span>
                                <span class="info-value">${basedOnYears} years of NEET data</span>
                            </div>
                        </div>
                        <div class="info-card">
                            <div class="info-icon">📋</div>
                            <div class="info-content">
                                <span class="info-label">Exam Pattern</span>
                                <span class="info-value">${toAttempt} out of ${totalQuestions} questions</span>
                            </div>
                        </div>
                        <div class="info-card">
                            <div class="info-icon">✏️</div>
                            <div class="info-content">
                                <span class="info-label">Marking Scheme</span>
                                <span class="info-value">+4 correct, -1 incorrect</span>
                            </div>
                        </div>
                        <div class="info-card">
                            <div class="info-icon">🎓</div>
                            <div class="info-content">
                                <span class="info-label">Exam Type</span>
                                <span class="info-value">NEET UG ${year}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="complete-neet-actions">
                    <button class="btn-primary btn-large" onclick="window.predictionsUI.startCompleteTest(${year})">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                        </svg>
                        <span>Start Complete Test</span>
                    </button>
                    <button class="btn-secondary btn-large" onclick="window.predictionsUI.viewAllQuestions(${year})">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                        <span>View All Questions</span>
                    </button>
                    <button class="btn-secondary btn-large" onclick="window.predictionsUI.downloadCompletePaper(${year})">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        <span>Download PDF</span>
                    </button>
                </div>
            </div>
        `;

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    /**
     * Get subject breakdown card HTML
     * @param {string} subjectName - Subject name
     * @param {Object} subjectData - Subject data
     * @param {string} icon - Subject icon
     * @returns {string} - HTML for subject card
     */
    getSubjectBreakdownCard(subjectName, subjectData, icon) {
        const questions = subjectData.questions || '50';
        const toAttempt = subjectData.to_attempt || '45';
        const confidence = subjectData.confidence || 0;
        const confidencePercent = Math.round(confidence * 100);
        
        // Determine confidence level
        let confidenceClass = 'low';
        if (confidence >= 0.8) confidenceClass = 'high';
        else if (confidence >= 0.6) confidenceClass = 'medium';
        
        return `
            <div class="subject-breakdown-card">
                <div class="subject-card-header">
                    <div class="subject-icon">${icon}</div>
                    <h4 class="subject-name">${subjectName}</h4>
                </div>
                <div class="subject-card-body">
                    <div class="subject-stat-row">
                        <span class="subject-stat-label">Questions:</span>
                        <span class="subject-stat-value">${questions}</span>
                    </div>
                    <div class="subject-stat-row">
                        <span class="subject-stat-label">To Attempt:</span>
                        <span class="subject-stat-value">${toAttempt}</span>
                    </div>
                    <div class="subject-confidence-row">
                        <span class="subject-stat-label">Confidence:</span>
                        <div class="confidence-bar-wrapper">
                            <div class="confidence-bar-track">
                                <div class="confidence-bar-fill ${confidenceClass}" style="width: ${confidencePercent}%"></div>
                            </div>
                            <span class="confidence-percent-text">${confidencePercent}%</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Start complete NEET test - Show test interface
     * @param {number} year - Year
     */
    startCompleteTest(year) {
        console.log(`Starting complete NEET test for ${year}`);
        // Redirect to exam interface
        window.location.href = `/exam/neet/${year}`;
    }

    /**
     * View all questions in the complete paper
     * @param {number} year - Year
     */
    async viewAllQuestions(year) {
        console.log(`Viewing all questions for ${year}`);
        
        try {
            // Show loading
            const totalQuestions = paperData.paper_info?.total_questions || 180;
            this.showToast(`Loading all ${totalQuestions} questions...`, 'info');
            
            // Get the complete paper data
            const paperData = await this.api.getCompleteNEET(year, true);
            const paper = paperData.paper || paperData;
            const questions = paper.questions || [];
            
            if (questions.length === 0) {
                this.showToast('No questions available', 'error');
                return;
            }
            
            // Create modal to show all questions
            this.showQuestionsModal(questions, `Complete NEET ${year} - All Questions`);
            
        } catch (error) {
            console.error('Error loading questions:', error);
            this.showToast('Failed to load questions', 'error');
        }
    }

    /**
     * Download complete paper as PDF
     * @param {number} year - Year
     */
    async downloadCompletePaper(year) {
        console.log(`Downloading complete paper for ${year}`);
        
        try {
            // Show loading
            this.showToast('Preparing PDF download...', 'info');
            
            // Get the complete paper data
            const paperData = await this.api.getCompleteNEET(year, true);
            const paper = paperData.paper || paperData;
            
            // Generate and download PDF
            this.generatePaperPDF(paper, `NEET_${year}_Complete_Paper.pdf`);
            
        } catch (error) {
            console.error('Error downloading paper:', error);
            this.showToast('Failed to download paper', 'error');
        }
    }

    /**
     * Show questions in a modal
     * @param {Array} questions - Array of questions
     * @param {string} title - Modal title
     */
    showQuestionsModal(questions, title) {
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'questions-modal-overlay';
        modal.innerHTML = `
            <div class="questions-modal">
                <div class="questions-modal-header">
                    <h2>${title}</h2>
                    <button class="modal-close-btn" onclick="this.closest('.questions-modal-overlay').remove()">×</button>
                </div>
                <div class="questions-modal-body">
                    ${questions.map((q, index) => `
                        <div class="question-item" id="question-${index + 1}">
                            <div class="question-number-badge">Question ${index + 1}</div>
                            <div class="question-meta">
                                <span class="question-chapter">${q.chapter || 'General'}</span>
                                <span class="question-difficulty ${q.difficulty}">${q.difficulty || 'medium'}</span>
                                <span class="question-marks">${q.marks || 4} marks</span>
                            </div>
                            <div class="question-text">${q.question_text || 'Question text not available'}</div>
                            ${q.options ? `
                                <div class="question-options">
                                    ${q.options.map((opt, i) => `
                                        <div class="option-item">
                                            <span class="option-label">${String.fromCharCode(65 + i)})</span>
                                            <span class="option-text">${opt}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            ` : ''}
                            ${q.solution ? `
                                <details class="question-solution">
                                    <summary>View Solution</summary>
                                    <div class="solution-content">${q.solution}</div>
                                </details>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
                <div class="questions-modal-footer">
                    <button class="btn-secondary" onclick="this.closest('.questions-modal-overlay').remove()">Close</button>
                    <button class="btn-primary" onclick="window.predictionsUI.downloadCompletePaper(${title.match(/\d{4}/)?.[0] || 2026})">
                        Download PDF
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        this.showToast(`Loaded ${questions.length} questions`, 'success');
    }

    /**
     * Generate and download paper as PDF
     * @param {Object} paper - Paper data
     * @param {string} filename - PDF filename
     */
    generatePaperPDF(paper, filename) {
        // For now, create a printable HTML version
        const paper_info = paper.paper_info || {};
        const questions = paper.questions || [];
        
        const printWindow = window.open('', '_blank');
        
        // Check if popup was blocked
        if (!printWindow) {
            this.showToast('Please allow popups to download the paper', 'error');
            return;
        }
        
        printWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>${filename}</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; max-width: 800px; margin: 0 auto; }
                    h1 { text-align: center; color: #1e40af; }
                    .paper-info { background: #f3f4f6; padding: 15px; margin: 20px 0; border-radius: 8px; }
                    .question { margin: 20px 0; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; page-break-inside: avoid; }
                    .question-number { font-weight: bold; color: #1e40af; margin-bottom: 10px; }
                    .question-text { margin: 10px 0; line-height: 1.6; }
                    .options { margin: 10px 0; }
                    .option { margin: 5px 0; padding: 8px; background: #f9fafb; border-radius: 4px; }
                    .meta { font-size: 12px; color: #6b7280; margin-top: 10px; }
                    @media print { .no-print { display: none; } }
                </style>
            </head>
            <body>
                <h1>NEET ${paper_info.year || 2026} Predicted Paper</h1>
                <div class="paper-info">
                    <p><strong>Total Questions:</strong> ${paper_info.total_questions || 180}</p>
                    <p><strong>To Attempt:</strong> ${paper_info.to_attempt || 180}</p>
                    <p><strong>Duration:</strong> ${paper_info.duration_minutes || 180} minutes</p>
                    <p><strong>Pattern:</strong> ${paper_info.pattern_type === 'all_compulsory' ? 'All Compulsory (2026+)' : 'With Choice (2025)'}</p>
                    <p><strong>Total Marks:</strong> ${paper_info.total_marks || 720}</p>
                    <p><strong>Marking Scheme:</strong> +4 for correct, -1 for incorrect</p>
                </div>
                ${questions.map((q, index) => `
                    <div class="question">
                        <div class="question-number">Question ${index + 1}</div>
                        <div class="question-text">${q.question_text || 'Question not available'}</div>
                        ${q.options ? `
                            <div class="options">
                                ${q.options.map((opt, i) => `
                                    <div class="option">${String.fromCharCode(65 + i)}) ${opt}</div>
                                `).join('')}
                            </div>
                        ` : ''}
                        <div class="meta">
                            Chapter: ${q.chapter || 'General'} | 
                            Difficulty: ${q.difficulty || 'medium'} | 
                            Marks: ${q.marks || 4}
                        </div>
                    </div>
                `).join('')}
                <div class="no-print" style="text-align: center; margin: 30px 0;">
                    <button onclick="window.print()" style="padding: 10px 20px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px;">
                        Print / Save as PDF
                    </button>
                </div>
            </body>
            </html>
        `);
        printWindow.document.close();
        
        this.showToast('PDF preview opened in new window', 'success');
    }

    /**
     * Handle smart paper subject change - populate chapters
     * @param {Event} e - Change event
     */
    async handleSmartSubjectChange(e) {
        const subject = e.target.value;
        const chaptersSelector = document.getElementById('focus-chapters-selector');
        
        if (!subject || !chaptersSelector) return;

        try {
            // Show loading state
            chaptersSelector.innerHTML = '<option disabled>Loading chapters...</option>';
            chaptersSelector.disabled = true;

            // Get chapter analysis for the subject
            const analysisData = await this.api.getChapterAnalysis(subject);
            
            // Populate chapters
            if (analysisData && analysisData.analysis && analysisData.analysis.chapters) {
                const chapters = analysisData.analysis.chapters;
                
                // Sort by probability (highest first)
                chapters.sort((a, b) => (b.probability || 0) - (a.probability || 0));
                
                // Build options HTML
                let optionsHTML = '';
                chapters.forEach(chapter => {
                    const probability = Math.round((chapter.probability || 0) * 100);
                    const recommended = chapter.recommended ? ' ⭐' : '';
                    optionsHTML += `<option value="${chapter.name}">${chapter.name} (${probability}%)${recommended}</option>`;
                });
                
                chaptersSelector.innerHTML = optionsHTML;
                chaptersSelector.disabled = false;
            } else {
                chaptersSelector.innerHTML = '<option disabled>No chapters available</option>';
            }

        } catch (error) {
            console.error('Error loading chapters:', error);
            chaptersSelector.innerHTML = '<option disabled>Error loading chapters</option>';
            this.showToast('Failed to load chapters. You can still generate without selecting specific chapters.', 'warning');
            chaptersSelector.disabled = false;
        }
    }

    /**
     * Handle smart paper generation form submission
     * @param {Event} e - Form submit event
     */
    async handleSmartPaperGeneration(e) {
        e.preventDefault();

        const form = e.target;
        const subjectField = form.querySelector('#smart-subject-selector');
        const chaptersField = form.querySelector('#focus-chapters-selector');
        const difficultyField = form.querySelector('#difficulty-selector');
        
        const subject = subjectField?.value;
        const difficulty = difficultyField?.value;
        
        // Get selected chapters (multi-select)
        const selectedOptions = Array.from(chaptersField?.selectedOptions || []);
        const focusChapters = selectedOptions.map(option => option.value);
        
        const generateBtn = form.querySelector('.generate-btn');
        const resultsContainer = document.getElementById('smart-paper-results');

        // Clear previous errors
        this.clearFieldErrors(form);

        // Validate inputs
        let hasErrors = false;

        if (!subject || subject === '') {
            this.showFieldError('smart-subject-selector', 'Please select a subject');
            hasErrors = true;
        }

        if (!difficulty || difficulty === '') {
            this.showFieldError('difficulty-selector', 'Please select a difficulty level');
            hasErrors = true;
        }

        // Prevent submission if there are errors
        if (hasErrors) {
            this.showToast('Please fix the form errors before submitting', 'warning');
            return;
        }

        try {
            // Show loading state
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<span class="spinner small"></span> Generating...';
            resultsContainer.innerHTML = '<div class="loading-container"><div class="spinner"></div><p>Generating smart paper...</p></div>';
            resultsContainer.classList.remove('hidden');

            // Call API
            const response = await this.api.generateSmartPaper(subject, focusChapters, difficulty);
            
            // Extract paper data from response
            const paperData = response.paper || response;

            // Display results
            this.displaySmartPaper(paperData, resultsContainer);

            // Show success toast
            const chaptersText = focusChapters.length > 0 ? ` focusing on ${focusChapters.length} chapter(s)` : '';
            this.showToast(`Successfully generated ${difficulty} ${subject} smart paper${chaptersText}`, 'success');

            // Reset button
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg> Generate Smart Paper';

        } catch (error) {
            console.error('Smart paper generation error:', error);
            
            // Reset button
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg> Generate Smart Paper';

            // Handle errors
            if (error.message === 'PERMISSION_DENIED') {
                resultsContainer.innerHTML = this.getPermissionErrorHTML();
                this.showUpgradeModal('Smart Paper Generation');
            } else if (error.message === 'NETWORK_ERROR' || error.message === 'Failed to fetch') {
                resultsContainer.innerHTML = this.getErrorHTML('Network error. Please check your connection.', () => {
                    form.dispatchEvent(new Event('submit'));
                });
                this.showNetworkErrorModal(
                    'Unable to connect to the server. Please check your internet connection and try again.',
                    () => form.dispatchEvent(new Event('submit'))
                );
            } else {
                resultsContainer.innerHTML = this.getErrorHTML(error.message, () => {
                    form.dispatchEvent(new Event('submit'));
                });
                this.showError(error.message, () => form.dispatchEvent(new Event('submit')));
            }
            resultsContainer.classList.remove('hidden');
        }
    }

    /**
     * Display smart paper results
     * @param {Object} paperData - Smart paper data
     * @param {HTMLElement} container - Container element
     */
    displaySmartPaper(paperData, container) {
        // Safety checks for undefined data
        if (!paperData) {
            container.innerHTML = '<div class="error-message">No paper data received</div>';
            return;
        }
        
        // Store paper data for later use (Start Practice, Download)
        this.currentSmartPaper = paperData;
        
        const { paper_info, questions } = paperData;
        
        // Ensure questions is an array
        const questionsList = Array.isArray(questions) ? questions : [];
        
        const confidence = (paper_info && paper_info.prediction_confidence) ? paper_info.prediction_confidence : 0.75;
        const confidenceLevel = confidence >= 0.75 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';
        const confidencePercentage = Math.round(confidence * 100);

        // Build targeting information
        let targetingInfo = '';
        if (paper_info && paper_info.metadata) {
            const metadata = paper_info.metadata;
            
            if (metadata.focus_chapters && metadata.focus_chapters.length > 0) {
                targetingInfo += `
                    <div class="targeting-section">
                        <h4>📍 Focus Chapters</h4>
                        <div class="chapter-tags">
                            ${metadata.focus_chapters.map(chapter => `<span class="chapter-tag">${chapter}</span>`).join('')}
                        </div>
                    </div>
                `;
            }

            if (metadata.difficulty_level) {
                const difficultyEmoji = {
                    'easy': '🟢',
                    'medium': '🟡',
                    'hard': '🔴',
                    'mixed': '🎯'
                };
                targetingInfo += `
                    <div class="targeting-section">
                        <h4>${difficultyEmoji[metadata.difficulty_level] || '🎯'} Difficulty Level</h4>
                        <p class="difficulty-badge ${metadata.difficulty_level}">${metadata.difficulty_level.charAt(0).toUpperCase() + metadata.difficulty_level.slice(1)}</p>
                    </div>
                `;
            }

            if (metadata.weak_areas && metadata.weak_areas.length > 0) {
                targetingInfo += `
                    <div class="targeting-section">
                        <h4>🎯 Targeting Your Weak Areas</h4>
                        <ul class="weak-areas-list">
                            ${metadata.weak_areas.map(area => `<li>${area}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
        }

        // Build HTML
        const subject = (paper_info && paper_info.subject) ? paper_info.subject : 'Practice';
        
        const html = `
            <div class="smart-paper-display">
                <div class="paper-header">
                    <div class="paper-title">
                        <h3>⚡ Smart ${subject} Paper</h3>
                        <span class="confidence-badge ${confidenceLevel}">${confidencePercentage}% Confidence</span>
                    </div>
                    <p class="paper-description">Personalized practice paper based on your learning profile</p>
                </div>

                <div class="paper-stats">
                    <div class="stat-card">
                        <span class="stat-icon">📝</span>
                        <div class="stat-content">
                            <span class="stat-value">${questionsList.length}</span>
                            <span class="stat-label">Questions</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-icon">⏱</span>
                        <div class="stat-content">
                            <span class="stat-value">${paper_info.duration_minutes || (questionsList.length * 3)}</span>
                            <span class="stat-label">Minutes</span>
                        </div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-icon">🎯</span>
                        <div class="stat-content">
                            <span class="stat-value">${paper_info.total_marks || (questionsList.length * 4)}</span>
                            <span class="stat-label">Marks</span>
                        </div>
                    </div>
                </div>

                ${targetingInfo ? `
                    <div class="targeting-info">
                        <h3>🎯 Paper Targeting</h3>
                        ${targetingInfo}
                    </div>
                ` : ''}

                <div class="paper-actions">
                    <button class="btn-primary" onclick="window.predictionsUI.startSmartPaper('${paper_info.subject}', '${(paper_info.metadata && paper_info.metadata.difficulty_level) || paper_info.difficulty_level || 'mixed'}')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                        Start Practice
                    </button>
                    <button class="btn-secondary" onclick="window.predictionsUI.downloadSmartPaper('${paper_info.subject}')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                        Download PDF
                    </button>
                </div>

                <div class="paper-preview">
                    <h4>Question Preview</h4>
                    <div class="preview-questions">
                        ${questionsList.slice(0, 3).map((q, index) => `
                            <div class="preview-question">
                                <div class="question-header">
                                    <span class="question-number">Q${index + 1}</span>
                                    ${q.difficulty ? `<span class="difficulty-badge ${q.difficulty}">${q.difficulty}</span>` : ''}
                                    ${q.chapter ? `<span class="chapter-badge">${q.chapter}</span>` : ''}
                                </div>
                                <p class="question-text">${q.question_text}</p>
                            </div>
                        `).join('')}
                        ${questionsList.length > 3 ? `<p class="more-questions">+ ${questionsList.length - 3} more questions</p>` : ''}
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    /**
     * Start smart paper practice
     * @param {string} subject - Subject name
     * @param {string} difficulty - Difficulty level
     */
    startSmartPaper(subject, difficulty) {
        // Navigate to smart paper exam page
        window.location.href = `/exam/smart-paper?subject=${encodeURIComponent(subject)}&difficulty=${encodeURIComponent(difficulty)}`;
    }

    /**
     * Download smart paper as PDF
     * @param {string} subject - Subject name
     */
    async downloadSmartPaper(subject) {
        try {
            const paperData = this.currentSmartPaper;
            if (!paperData) {
                this.showToast('No paper data available to download', 'error');
                return;
            }
            
            // Show loading toast
            this.showToast('Generating PDF...', 'info');
            
            // Create PDF content
            const { paper_info, questions } = paperData;
            
            // Generate PDF using browser print
            const printWindow = window.open('', '_blank');
            const html = this.generatePrintableHTML(paper_info, questions);
            
            printWindow.document.write(html);
            printWindow.document.close();
            
            // Trigger print dialog
            setTimeout(() => {
                printWindow.print();
            }, 500);
            
        } catch (error) {
            console.error('Download error:', error);
            this.showToast('Failed to generate PDF', 'error');
        }
    }
    
    /**
     * Generate printable HTML for smart paper
     * @param {Object} paperInfo - Paper information
     * @param {Array} questions - Questions array
     * @returns {string} HTML string
     */
    generatePrintableHTML(paperInfo, questions) {
        const subject = paperInfo.subject || 'Practice';
        const difficulty = paperInfo.difficulty_level || 'Mixed';
        const date = new Date().toLocaleDateString();
        
        return `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Smart ${subject} Paper - ${difficulty}</title>
    <style>
        @media print {
            @page { margin: 1cm; }
            body { margin: 0; }
        }
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            color: #000;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20px;
            background: white;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #000;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
            text-transform: uppercase;
        }
        .header .subtitle {
            font-size: 14px;
            margin-top: 5px;
        }
        .info-section {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            padding: 10px;
            background: #f5f5f5;
            border: 1px solid #ddd;
        }
        .info-item {
            font-size: 12px;
        }
        .info-item strong {
            font-weight: bold;
        }
        .instructions {
            background: #fffacd;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #f0e68c;
            font-size: 12px;
        }
        .question {
            margin-bottom: 25px;
            page-break-inside: avoid;
        }
        .question-header {
            font-weight: bold;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .question-number {
            font-size: 14px;
        }
        .question-meta {
            font-size: 11px;
            color: #666;
        }
        .question-text {
            margin-bottom: 10px;
            font-size: 13px;
        }
        .options {
            margin-left: 20px;
        }
        .option {
            margin-bottom: 5px;
            font-size: 13px;
        }
        .answer-section {
            margin-top: 40px;
            page-break-before: always;
        }
        .answer-section h2 {
            border-bottom: 2px solid #000;
            padding-bottom: 5px;
        }
        .answer {
            margin-bottom: 15px;
            font-size: 12px;
        }
        .answer strong {
            color: #006400;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 11px;
            color: #666;
            border-top: 1px solid #ddd;
            padding-top: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Smart ${subject} Practice Paper</h1>
        <div class="subtitle">Difficulty: ${difficulty} | Date: ${date}</div>
    </div>
    
    <div class="info-section">
        <div class="info-item"><strong>Total Questions:</strong> ${questions.length}</div>
        <div class="info-item"><strong>Duration:</strong> ${paperInfo.duration_minutes || questions.length * 3} minutes</div>
        <div class="info-item"><strong>Total Marks:</strong> ${paperInfo.total_marks || questions.length * 4}</div>
        <div class="info-item"><strong>Marking:</strong> +4 for correct, -1 for incorrect</div>
    </div>
    
    <div class="instructions">
        <strong>Instructions:</strong>
        <ul style="margin: 5px 0; padding-left: 20px;">
            <li>All questions are compulsory</li>
            <li>Each question carries 4 marks</li>
            <li>There is negative marking of 1 mark for each incorrect answer</li>
            <li>Choose the most appropriate answer</li>
        </ul>
    </div>
    
    <div class="questions-section">
        ${questions.map((q, idx) => `
            <div class="question">
                <div class="question-header">
                    <span class="question-number">Question ${idx + 1}.</span>
                    <span class="question-meta">[${q.chapter || q.topic || subject}] [${q.difficulty || difficulty}]</span>
                </div>
                <div class="question-text">${q.question_text}</div>
                <div class="options">
                    ${q.options.map((opt, optIdx) => `
                        <div class="option">${String.fromCharCode(65 + optIdx)}) ${opt}</div>
                    `).join('')}
                </div>
            </div>
        `).join('')}
    </div>
    
    <div class="answer-section">
        <h2>Answer Key & Explanations</h2>
        ${questions.map((q, idx) => `
            <div class="answer">
                <strong>Q${idx + 1}:</strong> ${q.correct_answer}
                ${q.solution ? `<br><em>Explanation:</em> ${q.solution}` : ''}
            </div>
        `).join('')}
    </div>
    
    <div class="footer">
        Generated by VidyaTid - Smart Learning Platform | ${date}
    </div>
</body>
</html>
        `;
    }

    /**
     * Load and display prediction insights for a subject
     * @param {string} subject - Subject name (Physics, Chemistry, Biology)
     */
    async loadPredictionInsights(subject) {
        const container = document.getElementById('insights-container');
        if (!container) return;

        try {
            // Show skeleton loader
            this.showSkeletonLoader(container, 'card');
            container.classList.remove('hidden');

            // Call API
            const insightsData = await this.api.getInsights(subject);

            // Display insights
            this.displayPredictionInsights(subject, insightsData, container);

            // Show success toast
            this.showToast(`Loaded ${subject} insights`, 'success', 3000);

        } catch (error) {
            console.error('Insights error:', error);
            
            if (error.message === 'PERMISSION_DENIED') {
                container.innerHTML = this.getPermissionErrorHTML();
                this.showUpgradeModal('Prediction Insights');
            } else if (error.message === 'NETWORK_ERROR' || error.message === 'Failed to fetch') {
                container.innerHTML = this.getErrorHTML('Network error. Please check your connection.', () => {
                    this.loadPredictionInsights(subject);
                });
                this.showNetworkErrorModal(
                    'Unable to load insights. Please check your internet connection and try again.',
                    () => this.loadPredictionInsights(subject)
                );
            } else {
                container.innerHTML = this.getErrorHTML(error.message, () => {
                    this.loadPredictionInsights(subject);
                });
                this.showError(error.message, () => this.loadPredictionInsights(subject));
            }
        }
    }

    /**
     * Display prediction insights
     * @param {string} subject - Subject name
     * @param {Object} insightsData - Insights data
     * @param {HTMLElement} container - Container element
     */
    displayPredictionInsights(subject, insightsData, container) {
        const {
            high_probability_chapters,
            recommended_focus,
            difficulty_trend,
            data_confidence,
            total_questions_analyzed
        } = insightsData;

        // Build HTML
        let html = `
            <div class="insights-content">
                <div class="insights-header">
                    <h3>${subject} Prediction Insights</h3>
                    ${this.getConfidenceBadge(data_confidence)}
                </div>

                <div class="insights-grid">
                    <!-- High Probability Topics -->
                    <div class="insight-card">
                        <h4>High Probability Topics</h4>
                        <ul class="topic-list">
                            ${high_probability_chapters.map(topic => `
                                <li class="topic-list-item">
                                    <span class="topic-bullet">●</span>
                                    <span class="topic-text">${topic}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>

                    <!-- Recommended Focus Areas -->
                    <div class="insight-card">
                        <h4>Recommended Focus Areas</h4>
                        <ul class="topic-list">
                            ${recommended_focus.map(area => `
                                <li class="topic-list-item">
                                    <span class="topic-bullet">⭐</span>
                                    <span class="topic-text">${area}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                </div>

                <!-- Difficulty Distribution Chart -->
                <div class="chart-container">
                    <h4>Difficulty Distribution</h4>
                    <div class="chart-wrapper medium">
                        <canvas id="difficulty-chart"></canvas>
                    </div>
                </div>

                <!-- Topic Frequency Chart -->
                <div class="chart-container">
                    <h4>Topic Frequency Analysis</h4>
                    <div class="chart-wrapper medium">
                        <canvas id="topic-frequency-chart"></canvas>
                    </div>
                </div>

                <div class="insights-footer">
                    <p><strong>Analysis Based On:</strong> ${total_questions_analyzed} questions analyzed</p>
                    <p><strong>Data Confidence:</strong> ${Math.round(data_confidence * 100)}%</p>
                </div>
            </div>
        `;

        container.innerHTML = html;

        // Render charts after DOM update
        setTimeout(() => {
            this.renderInsightsCharts(difficulty_trend, high_probability_chapters);
        }, 100);
    }

    /**
     * Render charts for insights with lazy loading
     * @param {Object} difficultyTrend - Difficulty distribution data
     * @param {Array} topics - High probability topics
     */
    renderInsightsCharts(difficultyTrend, topics) {
        // Use requestAnimationFrame to defer chart rendering
        requestAnimationFrame(() => {
            const difficultyCanvas = document.getElementById('difficulty-chart');
            const topicCanvas = document.getElementById('topic-frequency-chart');

            if (difficultyCanvas && difficultyTrend) {
                // Register for lazy loading
                this.chartManager.registerLazyChart('difficulty-chart', {
                    type: 'difficultyDistribution',
                    data: difficultyTrend
                });
            }

            if (topicCanvas && topics && topics.length > 0) {
                // Convert topics to frequency data (mock data for now)
                const topicData = topics.map((topic, index) => ({
                    topic: topic,
                    frequency: topics.length - index // Descending frequency
                }));
                
                // Register for lazy loading
                this.chartManager.registerLazyChart('topic-frequency-chart', {
                    type: 'topicFrequency',
                    data: {
                        topics: topicData,
                        sortOrder: 'desc'
                    }
                });
            }
        });
    }

    /**
     * Show skeleton loader for a container
     * @param {HTMLElement} container - Container element
     * @param {string} type - Type of skeleton (card, list, chart)
     */
    showSkeletonLoader(container, type = 'card') {
        if (!container) return;

        let skeletonHTML = '';

        switch (type) {
            case 'card':
                skeletonHTML = `
                    <div class="skeleton skeleton-card"></div>
                `;
                break;
            case 'list':
                skeletonHTML = `
                    <div class="skeleton skeleton-title"></div>
                    <div class="skeleton skeleton-text"></div>
                    <div class="skeleton skeleton-text"></div>
                    <div class="skeleton skeleton-text"></div>
                `;
                break;
            case 'chart':
                skeletonHTML = `
                    <div class="skeleton skeleton-card"></div>
                `;
                break;
            default:
                skeletonHTML = `
                    <div class="skeleton skeleton-text"></div>
                    <div class="skeleton skeleton-text"></div>
                `;
        }

        container.innerHTML = skeletonHTML;
    }

    /**
     * Load latest prediction (quick action)
     */
    async loadLatestPrediction() {
        // Get the most recent prediction from storage or default to Physics 2026
        const subject = 'Physics';
        const year = 2026;
        
        // Scroll to subject prediction panel
        const panel = document.getElementById('subject-prediction-panel');
        if (panel) {
            panel.scrollIntoView({ behavior: 'smooth' });
            
            // Pre-fill form
            const subjectSelect = document.getElementById('subject-selector');
            const yearSelect = document.getElementById('year-selector');
            if (subjectSelect) subjectSelect.value = subject;
            if (yearSelect) yearSelect.value = year;
        }
    }

    /**
     * Show high probability topics (quick action)
     */
    async showHighProbabilityTopics() {
        // Default to Physics
        const subject = 'Physics';
        
        // Scroll to insights panel
        const panel = document.getElementById('insights-panel');
        if (panel) {
            panel.scrollIntoView({ behavior: 'smooth' });
            
            // Load insights
            await this.loadPredictionInsights(subject);
        }
    }

    /**
     * Show accuracy report (quick action)
     */
    async showAccuracyReport() {
        // Scroll to accuracy tracker
        const tracker = document.getElementById('accuracy-tracker');
        if (tracker) {
            tracker.scrollIntoView({ behavior: 'smooth' });
            
            // Load accuracy data (placeholder)
            await this.loadAccuracyData();
        }
    }

    /**
     * Load accuracy data (placeholder)
     */
    async loadAccuracyData() {
        // This would call an accuracy API endpoint
        // For now, just show a placeholder message
        console.log('Loading accuracy data...');
    }

    /**
     * Show toast notification
     * @param {string} message - Toast message
     * @param {string} type - Toast type (success, error, warning, info)
     * @param {number} duration - Duration in milliseconds (default: 5000)
     */
    showToast(message, type = 'info', duration = 5000) {
        // Create toast container if it doesn't exist
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }

        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        // Icon based on type
        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };

        // Titles based on type
        const titles = {
            success: 'Success',
            error: 'Error',
            warning: 'Warning',
            info: 'Info'
        };

        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-content">
                <div class="toast-title">${titles[type] || titles.info}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;

        container.appendChild(toast);

        // Auto-remove after duration
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, duration);
    }

    /**
     * Show network error modal with retry option
     * @param {string} message - Error message
     * @param {Function} retryCallback - Function to call on retry
     * @param {number} retryCount - Current retry count
     */
    showNetworkErrorModal(message, retryCallback, retryCount = 0) {
        // Remove existing modal if any
        const existingModal = document.querySelector('.network-error-overlay');
        if (existingModal) {
            existingModal.remove();
        }

        // Create modal overlay
        const overlay = document.createElement('div');
        overlay.className = 'network-error-overlay';
        
        overlay.innerHTML = `
            <div class="network-error-modal">
                <div class="network-error-icon">⚠</div>
                <h3 class="network-error-title">Connection Error</h3>
                <p class="network-error-message">${message}</p>
                <div class="network-error-actions">
                    <button class="btn-primary" id="network-retry-btn">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="23 4 23 10 17 10"></polyline>
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                        </svg>
                        Retry
                    </button>
                    <button class="btn-secondary" id="network-cancel-btn">Cancel</button>
                </div>
                ${retryCount > 0 ? `<p class="retry-count">Retry attempt ${retryCount}</p>` : ''}
            </div>
        `;

        document.body.appendChild(overlay);

        // Add event listeners
        const retryBtn = overlay.querySelector('#network-retry-btn');
        const cancelBtn = overlay.querySelector('#network-cancel-btn');

        retryBtn.addEventListener('click', () => {
            overlay.remove();
            if (retryCallback) {
                retryCallback();
            }
        });

        cancelBtn.addEventListener('click', () => {
            overlay.remove();
        });

        // Close on overlay click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.remove();
            }
        });
    }

    /**
     * Show upgrade modal for premium features
     * @param {string} featureName - Name of the premium feature
     */
    showUpgradeModal(featureName = 'AI Predictions') {
        // Remove existing modal if any
        const existingModal = document.querySelector('.upgrade-modal-overlay');
        if (existingModal) {
            existingModal.remove();
        }

        // Create modal overlay
        const overlay = document.createElement('div');
        overlay.className = 'upgrade-modal-overlay';
        
        overlay.innerHTML = `
            <div class="upgrade-modal">
                <button class="upgrade-modal-close" id="upgrade-close-btn">×</button>
                <div class="upgrade-modal-header">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                    </svg>
                    <h3 class="upgrade-modal-title">Premium Feature Locked</h3>
                    <p class="upgrade-modal-subtitle">
                        ${featureName} requires a <strong>Premium</strong> or <strong>Ultimate</strong> subscription
                    </p>
                </div>
                
                <div class="upgrade-tiers">
                    <div class="tier-card">
                        <div class="tier-badge">Premium</div>
                        <div class="tier-price">₹499/month</div>
                        <div class="tier-features">
                            <div class="tier-feature">✓ All prediction features</div>
                            <div class="tier-feature">✓ 200 queries per day</div>
                            <div class="tier-feature">✓ Chapter analysis</div>
                        </div>
                    </div>
                    <div class="tier-card recommended">
                        <div class="tier-badge recommended">Ultimate</div>
                        <div class="tier-price">₹999/month</div>
                        <div class="tier-features">
                            <div class="tier-feature">✓ Everything in Premium</div>
                            <div class="tier-feature">✓ Unlimited queries</div>
                            <div class="tier-feature">✓ Priority support</div>
                        </div>
                    </div>
                </div>
                
                <ul class="upgrade-features">
                    <li class="upgrade-feature">
                        <div class="upgrade-feature-icon">✓</div>
                        <div class="upgrade-feature-text">AI-powered question predictions with 75-85% accuracy</div>
                    </li>
                    <li class="upgrade-feature">
                        <div class="upgrade-feature-icon">✓</div>
                        <div class="upgrade-feature-text">Chapter-wise probability analysis for all subjects</div>
                    </li>
                    <li class="upgrade-feature">
                        <div class="upgrade-feature-icon">✓</div>
                        <div class="upgrade-feature-text">Complete 200-question NEET predicted papers</div>
                    </li>
                    <li class="upgrade-feature">
                        <div class="upgrade-feature-icon">✓</div>
                        <div class="upgrade-feature-text">Smart paper generation based on your weak areas</div>
                    </li>
                    <li class="upgrade-feature">
                        <div class="upgrade-feature-icon">✓</div>
                        <div class="upgrade-feature-text">Detailed prediction insights and trend analysis</div>
                    </li>
                </ul>
                <div class="upgrade-modal-actions">
                    <button class="btn-primary" onclick="window.location.href='/pricing'">
                        View All Plans & Pricing
                    </button>
                    <button class="btn-secondary" id="upgrade-cancel-btn">Maybe Later</button>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        // Add event listeners
        const closeBtn = overlay.querySelector('#upgrade-close-btn');
        const cancelBtn = overlay.querySelector('#upgrade-cancel-btn');

        closeBtn.addEventListener('click', () => {
            overlay.remove();
        });

        cancelBtn.addEventListener('click', () => {
            overlay.remove();
        });

        // Close on overlay click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.remove();
            }
        });
    }

    /**
     * Get confidence badge HTML
     * @param {number} confidence - Confidence score (0-1)
     * @returns {string} - HTML for confidence badge
     */
    getConfidenceBadge(confidence) {
        const percentage = Math.round(confidence * 100);
        const level = confidence >= 0.75 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';
        return `<span class="confidence-badge ${level}">${percentage}% Confidence</span>`;
    }

    /**
     * Load and display prediction insights for a subject
     * @param {string} subject - Subject name (Physics, Chemistry, Biology)
     */
    async loadPredictionInsights(subject) {
        console.log('Loading prediction insights for', subject);
        
        // Show loading state
        const loadingState = document.getElementById('insights-loading');
        if (loadingState) {
            loadingState.classList.remove('hidden');
        }
        
        try {
            // Call API to get insights
            const insightsData = await this.api.getInsights(subject);
            
            // Hide loading state
            if (loadingState) {
                loadingState.classList.add('hidden');
            }
            
            // Populate High Probability Topics
            const highProbList = document.getElementById('high-probability-list');
            if (highProbList && insightsData.high_probability_chapters) {
                highProbList.innerHTML = insightsData.high_probability_chapters.map(topic => `
                    <li class="topic-list-item">
                        <span class="topic-bullet">•</span>
                        <span class="topic-text">${topic}</span>
                    </li>
                `).join('');
            }
            
            // Populate Recommended Focus Areas
            const focusList = document.getElementById('recommended-focus-list');
            if (focusList && insightsData.recommended_focus) {
                focusList.innerHTML = insightsData.recommended_focus.map(area => `
                    <li class="topic-list-item">
                        <span class="topic-bullet">•</span>
                        <span class="topic-text">${area}</span>
                    </li>
                `).join('');
            }
            
            // Populate Trend Analysis
            const trendSummary = document.getElementById('trend-summary');
            if (trendSummary && insightsData.trend_analysis) {
                trendSummary.innerHTML = `
                    <p><strong>Analysis:</strong> ${insightsData.trend_analysis}</p>
                    <p><strong>Total Questions Analyzed:</strong> ${insightsData.total_questions_analyzed || 0}</p>
                    <p><strong>Data Confidence:</strong> ${Math.round((insightsData.data_confidence || 0) * 100)}%</p>
                `;
            }
            
            this.showToast(`${subject} insights loaded successfully`, 'success', 3000);
            
        } catch (error) {
            console.error('Insights error:', error);
            
            // Hide loading state
            if (loadingState) {
                loadingState.classList.add('hidden');
            }
            
            // Handle errors
            if (error.message === 'PERMISSION_DENIED') {
                // Show upgrade modal for premium feature
                this.showUpgradeModal('Prediction Insights');
                
                // Show error in the lists
                const highProbList = document.getElementById('high-probability-list');
                const focusList = document.getElementById('recommended-focus-list');
                const trendSummary = document.getElementById('trend-summary');
                
                const errorHTML = '<li class="error-state">Premium feature - Upgrade to access</li>';
                if (highProbList) highProbList.innerHTML = errorHTML;
                if (focusList) focusList.innerHTML = errorHTML;
                if (trendSummary) trendSummary.innerHTML = '<p class="error-state">Premium feature - Upgrade to access</p>';
                
            } else if (error.message === 'NETWORK_ERROR' || error.message === 'Failed to fetch') {
                this.showNetworkErrorModal(
                    'Unable to load insights. Please check your internet connection and try again.',
                    () => this.loadPredictionInsights(subject)
                );
            } else {
                this.showError(error.message, () => this.loadPredictionInsights(subject));
            }
        }
    }

    /**
     * Load accuracy data for the prediction system
     */
    async loadAccuracyData() {
        // For now, just show a message
        console.log('Loading accuracy data...');
        
        // Mock data for demonstration
        const accuracyData = [
            { year: 2020, accuracy: 0.78 },
            { year: 2021, accuracy: 0.82 },
            { year: 2022, accuracy: 0.85 },
            { year: 2023, accuracy: 0.83 },
            { year: 2024, accuracy: 0.87 }
        ];

        // Update accuracy display
        const overallAccuracy = document.getElementById('overall-accuracy-value');
        if (overallAccuracy) {
            overallAccuracy.textContent = '83%';
        }

        // Render accuracy trend chart
        const trendCanvas = document.getElementById('accuracy-trend-chart');
        if (trendCanvas) {
            this.chartManager.createAccuracyTrendChart('accuracy-trend-chart', accuracyData);
        }
    }

    /**
     * Display confidence score with color coding
     * @param {number} score - Confidence score (0-1)
     * @returns {string} - HTML for confidence badge
     */
    getConfidenceBadge(score) {
        let level = 'low';
        if (score >= 0.75) level = 'high';
        else if (score >= 0.5) level = 'medium';

        const percentage = Math.round(score * 100);
        return `<span class="confidence-badge ${level}">${percentage}% Confidence</span>`;
    }

    /**
     * Create progress bar HTML
     * @param {number} percentage - Progress percentage (0-100)
     * @returns {string} - HTML for progress bar
     */
    getProgressBar(percentage) {
        return `
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: ${percentage}%"></div>
            </div>
        `;
    }

    /**
     * View full paper with all questions
     * @param {string} subject - Subject name
     * @param {number} year - Year
     */
    viewFullPaper(subject, year) {
        console.log(`Viewing full paper for ${subject} ${year}`);
        
        // Create a modal or new page to show all questions
        const modal = document.createElement('div');
        modal.className = 'full-paper-modal';
        modal.innerHTML = `
            <div class="full-paper-content">
                <div class="full-paper-header">
                    <h2>${subject} - ${year} Predicted Paper</h2>
                    <button class="close-btn" onclick="this.closest('.full-paper-modal').remove()">×</button>
                </div>
                <div class="full-paper-body">
                    <p>Loading full paper...</p>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Load the full paper data
        this.api.predictPaper(subject, year, true, true).then(data => {
            const paper = data.paper || data;
            const questions = paper.questions || [];
            
            const questionsHTML = questions.map((q, index) => `
                <div class="full-question-item">
                    <div class="question-header">
                        <span class="q-number">Question ${index + 1}</span>
                        <span class="q-chapter">${q.chapter || 'Unknown'}</span>
                        <span class="q-difficulty ${q.difficulty}">${q.difficulty || 'medium'}</span>
                        <span class="q-marks">${q.marks || 4} marks</span>
                    </div>
                    <div class="question-text">${q.question_text || 'Question not available'}</div>
                    ${q.options ? `
                        <div class="question-options">
                            ${q.options.map((opt, i) => `
                                <div class="option">
                                    <span class="option-label">${String.fromCharCode(65 + i)}.</span>
                                    <span class="option-text">${opt}</span>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `).join('');
            
            modal.querySelector('.full-paper-body').innerHTML = questionsHTML;
        }).catch(error => {
            console.error('Error loading full paper:', error);
            modal.querySelector('.full-paper-body').innerHTML = '<p class="error">Failed to load paper</p>';
        });
    }

    /**
     * Download paper as PDF
     * @param {string} subject - Subject name
     * @param {number} year - Year
     */
    downloadPaper(subject, year) {
        console.log(`Downloading paper for ${subject} ${year}`);
        this.showToast('PDF download feature coming soon!', 'info');
        
        // TODO: Implement PDF generation and download
        // This would typically call a backend endpoint that generates a PDF
    }

    /**
     * Start practice test
     * @param {string} subject - Subject name
     * @param {number} year - Year
     */
    startPracticeTest(subject, year) {
        console.log(`Starting practice test for ${subject} ${year}`);
        this.showToast('Practice test feature coming soon!', 'info');
        
        // TODO: Implement practice test mode
        // This would navigate to a test interface with timer and scoring
    }


    /**
     * Display chapter analysis results
     * @param {Object} analysisData - Chapter analysis data from API
     * @param {HTMLElement} container - Container element
     */
    displayChapterAnalysis(analysisData, container) {
        if (!analysisData || !analysisData.analysis) {
            container.innerHTML = '<p class="no-data">No chapter analysis data available</p>';
            return;
        }

        const chapters = analysisData.analysis.chapters || [];
        const totalAnalyzed = analysisData.analysis.total_analyzed || 0;

        if (chapters.length === 0) {
            container.innerHTML = `
                <div class="no-data-state">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    <h3>No Data Available</h3>
                    <p>No chapter analysis data found for this subject.</p>
                </div>
            `;
            return;
        }

        // Build chapter list HTML
        let html = `
            <div class="chapter-analysis-results">
                <div class="analysis-summary">
                    <div class="summary-card">
                        <span class="summary-label">Total Chapters Analyzed</span>
                        <span class="summary-value">${chapters.length}</span>
                    </div>
                    <div class="summary-card">
                        <span class="summary-label">Questions Analyzed</span>
                        <span class="summary-value">${totalAnalyzed}</span>
                    </div>
                    <div class="summary-card">
                        <span class="summary-label">High Priority Chapters</span>
                        <span class="summary-value">${chapters.filter(ch => ch.probability >= 0.75).length}</span>
                    </div>
                </div>

                <div class="chapters-grid">
        `;

        // Add each chapter card
        chapters.forEach((chapter, index) => {
            const probability = chapter.probability || 0;
            const frequency = chapter.frequency || 0;
            const recommended = chapter.recommended || false;
            const probabilityPercent = Math.round(probability * 100);
            
            // Determine priority level
            let priorityClass = 'low';
            let priorityLabel = 'Low Priority';
            let priorityIcon = '🔵';
            
            if (probability >= 0.75) {
                priorityClass = 'high';
                priorityLabel = 'High Priority';
                priorityIcon = '🔴';
            } else if (probability >= 0.5) {
                priorityClass = 'medium';
                priorityLabel = 'Medium Priority';
                priorityIcon = '🟡';
            }

            html += `
                <div class="chapter-card ${priorityClass}" data-chapter="${chapter.name}">
                    <div class="chapter-card-header">
                        <div class="chapter-rank">#${index + 1}</div>
                        <div class="chapter-priority ${priorityClass}">
                            <span class="priority-icon">${priorityIcon}</span>
                            <span class="priority-label">${priorityLabel}</span>
                        </div>
                    </div>
                    
                    <div class="chapter-card-body">
                        <h4 class="chapter-name">${chapter.name}</h4>
                        
                        <div class="chapter-stats">
                            <div class="stat-item">
                                <span class="stat-icon">📊</span>
                                <div class="stat-content">
                                    <span class="stat-label">Frequency</span>
                                    <span class="stat-value">${frequency} times</span>
                                </div>
                            </div>
                            
                            <div class="stat-item">
                                <span class="stat-icon">🎯</span>
                                <div class="stat-content">
                                    <span class="stat-label">Probability</span>
                                    <span class="stat-value">${probabilityPercent}%</span>
                                </div>
                            </div>
                        </div>

                        <div class="probability-visualization">
                            <div class="probability-bar-container">
                                <div class="probability-bar-fill ${priorityClass}" style="width: ${probabilityPercent}%">
                                    <span class="probability-bar-label">${probabilityPercent}%</span>
                                </div>
                            </div>
                        </div>

                        ${recommended ? '<div class="recommended-badge">⭐ Recommended</div>' : ''}
                    </div>

                    <div class="chapter-card-footer">
                        <button class="chapter-action-btn" onclick="window.predictionsUI.viewChapterDetails('${chapter.name}')">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="16" x2="12" y2="12"></line>
                                <line x1="12" y1="8" x2="12.01" y2="8"></line>
                            </svg>
                            View Details
                        </button>
                        <button class="chapter-action-btn" onclick="window.predictionsUI.practiceChapter('${chapter.name}')">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="5 3 19 12 5 21 5 3"></polygon>
                            </svg>
                            Practice
                        </button>
                    </div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;

        container.innerHTML = html;
        container.classList.remove('hidden');
    }

    /**
     * View chapter details
     * @param {string} chapterName - Chapter name
     */
    viewChapterDetails(chapterName) {
        console.log(`Viewing details for chapter: ${chapterName}`);
        this.showToast(`Chapter details for "${chapterName}" coming soon!`, 'info');
        // TODO: Implement chapter details modal
    }

    /**
     * Practice chapter
     * @param {string} chapterName - Chapter name
     */
    practiceChapter(chapterName) {
        console.log(`Starting practice for chapter: ${chapterName}`);
        this.showToast(`Practice mode for "${chapterName}" coming soon!`, 'info');
        // TODO: Implement chapter practice mode
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.predictionsUI = new PredictionsUI();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PredictionAPI, PredictionsUI };
}
