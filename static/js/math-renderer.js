// Mathematical Expression Renderer using KaTeX
// Handles both inline ($...$) and display ($$...$$) math modes

const MathRenderer = {
    // Initialize KaTeX auto-render when DOM is ready
    init() {
        // Wait for KaTeX to load
        if (typeof renderMathInElement === 'undefined') {
            console.warn('KaTeX auto-render not loaded yet, retrying...');
            setTimeout(() => this.init(), 100);
            return;
        }
        console.log('MathRenderer initialized');
    },

    // Render math in a specific element
    renderMath(element) {
        if (!element) return;
        
        // Wait for KaTeX to be available
        if (typeof renderMathInElement === 'undefined') {
            console.warn('KaTeX not loaded, deferring math rendering');
            setTimeout(() => this.renderMath(element), 100);
            return;
        }

        try {
            renderMathInElement(element, {
                // Delimiters for inline and display math
                delimiters: [
                    {left: '$$', right: '$$', display: true},   // Display mode (centered, larger)
                    {left: '$', right: '$', display: false},    // Inline mode
                    {left: '\\[', right: '\\]', display: true}, // LaTeX display mode
                    {left: '\\(', right: '\\)', display: false} // LaTeX inline mode
                ],
                // Throw on error to catch issues
                throwOnError: false,
                // Error color
                errorColor: '#cc0000',
                // Ignore certain HTML tags
                ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                // Trust certain commands
                trust: true
            });
        } catch (error) {
            console.error('Error rendering math:', error);
        }
    },

    // Render math in text content and return HTML
    renderMathInText(text) {
        if (!text) return '';
        
        // Create a temporary element to render math
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = text;
        
        // Render math in the temporary element
        this.renderMath(tempDiv);
        
        return tempDiv.innerHTML;
    },

    // Check if text contains LaTeX expressions
    containsMath(text) {
        if (!text) return false;
        
        // Check for common LaTeX delimiters
        const mathPatterns = [
            /\$\$[\s\S]+?\$\$/,  // Display math $$...$$
            /\$[^\$\n]+?\$/,      // Inline math $...$
            /\\\[[\s\S]+?\\\]/,   // Display math \[...\]
            /\\\([\s\S]+?\\\)/,   // Inline math \(...\)
            /\\frac\{/,           // Fractions
            /\\sqrt\{/,           // Square roots
            /\\sum/,              // Summation
            /\\int/,              // Integral
            /\\lim/,              // Limit
            /\\alpha|\\beta|\\gamma|\\delta|\\theta|\\pi|\\sigma/, // Greek letters
            /\^[{]?[^}]+[}]?/,    // Superscripts
            /_[{]?[^}]+[}]?/      // Subscripts
        ];
        
        return mathPatterns.some(pattern => pattern.test(text));
    },

    // Format common mathematical expressions
    formatMathExpression(text) {
        if (!text) return '';
        
        // Convert common patterns to LaTeX if not already formatted
        let formatted = text;
        
        // Don't modify if already has LaTeX delimiters
        if (this.containsMath(formatted)) {
            return formatted;
        }
        
        // Convert simple patterns (be conservative to avoid false positives)
        // These are just helpers for plain text that should be math
        
        return formatted;
    },

    // Escape HTML but preserve LaTeX delimiters
    escapeHtmlPreserveMath(text) {
        if (!text) return '';
        
        // Split by math delimiters
        const parts = [];
        let current = '';
        let inMath = false;
        let mathDelimiter = '';
        
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            const next = text[i + 1];
            
            // Check for math delimiters
            if (!inMath) {
                if (char === '$' && next === '$') {
                    parts.push({type: 'text', content: current});
                    current = '$$';
                    inMath = true;
                    mathDelimiter = '$$';
                    i++; // Skip next $
                } else if (char === '$') {
                    parts.push({type: 'text', content: current});
                    current = '$';
                    inMath = true;
                    mathDelimiter = '$';
                } else if (char === '\\' && next === '[') {
                    parts.push({type: 'text', content: current});
                    current = '\\[';
                    inMath = true;
                    mathDelimiter = '\\]';
                    i++; // Skip next [
                } else if (char === '\\' && next === '(') {
                    parts.push({type: 'text', content: current});
                    current = '\\(';
                    inMath = true;
                    mathDelimiter = '\\)';
                    i++; // Skip next (
                } else {
                    current += char;
                }
            } else {
                current += char;
                
                // Check for closing delimiter
                if (mathDelimiter === '$$' && char === '$' && next === '$') {
                    current += '$';
                    parts.push({type: 'math', content: current});
                    current = '';
                    inMath = false;
                    mathDelimiter = '';
                    i++; // Skip next $
                } else if (mathDelimiter === '$' && char === '$') {
                    parts.push({type: 'math', content: current});
                    current = '';
                    inMath = false;
                    mathDelimiter = '';
                } else if (mathDelimiter === '\\]' && char === ']' && text[i-1] === '\\') {
                    parts.push({type: 'math', content: current});
                    current = '';
                    inMath = false;
                    mathDelimiter = '';
                } else if (mathDelimiter === '\\)' && char === ')' && text[i-1] === '\\') {
                    parts.push({type: 'math', content: current});
                    current = '';
                    inMath = false;
                    mathDelimiter = '';
                }
            }
        }
        
        // Add remaining content
        if (current) {
            parts.push({type: inMath ? 'math' : 'text', content: current});
        }
        
        // Escape HTML in text parts only
        return parts.map(part => {
            if (part.type === 'text') {
                return part.content
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
            }
            return part.content;
        }).join('');
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => MathRenderer.init());
} else {
    MathRenderer.init();
}

// Export for use in other scripts
window.MathRenderer = MathRenderer;
