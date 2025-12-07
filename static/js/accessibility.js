/**
 * Accessibility Enhancements - VidyaTid
 * Keyboard navigation, ARIA attributes, and screen reader support
 */

class AccessibilityManager {
    constructor() {
        this.keyboardNavActive = false;
        this.init();
    }

    init() {
        this.setupKeyboardNavigation();
        this.setupFocusTrap();
        this.setupARIAAttributes();
        this.setupSkipLinks();
        this.setupAnnouncements();
    }

    // Keyboard Navigation
    setupKeyboardNavigation() {
        // Detect keyboard usage
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                this.keyboardNavActive = true;
                document.body.classList.add('keyboard-nav');
            }
        });

        document.addEventListener('mousedown', () => {
            this.keyboardNavActive = false;
            document.body.classList.remove('keyboard-nav');
        });

        // Escape key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeActiveModal();
            }
        });

        // Arrow key navigation for menus
        document.querySelectorAll('[role="menu"]').forEach(menu => {
            this.setupMenuNavigation(menu);
        });

        // Tab navigation for tab panels
        document.querySelectorAll('[role="tablist"]').forEach(tablist => {
            this.setupTabNavigation(tablist);
        });
    }

    setupMenuNavigation(menu) {
        const items = menu.querySelectorAll('[role="menuitem"]');
        let currentIndex = 0;

        menu.addEventListener('keydown', (e) => {
            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    currentIndex = (currentIndex + 1) % items.length;
                    items[currentIndex].focus();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    currentIndex = (currentIndex - 1 + items.length) % items.length;
                    items[currentIndex].focus();
                    break;
                case 'Home':
                    e.preventDefault();
                    currentIndex = 0;
                    items[currentIndex].focus();
                    break;
                case 'End':
                    e.preventDefault();
                    currentIndex = items.length - 1;
                    items[currentIndex].focus();
                    break;
            }
        });
    }

    setupTabNavigation(tablist) {
        const tabs = tablist.querySelectorAll('[role="tab"]');
        
        tabs.forEach((tab, index) => {
            tab.addEventListener('keydown', (e) => {
                let newIndex = index;
                
                switch (e.key) {
                    case 'ArrowLeft':
                        e.preventDefault();
                        newIndex = (index - 1 + tabs.length) % tabs.length;
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        newIndex = (index + 1) % tabs.length;
                        break;
                    case 'Home':
                        e.preventDefault();
                        newIndex = 0;
                        break;
                    case 'End':
                        e.preventDefault();
                        newIndex = tabs.length - 1;
                        break;
                    default:
                        return;
                }
                
                tabs[newIndex].click();
                tabs[newIndex].focus();
            });
        });
    }

    // Focus Trap for Modals
    setupFocusTrap() {
        document.addEventListener('focusin', (e) => {
            const activeModal = document.querySelector('.modal.active, .overlay.active');
            if (!activeModal) return;

            const focusableElements = activeModal.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );

            if (focusableElements.length === 0) return;

            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];

            if (!activeModal.contains(e.target)) {
                firstElement.focus();
            }
        });
    }

    // ARIA Attributes
    setupARIAAttributes() {
        // Add ARIA labels to buttons without text
        document.querySelectorAll('button:not([aria-label])').forEach(button => {
            if (!button.textContent.trim()) {
                const icon = button.querySelector('svg, img');
                if (icon) {
                    button.setAttribute('aria-label', 'Button');
                }
            }
        });

        // Add ARIA labels to links without text
        document.querySelectorAll('a:not([aria-label])').forEach(link => {
            if (!link.textContent.trim()) {
                const icon = link.querySelector('svg, img');
                if (icon) {
                    link.setAttribute('aria-label', 'Link');
                }
            }
        });

        // Add ARIA expanded to collapsible elements
        document.querySelectorAll('[data-toggle]').forEach(toggle => {
            toggle.setAttribute('aria-expanded', 'false');
            
            toggle.addEventListener('click', () => {
                const expanded = toggle.getAttribute('aria-expanded') === 'true';
                toggle.setAttribute('aria-expanded', !expanded);
            });
        });

        // Add ARIA live regions for dynamic content
        const liveRegions = document.querySelectorAll('[data-live]');
        liveRegions.forEach(region => {
            region.setAttribute('aria-live', region.dataset.live || 'polite');
            region.setAttribute('aria-atomic', 'true');
        });
    }

    // Skip Links
    setupSkipLinks() {
        if (!document.querySelector('.skip-to-main')) {
            const skipLink = document.createElement('a');
            skipLink.href = '#main-content';
            skipLink.className = 'skip-to-main';
            skipLink.textContent = 'Skip to main content';
            document.body.insertBefore(skipLink, document.body.firstChild);
        }

        // Ensure main content has ID
        const main = document.querySelector('main');
        if (main && !main.id) {
            main.id = 'main-content';
        }
    }

    // Screen Reader Announcements
    setupAnnouncements() {
        // Create announcement region if it doesn't exist
        if (!document.getElementById('aria-announcements')) {
            const announcer = document.createElement('div');
            announcer.id = 'aria-announcements';
            announcer.className = 'sr-only';
            announcer.setAttribute('aria-live', 'polite');
            announcer.setAttribute('aria-atomic', 'true');
            document.body.appendChild(announcer);
        }
    }

    announce(message, priority = 'polite') {
        const announcer = document.getElementById('aria-announcements');
        if (!announcer) return;

        announcer.setAttribute('aria-live', priority);
        announcer.textContent = message;

        // Clear after announcement
        setTimeout(() => {
            announcer.textContent = '';
        }, 1000);
    }

    // Close Active Modal
    closeActiveModal() {
        const activeModal = document.querySelector('.modal.active, .overlay.active');
        if (activeModal) {
            const closeButton = activeModal.querySelector('[data-close], .modal-close');
            if (closeButton) {
                closeButton.click();
            }
        }
    }

    // Form Validation Announcements
    announceFormError(fieldName, errorMessage) {
        this.announce(`Error in ${fieldName}: ${errorMessage}`, 'assertive');
    }

    announceFormSuccess(message) {
        this.announce(message, 'polite');
    }

    // Loading State Announcements
    announceLoading(message = 'Loading...') {
        this.announce(message, 'polite');
    }

    announceLoadingComplete(message = 'Loading complete') {
        this.announce(message, 'polite');
    }
}

// Initialize accessibility manager
const accessibilityManager = new AccessibilityManager();

// Helper functions for use in other scripts
function announceToScreenReader(message, priority = 'polite') {
    accessibilityManager.announce(message, priority);
}

function announceError(fieldName, errorMessage) {
    accessibilityManager.announceFormError(fieldName, errorMessage);
}

function announceSuccess(message) {
    accessibilityManager.announceFormSuccess(message);
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AccessibilityManager,
        announceToScreenReader,
        announceError,
        announceSuccess
    };
}

// Add ARIA attributes to dynamically created elements
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
            if (node.nodeType === 1) { // Element node
                // Add ARIA labels to buttons without text
                if (node.tagName === 'BUTTON' && !node.getAttribute('aria-label') && !node.textContent.trim()) {
                    node.setAttribute('aria-label', 'Button');
                }
                
                // Add ARIA labels to links without text
                if (node.tagName === 'A' && !node.getAttribute('aria-label') && !node.textContent.trim()) {
                    node.setAttribute('aria-label', 'Link');
                }
                
                // Add role to modals
                if (node.classList.contains('modal') || node.classList.contains('overlay')) {
                    node.setAttribute('role', 'dialog');
                    node.setAttribute('aria-modal', 'true');
                }
            }
        });
    });
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});
