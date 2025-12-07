/**
 * Feature Gate Modal - VidyaTid
 * Reusable modal component for blocked feature access
 */

class FeatureGateModal {
    constructor() {
        this.overlay = null;
        this.modal = null;
        this.currentFeature = null;
        this.init();
    }

    init() {
        // Create modal HTML if it doesn't exist
        if (!document.getElementById('feature-gate-overlay')) {
            this.createModal();
        }
        
        this.overlay = document.getElementById('feature-gate-overlay');
        this.modal = document.getElementById('feature-gate-modal');
        
        // Setup event listeners
        this.setupEventListeners();
    }

    createModal() {
        const modalHTML = `
            <div id="feature-gate-overlay" class="feature-gate-overlay">
                <div id="feature-gate-modal" class="feature-gate-modal">
                    <div class="modal-header">
                        <button class="modal-close" onclick="featureGate.close()">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                        <div class="modal-icon" id="modal-icon">🔒</div>
                        <h2 class="modal-title" id="modal-title">Premium Feature</h2>
                        <p class="modal-subtitle" id="modal-subtitle">Upgrade to access this feature</p>
                    </div>
                    
                    <div class="modal-body">
                        <p class="feature-description" id="feature-description"></p>
                        
                        <div class="feature-benefits" id="feature-benefits" style="display: none;">
                            <h3>What you'll get:</h3>
                            <ul id="benefits-list"></ul>
                        </div>
                        
                        <div class="tier-options" id="tier-options"></div>
                    </div>
                    
                    <div class="modal-footer">
                        <button class="modal-btn modal-btn-secondary" onclick="featureGate.close()">
                            Maybe Later
                        </button>
                        <button class="modal-btn modal-btn-primary" id="upgrade-btn" onclick="featureGate.handleUpgrade()">
                            Upgrade Now
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }

    setupEventListeners() {
        // Close on overlay click
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.close();
            }
        });
        
        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.overlay.classList.contains('active')) {
                this.close();
            }
        });
    }

    show(featureConfig) {
        this.currentFeature = featureConfig;
        
        // Update modal content
        this.updateContent(featureConfig);
        
        // Show modal
        this.overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Add animation class
        this.modal.classList.add('animating');
        setTimeout(() => {
            this.modal.classList.remove('animating');
        }, 300);
    }

    updateContent(config) {
        // Update icon
        document.getElementById('modal-icon').textContent = config.icon || '🔒';
        
        // Update title
        document.getElementById('modal-title').textContent = config.title || 'Premium Feature';
        
        // Update subtitle
        document.getElementById('modal-subtitle').textContent = config.subtitle || 'Upgrade to access this feature';
        
        // Update description
        document.getElementById('feature-description').textContent = config.description || '';
        
        // Update benefits
        if (config.benefits && config.benefits.length > 0) {
            const benefitsContainer = document.getElementById('feature-benefits');
            const benefitsList = document.getElementById('benefits-list');
            
            benefitsList.innerHTML = config.benefits.map(benefit => 
                `<li>${benefit}</li>`
            ).join('');
            
            benefitsContainer.style.display = 'block';
        } else {
            document.getElementById('feature-benefits').style.display = 'none';
        }
        
        // Update tier options
        this.updateTierOptions(config.tiers || []);
    }

    updateTierOptions(tiers) {
        const container = document.getElementById('tier-options');
        
        if (tiers.length === 0) {
            container.innerHTML = '';
            return;
        }
        
        container.innerHTML = tiers.map(tier => `
            <div class="tier-option ${tier.recommended ? 'recommended' : ''}" 
                 data-tier="${tier.key}">
                <div class="tier-header">
                    <div class="tier-name">${tier.name}</div>
                    <div class="tier-price">
                        ₹${tier.price}<span>/month</span>
                    </div>
                </div>
                <div class="tier-features">${tier.features}</div>
            </div>
        `).join('');
        
        // Add click handlers
        container.querySelectorAll('.tier-option').forEach(option => {
            option.addEventListener('click', () => {
                // Remove selected class from all options
                container.querySelectorAll('.tier-option').forEach(opt => {
                    opt.style.borderColor = opt.classList.contains('recommended') ? 
                        '#4F46E5' : 'rgba(255, 255, 255, 0.1)';
                });
                
                // Add selected class to clicked option
                option.style.borderColor = '#10B981';
                
                // Update selected tier
                this.selectedTier = option.dataset.tier;
            });
        });
        
        // Pre-select recommended tier
        const recommended = container.querySelector('.tier-option.recommended');
        if (recommended) {
            this.selectedTier = recommended.dataset.tier;
        } else if (tiers.length > 0) {
            this.selectedTier = tiers[0].key;
        }
    }

    close() {
        this.overlay.classList.remove('active');
        document.body.style.overflow = '';
        this.currentFeature = null;
        this.selectedTier = null;
    }

    async handleUpgrade() {
        if (!this.selectedTier) {
            alert('Please select a plan');
            return;
        }
        
        const upgradeBtn = document.getElementById('upgrade-btn');
        upgradeBtn.classList.add('loading');
        
        try {
            // Redirect to pricing page with selected tier
            window.location.href = `/pricing?tier=${this.selectedTier}`;
        } catch (error) {
            console.error('Error handling upgrade:', error);
            alert('Failed to process upgrade. Please try again.');
            upgradeBtn.classList.remove('loading');
        }
    }
}

// Feature configurations
const FEATURE_CONFIGS = {
    diagrams: {
        icon: '📊',
        title: 'Diagrams Access',
        subtitle: 'Unlock visual learning with NCERT diagrams',
        description: 'Access thousands of high-quality diagrams from NCERT textbooks to enhance your understanding of complex concepts.',
        benefits: [
            'Complete diagram collection from all NCERT books',
            'High-resolution images for better clarity',
            'Organized by chapter and subject',
            'Download and save for offline study'
        ],
        tiers: [
            {
                key: 'starter',
                name: 'Starter',
                price: 99,
                features: 'Diagrams + 50 queries/day + Previous papers (2015-2024)',
                recommended: false
            },
            {
                key: 'premium',
                name: 'Premium',
                price: 299,
                features: 'Everything in Starter + Image solving + Mock tests + 200 queries/day',
                recommended: true
            },
            {
                key: 'ultimate',
                name: 'Ultimate',
                price: 499,
                features: 'Everything in Premium + Unlimited queries + Priority support',
                recommended: false
            }
        ]
    },
    
    image_solving: {
        icon: '📸',
        title: 'Image Doubt Solving',
        subtitle: 'Snap a photo and get instant solutions',
        description: 'Upload images of problems from your textbooks or worksheets and get detailed step-by-step solutions powered by AI.',
        benefits: [
            'Instant recognition of handwritten and printed questions',
            'Step-by-step solutions with explanations',
            'Support for all JEE/NEET subjects',
            'Save solved problems for future reference'
        ],
        tiers: [
            {
                key: 'premium',
                name: 'Premium',
                price: 299,
                features: 'Image solving + Mock tests + 200 queries/day + Advanced analytics',
                recommended: true
            },
            {
                key: 'ultimate',
                name: 'Ultimate',
                price: 499,
                features: 'Everything in Premium + Unlimited queries + Priority support',
                recommended: false
            }
        ]
    },
    
    mock_tests: {
        icon: '📝',
        title: 'Mock Tests',
        subtitle: 'Practice with JEE/NEET pattern tests',
        description: 'Take full-length mock tests designed to match the exact pattern and difficulty of JEE Main, JEE Advanced, and NEET exams.',
        benefits: [
            'Full-length tests matching exam patterns',
            'Detailed performance analysis',
            'Time management practice',
            'Identify weak areas and improve'
        ],
        tiers: [
            {
                key: 'premium',
                name: 'Premium',
                price: 299,
                features: 'Mock tests + Image solving + 200 queries/day + Advanced analytics',
                recommended: true
            },
            {
                key: 'ultimate',
                name: 'Ultimate',
                price: 499,
                features: 'Everything in Premium + Unlimited queries + Priority support',
                recommended: false
            }
        ]
    },
    
    predictions: {
        icon: '🔮',
        title: 'Question Predictions',
        subtitle: 'AI-powered exam predictions',
        description: 'Get AI-powered predictions of high-probability questions for upcoming JEE/NEET exams based on historical patterns.',
        benefits: [
            'Chapter-wise probability analysis',
            'Smart paper generation based on weak areas',
            'Complete predicted papers',
            'Confidence scores for each prediction'
        ],
        tiers: [
            {
                key: 'starter',
                name: 'Starter',
                price: 99,
                features: '2 predictions/month + Chapter analysis + 50 queries/day',
                recommended: false
            },
            {
                key: 'premium',
                name: 'Premium',
                price: 299,
                features: '10 predictions/month + Smart papers + 200 queries/day',
                recommended: true
            },
            {
                key: 'ultimate',
                name: 'Ultimate',
                price: 499,
                features: 'Unlimited predictions + Everything in Premium',
                recommended: false
            }
        ]
    },
    
    unlimited_queries: {
        icon: '∞',
        title: 'Unlimited Queries',
        subtitle: 'Ask as many questions as you need',
        description: 'Remove all daily limits and ask unlimited questions to accelerate your learning without any restrictions.',
        benefits: [
            'No daily query limits',
            'Ask questions anytime, anywhere',
            'Perfect for intensive study sessions',
            'All premium features included'
        ],
        tiers: [
            {
                key: 'ultimate',
                name: 'Ultimate',
                price: 499,
                features: 'Unlimited queries + Unlimited predictions + Priority support + All features',
                recommended: true
            }
        ]
    }
};

// Initialize global feature gate instance
const featureGate = new FeatureGateModal();

// Helper function to show feature gate
function showFeatureGate(featureName) {
    const config = FEATURE_CONFIGS[featureName];
    if (config) {
        featureGate.show(config);
    } else {
        console.error(`Feature config not found: ${featureName}`);
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FeatureGateModal, showFeatureGate, FEATURE_CONFIGS };
}
