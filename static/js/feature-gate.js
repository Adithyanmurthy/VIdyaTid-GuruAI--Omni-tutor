/**
 * Feature Gate System - Controls access to features based on subscription tier
 * Include this file on pages that need feature restrictions
 */

// Feature access configuration by tier
const FEATURE_ACCESS = {
    'free': {
        'progress_tracking': false,
        'search': true,
        'question_paper': false,
        'mock_tests': false,
        'diagrams': false,
        'image_solving': false,
        'advanced_analytics': false
    },
    'starter': {
        'progress_tracking': true,
        'search': true,
        'question_paper': true,
        'mock_tests': false,
        'diagrams': true,
        'image_solving': false,
        'advanced_analytics': false
    },
    'premium': {
        'progress_tracking': true,
        'search': true,
        'question_paper': true,
        'mock_tests': true,
        'diagrams': true,
        'image_solving': true,
        'advanced_analytics': true
    },
    'ultimate': {
        'progress_tracking': true,
        'search': true,
        'question_paper': true,
        'mock_tests': true,
        'diagrams': true,
        'image_solving': true,
        'advanced_analytics': true
    }
};

// Get user's current subscription tier
async function getUserTier() {
    if (!window.authCheck || !window.authCheck.isAuthenticated()) {
        return 'free';
    }
    
    try {
        const sessionToken = localStorage.getItem('session_token');
        const response = await fetch('/api/subscription/current', {
            headers: {
                'Authorization': `Bearer ${sessionToken}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            return data.subscription ? data.subscription.tier : 'free';
        }
    } catch (error) {
        console.error('Error fetching subscription:', error);
    }
    
    return 'free';
}

// Check if user can access a feature
async function canAccessFeature(featureName) {
    const tier = await getUserTier();
    return FEATURE_ACCESS[tier] && FEATURE_ACCESS[tier][featureName];
}

// Show upgrade modal
function showUpgradeModal(featureName) {
    const modal = document.createElement('div');
    modal.className = 'upgrade-modal-overlay';
    modal.innerHTML = `
        <div class="upgrade-modal">
            <div class="upgrade-modal-header">
                <h2>🔒 Premium Feature</h2>
                <button class="close-modal" onclick="this.closest('.upgrade-modal-overlay').remove()">×</button>
            </div>
            <div class="upgrade-modal-body">
                <p>This feature is available in our paid plans.</p>
                <div class="upgrade-options">
                    <div class="upgrade-option">
                        <h3>Starter</h3>
                        <p class="price">₹99/month</p>
                        <button class="btn-primary" onclick="window.location.href='/pricing?tier=starter&price=99'">
                            Upgrade Now
                        </button>
                    </div>
                    <div class="upgrade-option featured">
                        <span class="badge">Most Popular</span>
                        <h3>Premium</h3>
                        <p class="price">₹299/month</p>
                        <button class="btn-primary" onclick="window.location.href='/pricing?tier=premium&price=299'">
                            Upgrade Now
                        </button>
                    </div>
                    <div class="upgrade-option">
                        <h3>Ultimate</h3>
                        <p class="price">₹499/month</p>
                        <button class="btn-primary" onclick="window.location.href='/pricing?tier=ultimate&price=499'">
                            Upgrade Now
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        .upgrade-modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        }
        .upgrade-modal {
            background: white;
            border-radius: 12px;
            max-width: 800px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
        }
        .upgrade-modal-header {
            padding: 20px;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .upgrade-modal-header h2 {
            margin: 0;
            font-size: 24px;
            color: #1f2937;
        }
        .close-modal {
            background: none;
            border: none;
            font-size: 32px;
            cursor: pointer;
            color: #6b7280;
        }
        .close-modal:hover {
            color: #374151;
        }
        .upgrade-modal-body {
            padding: 30px;
        }
        .upgrade-modal-body > p {
            color: #4b5563;
            font-size: 16px;
            margin-bottom: 10px;
        }
        .upgrade-options {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .upgrade-option {
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        .upgrade-option.featured {
            border-color: #667eea;
            transform: scale(1.05);
        }
        .upgrade-option .badge {
            position: absolute;
            top: -10px;
            left: 50%;
            transform: translateX(-50%);
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
        }
        .upgrade-option h3 {
            margin: 10px 0;
            color: #1f2937;
            font-size: 20px;
            font-weight: 600;
        }
        .upgrade-option .price {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .upgrade-option .btn-primary {
            width: 100%;
            padding: 10px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
        }
        .upgrade-option .btn-primary:hover {
            background: #5568d3;
        }
    `;
    document.head.appendChild(style);
    document.body.appendChild(modal);
}

// Protect page - check if user can access current page feature
async function protectPage(featureName) {
    const hasAccess = await canAccessFeature(featureName);
    
    if (!hasAccess) {
        // Show upgrade modal
        showUpgradeModal(featureName);
        
        // Optionally redirect after a delay
        setTimeout(() => {
            window.location.href = '/pricing';
        }, 5000);
        
        return false;
    }
    
    return true;
}

// Export functions
window.featureGate = {
    canAccessFeature,
    showUpgradeModal,
    protectPage,
    getUserTier
};
