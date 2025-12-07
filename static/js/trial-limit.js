/**
 * Trial Limit Manager - Freemium Model
 * Allows 3 free questions before requiring login/subscription
 */

class TrialLimitManager {
    constructor() {
        this.maxTrialQuestions = 3; // Before login
        this.maxFreeQuestions = 10; // After login (free tier)
        this.storageKey = 'guruai_trial_count';
        this.dailyCountKey = 'guruai_daily_count';
        this.dailyDateKey = 'guruai_daily_date';
        this.init();
    }

    init() {
        // Check if user is authenticated
        this.checkAuthStatus();
    }

    async checkAuthStatus() {
        try {
            const response = await fetch('/api/auth/check', {
                credentials: 'include'
            });
            const data = await response.json();
            this.isAuthenticated = data.authenticated;
            
            // If not authenticated, track trial usage
            if (!this.isAuthenticated) {
                this.loadTrialCount();
            }
        } catch (error) {
            console.error('Auth check failed:', error);
            this.isAuthenticated = false;
            this.loadTrialCount();
        }
    }

    loadTrialCount() {
        const stored = localStorage.getItem(this.storageKey);
        this.trialCount = stored ? parseInt(stored) : 0;
        this.updateTrialDisplay();
    }

    incrementTrialCount() {
        if (!this.isAuthenticated) {
            // Not logged in - increment trial count
            this.trialCount++;
            localStorage.setItem(this.storageKey, this.trialCount.toString());
            this.updateTrialDisplay();
            
            // Check if limit reached
            if (this.trialCount >= this.maxTrialQuestions) {
                this.showUpgradeModal('trial');
                return false; // Block further questions
            }
        } else if (this.userPlan === 'free') {
            // Free tier - increment daily count
            this.incrementDailyCount();
            this.updateTrialDisplay();
            
            // Check if daily limit reached
            if (this.getDailyCount() >= this.maxFreeQuestions) {
                this.showUpgradeModal('free');
                return false; // Block further questions
            }
        }
        return true; // Allow question
    }

    canAskQuestion() {
        if (this.isAuthenticated) {
            return true; // Authenticated users have unlimited access
        }
        return this.trialCount < this.maxTrialQuestions;
    }

    getRemainingTrials() {
        if (this.isAuthenticated) {
            return Infinity;
        }
        return Math.max(0, this.maxTrialQuestions - this.trialCount);
    }

    updateTrialDisplay() {
        const remaining = this.getRemainingTrials();
        
        // Update trial counter in UI
        let trialBadge = document.getElementById('trial-badge');
        
        // Show badge for non-authenticated or free tier users
        const shouldShowBadge = !this.isAuthenticated || (this.isAuthenticated && this.userPlan === 'free');
        
        if (!trialBadge && shouldShowBadge) {
            // Create trial badge
            trialBadge = document.createElement('div');
            trialBadge.id = 'trial-badge';
            trialBadge.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 25px;
                font-weight: 600;
                font-size: 0.9rem;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                z-index: 1000;
                animation: slideIn 0.3s ease;
            `;
            document.body.appendChild(trialBadge);
        }
        
        if (trialBadge) {
            if (remaining === Infinity || !shouldShowBadge) {
                trialBadge.style.display = 'none';
            } else {
                trialBadge.style.display = 'block';
                
                if (!this.isAuthenticated) {
                    // Trial user
                    trialBadge.innerHTML = `
                        <span style="font-size: 1.1rem;">🎁</span>
                        ${remaining} free ${remaining === 1 ? 'question' : 'questions'} left
                    `;
                } else if (this.userPlan === 'free') {
                    // Free tier user
                    trialBadge.innerHTML = `
                        <span style="font-size: 1.1rem;">📚</span>
                        ${remaining}/${this.maxFreeQuestions} questions today
                    `;
                }
                
                // Change color when running low
                if (remaining <= 2) {
                    trialBadge.style.background = 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)';
                }
            }
        }
    }

    showUpgradeModal(type = 'trial') {
        // Create modal overlay
        const modal = document.createElement('div');
        modal.id = 'upgrade-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            animation: fadeIn 0.3s ease;
        `;

        const content = type === 'trial' ? {
            icon: '🎓',
            title: 'Free Trial Complete!',
            message: `You've used all <strong>3 free questions</strong>.<br>Sign up for free to get <strong>10 questions daily</strong>, or upgrade for unlimited access!`,
            features: `
                ✨ 10 questions/day (Free)<br>
                📚 Full NCERT coverage<br>
                🎯 AI-powered quizzes<br>
                📊 Progress tracking
            `,
            primaryBtn: 'Sign Up Free',
            primaryAction: '/auth'
        } : {
            icon: '📚',
            title: 'Daily Limit Reached!',
            message: `You've used all <strong>10 free questions</strong> today.<br>Upgrade to get unlimited questions and premium features!`,
            features: `
                ∞ Unlimited questions<br>
                📝 Previous year papers<br>
                🖼️ Image doubt solving<br>
                🎯 Mock tests & more
            `,
            primaryBtn: 'View Plans 🚀',
            primaryAction: '/pricing'
        };

        modal.innerHTML = `
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                padding: 40px;
                max-width: 500px;
                text-align: center;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                animation: slideUp 0.4s ease;
            ">
                <div style="font-size: 4rem; margin-bottom: 20px;">${content.icon}</div>
                <h2 style="color: white; font-size: 2rem; margin-bottom: 15px; font-weight: 700;">
                    ${content.title}
                </h2>
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin-bottom: 30px; line-height: 1.6;">
                    ${content.message}
                </p>
                
                <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 30px;">
                    <p style="color: white; font-size: 0.95rem; margin: 0; line-height: 1.8;">
                        ${content.features}
                    </p>
                </div>

                <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
                    <button id="primary-action-btn" style="
                        background: white;
                        color: #667eea;
                        border: none;
                        padding: 15px 35px;
                        border-radius: 25px;
                        font-weight: 700;
                        font-size: 1.1rem;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    ">
                        ${content.primaryBtn}
                    </button>
                    <button id="close-modal-btn" style="
                        background: rgba(255, 255, 255, 0.2);
                        color: white;
                        border: 2px solid white;
                        padding: 15px 35px;
                        border-radius: 25px;
                        font-weight: 600;
                        font-size: 1.1rem;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    ">
                        Maybe Later
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Add hover effects
        const primaryBtn = document.getElementById('primary-action-btn');
        const closeModalBtn = document.getElementById('close-modal-btn');

        primaryBtn.onmouseover = () => {
            primaryBtn.style.transform = 'translateY(-2px)';
            primaryBtn.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.3)';
        };
        primaryBtn.onmouseout = () => {
            primaryBtn.style.transform = 'translateY(0)';
            primaryBtn.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.2)';
        };

        closeModalBtn.onmouseover = () => {
            closeModalBtn.style.background = 'rgba(255, 255, 255, 0.3)';
        };
        closeModalBtn.onmouseout = () => {
            closeModalBtn.style.background = 'rgba(255, 255, 255, 0.2)';
        };

        // Event listeners
        primaryBtn.onclick = () => {
            window.location.href = content.primaryAction;
        };

        closeModalBtn.onclick = () => {
            modal.remove();
        };

        // Close on outside click
        modal.onclick = (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        };
    }

    resetTrial() {
        // For testing purposes
        localStorage.removeItem(this.storageKey);
        this.trialCount = 0;
        this.updateTrialDisplay();
    }
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
`;
document.head.appendChild(style);

// Initialize trial manager
const trialManager = new TrialLimitManager();

// Export for use in other scripts
window.trialManager = trialManager;
