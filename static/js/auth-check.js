/**
 * Common authentication check and navigation update
 * Include this file in all pages that need auth state
 */

// Check if user is authenticated
function isAuthenticated() {
    const sessionToken = localStorage.getItem('session_token');
    const userData = localStorage.getItem('user_data');
    return !!(sessionToken && userData);
}

// Get current user data
function getCurrentUser() {
    const userData = localStorage.getItem('user_data');
    if (userData) {
        try {
            return JSON.parse(userData);
        } catch (e) {
            console.error('Error parsing user data:', e);
            return null;
        }
    }
    return null;
}

// Logout function
function logout() {
    // Clear local storage
    localStorage.removeItem('session_token');
    localStorage.removeItem('user_data');
    
    // Redirect to home page
    window.location.href = '/';
}

// Update navigation based on auth state
function updateAuthNavigation() {
    const navLinks = document.querySelector('.nav-links');
    if (!navLinks) return;
    
    if (isAuthenticated()) {
        const user = getCurrentUser();
        if (!user) {
            // Invalid user data, clear and show login
            logout();
            return;
        }
        
        // Find or create auth-related links
        let authLink = navLinks.querySelector('#auth-link');
        let ctaLink = navLinks.querySelector('#cta-link');
        let logoutBtn = navLinks.querySelector('#logout-btn');
        
        // Update or create auth link - Show "Dashboard" instead of username
        if (authLink) {
            authLink.textContent = 'Dashboard';
            authLink.href = '/chat';
            authLink.title = 'Go to Dashboard';
        } else {
            // Find login link and update it
            const loginLink = Array.from(navLinks.querySelectorAll('a')).find(a => 
                a.textContent.toLowerCase().includes('login') || a.href.includes('/auth')
            );
            if (loginLink) {
                loginLink.id = 'auth-link';
                loginLink.textContent = 'Dashboard';
                loginLink.href = '/chat';
                loginLink.title = 'Go to Dashboard';
            }
        }
        
        // Update CTA link - Disabled to avoid duplicate with profile dropdown
        // if (ctaLink) {
        //     ctaLink.textContent = 'Dashboard';
        //     ctaLink.href = '/chat';
        // }
        
        // Add logout button if it doesn't exist
        if (!logoutBtn) {
            logoutBtn = document.createElement('a');
            logoutBtn.id = 'logout-btn';
            logoutBtn.href = '#';
            logoutBtn.textContent = 'Logout';
            logoutBtn.className = 'btn-secondary';
            logoutBtn.style.marginLeft = '10px';
            logoutBtn.onclick = function(e) {
                e.preventDefault();
                logout();
            };
            navLinks.appendChild(logoutBtn);
        }
    } else {
        // User not logged in - ensure login links are shown
        let authLink = navLinks.querySelector('#auth-link');
        let ctaLink = navLinks.querySelector('#cta-link');
        let logoutBtn = navLinks.querySelector('#logout-btn');
        
        if (authLink) {
            authLink.textContent = 'Login';
            authLink.href = '/auth';
            authLink.title = '';
        }
        
        if (ctaLink) {
            ctaLink.textContent = 'Try Free';
            ctaLink.href = '/chat';
        }
        
        // Remove logout button
        if (logoutBtn) {
            logoutBtn.remove();
        }
    }
}

// Protect page - redirect to auth if not logged in
function requireAuth() {
    if (!isAuthenticated()) {
        // Store current page to redirect back after login
        sessionStorage.setItem('redirect_after_login', window.location.pathname);
        window.location.href = '/auth';
        return false;
    }
    return true;
}

// Get user's current subscription tier
async function getCurrentSubscription() {
    if (!isAuthenticated()) {
        return null;
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
    
    return 'free'; // Default to free if error
}

// Update pricing cards based on user's subscription
async function updatePricingCards() {
    if (!isAuthenticated()) {
        return; // Not logged in, show default subscribe buttons
    }
    
    const currentTier = await getCurrentSubscription();
    
    // Show current plan banner
    const banner = document.getElementById('current-plan-banner');
    const planName = document.getElementById('current-plan-name');
    const planDetails = document.getElementById('current-plan-details');
    
    if (banner && planName && planDetails) {
        const tierNames = {
            'free': 'Free',
            'starter': 'Starter',
            'premium': 'Premium',
            'ultimate': 'Ultimate'
        };
        
        const tierDetails = {
            'free': '10 queries per day',
            'starter': '50 queries per day',
            'premium': '200 queries per day',
            'ultimate': 'Unlimited queries'
        };
        
        planName.textContent = tierNames[currentTier] || 'Free';
        planDetails.textContent = tierDetails[currentTier] || '10 queries per day';
        banner.style.display = 'block';
    }
    
    // Update pricing cards
    const pricingCards = document.querySelectorAll('.pricing-card');
    
    pricingCards.forEach(card => {
        const cardTier = card.getAttribute('data-tier');
        const button = card.querySelector('.subscribe-btn, .btn-outline, .btn-primary');
        
        if (!button || !cardTier) return;
        
        if (cardTier === currentTier) {
            // This is the user's current plan
            button.textContent = 'Current Plan';
            button.disabled = true;
            button.style.opacity = '0.6';
            button.style.cursor = 'not-allowed';
            card.style.border = '2px solid #667eea';
        } else if (button.classList.contains('subscribe-btn')) {
            // Other paid tiers - keep upgrade button
            button.textContent = 'Upgrade';
        }
    });
}

// Handle subscription button clicks
function handleSubscribe(tier, price) {
    if (isAuthenticated()) {
        // User is logged in - go to payment page
        window.location.href = `/pricing?tier=${tier}&price=${price}`;
    } else {
        // User not logged in - save intended action and go to login
        sessionStorage.setItem('redirect_after_login', `/pricing?tier=${tier}&price=${price}`);
        sessionStorage.setItem('subscription_intent', JSON.stringify({ tier, price }));
        window.location.href = '/auth';
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    updateAuthNavigation();
    updatePricingCards(); // Update pricing cards based on subscription
    
    // Add click handlers to all subscribe buttons
    const subscribeButtons = document.querySelectorAll('.subscribe-btn');
    subscribeButtons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.disabled) return; // Don't handle if it's current plan
            
            const tier = this.getAttribute('data-tier');
            const price = this.getAttribute('data-price');
            handleSubscribe(tier, price);
        });
    });
});

// Export functions for use in other scripts
window.authCheck = {
    isAuthenticated,
    getCurrentUser,
    logout,
    updateAuthNavigation,
    requireAuth,
    checkAuth: isAuthenticated, // Alias for compatibility
    getCurrentSubscription
};
