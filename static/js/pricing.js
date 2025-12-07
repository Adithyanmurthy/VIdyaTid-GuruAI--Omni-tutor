/**
 * Pricing Page - VidyaTid
 * Subscription Pricing System: FREE, Starter (₹99), Premium (₹299), Ultimate (₹499)
 */

// Tier configurations based on design document
const TIER_CONFIG = {
    free: {
        name: 'Free',
        price_monthly: 0,
        price_yearly: 0,
        queries_per_day: 10,
        features: [
            { text: '10 queries per day', included: true },
            { text: 'Basic NCERT content', included: true },
            { text: 'AI explanations', included: true },
            { text: 'Quiz generation (2 questions)', included: true },
            { text: 'Diagrams access', included: false },
            { text: 'Previous year papers', included: false },
            { text: 'Image doubt solving', included: false },
            { text: 'Progress tracking', included: false }
        ],
        badge: '🎁 Always Free',
        description: 'Perfect for getting started'
    },
    starter: {
        name: 'Starter',
        price_monthly: 99,
        price_yearly: 990,  // 17% discount
        queries_per_day: 50,
        features: [
            { text: '50 queries per day', included: true },
            { text: 'Full NCERT content', included: true },
            { text: 'Diagrams access', included: true, highlight: true },
            { text: 'Previous papers (2015-2024)', included: true },
            { text: 'Quiz generation (5 questions)', included: true },
            { text: 'Basic progress tracking', included: true },
            { text: 'Chapter analysis', included: true },
            { text: '2 predictions per month', included: true }
        ],
        badge: '📚 Entry Level',
        description: 'Great for regular practice'
    },
    premium: {
        name: 'Premium',
        price_monthly: 299,
        price_yearly: 2990,  // 17% discount
        queries_per_day: 200,
        features: [
            { text: '200 queries per day', included: true },
            { text: 'Everything in Starter', included: true },
            { text: 'Image doubt solving', included: true, highlight: true },
            { text: 'Mock tests (JEE/NEET)', included: true, highlight: true },
            { text: 'Previous papers (2010-2024)', included: true },
            { text: 'Advanced analytics', included: true },
            { text: 'Smart paper generation', included: true },
            { text: '10 predictions per month', included: true }
        ],
        badge: '⭐ Most Popular',
        description: 'Best for serious aspirants',
        popular: true
    },
    ultimate: {
        name: 'Ultimate',
        price_monthly: 499,
        price_yearly: 4990,  // 17% discount
        queries_per_day: -1,  // Unlimited
        features: [
            { text: 'Unlimited queries', included: true, highlight: true },
            { text: 'Everything in Premium', included: true },
            { text: 'Previous papers (2005-2024)', included: true },
            { text: 'Unlimited predictions', included: true, highlight: true },
            { text: 'Priority support', included: true },
            { text: 'Personalized study plans', included: true },
            { text: 'Progress analytics', included: true },
            { text: 'Early access to features', included: true }
        ],
        badge: '👑 Ultimate',
        description: 'For top performers'
    }
};

// Current user state
let currentUser = null;
let currentTier = 'free';
let billingDuration = 'monthly';

// Initialize pricing page
document.addEventListener('DOMContentLoaded', async () => {
    await loadUserData();
    renderPricingCards();
    setupBillingToggle();
    setupUserProfile();
});

// Load user data from API
async function loadUserData() {
    try {
        const sessionToken = localStorage.getItem('session_token');
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (sessionToken) {
            headers['Authorization'] = `Bearer ${sessionToken}`;
        }
        
        const response = await fetch('/api/subscription/current', {
            credentials: 'include',
            headers: headers
        });
        
        if (response.ok) {
            const data = await response.json();
            currentUser = data.user;
            currentTier = data.subscription?.tier || 'free';
        }
    } catch (error) {
        console.log('User not logged in or error loading subscription');
    }
}

// Render pricing cards
function renderPricingCards() {
    const container = document.getElementById('pricingCards');
    if (!container) return;
    
    container.innerHTML = '';
    
    Object.keys(TIER_CONFIG).forEach(tierKey => {
        const tier = TIER_CONFIG[tierKey];
        const card = createPricingCard(tierKey, tier);
        container.appendChild(card);
    });
}

// Create individual pricing card
function createPricingCard(tierKey, tier) {
    const card = document.createElement('div');
    card.className = 'pricing-card';
    
    // Add special classes
    if (tier.popular) {
        card.classList.add('popular');
    }
    if (tierKey === currentTier) {
        card.classList.add('current-plan');
    }
    
    // Calculate price based on billing duration
    const price = billingDuration === 'monthly' ? tier.price_monthly : tier.price_yearly;
    const period = billingDuration === 'monthly' ? 'month' : 'year';
    const savings = billingDuration === 'yearly' && price > 0 ? 
        `Save ₹${(tier.price_monthly * 12 - tier.price_yearly)}` : '';
    
    // Build card HTML
    card.innerHTML = `
        ${tier.popular ? '<div class="popular-badge">Most Popular</div>' : ''}
        
        <div class="plan-name">${tier.name}</div>
        
        <div class="plan-price">
            <span class="price-currency">₹</span>
            <span class="price-amount">${price}</span>
            <span class="price-period">/${period}</span>
        </div>
        
        ${savings ? `<div class="price-savings">${savings}</div>` : ''}
        
        <p class="plan-description">${tier.description}</p>
        
        <ul class="plan-features">
            ${tier.features.map(feature => `
                <li class="${feature.highlight ? 'highlight' : ''}">
                    <span class="feature-icon">${feature.included ? '✓' : '✗'}</span>
                    <span class="feature-text">${feature.text}</span>
                </li>
            `).join('')}
        </ul>
        
        <button class="plan-button ${getButtonClass(tierKey)}" 
                onclick="handlePlanAction('${tierKey}')"
                ${tierKey === currentTier ? 'disabled' : ''}>
            ${getButtonText(tierKey)}
        </button>
    `;
    
    return card;
}

// Get button class based on tier and current plan
function getButtonClass(tierKey) {
    if (tierKey === currentTier) {
        return 'current';
    }
    if (TIER_CONFIG[tierKey].popular) {
        return 'primary';
    }
    return 'secondary';
}

// Get button text based on tier and current plan
function getButtonText(tierKey) {
    if (tierKey === currentTier) {
        return 'Current Plan';
    }
    if (tierKey === 'free') {
        return 'Start Free';
    }
    
    const tierOrder = ['free', 'starter', 'premium', 'ultimate'];
    const currentIndex = tierOrder.indexOf(currentTier);
    const targetIndex = tierOrder.indexOf(tierKey);
    
    if (targetIndex > currentIndex) {
        return 'Upgrade Now';
    } else {
        return 'Downgrade';
    }
}

// Handle plan action (subscribe, upgrade, downgrade)
async function handlePlanAction(tierKey) {
    // Check if user is logged in using localStorage
    const sessionToken = localStorage.getItem('session_token');
    const userData = localStorage.getItem('user_data');
    
    if (!sessionToken || !userData) {
        // User not logged in - redirect to login
        sessionStorage.setItem('redirect_after_login', `/pricing?tier=${tierKey}`);
        window.location.href = '/auth';
        return;
    }
    
    if (tierKey === 'free') {
        alert('You are already on the free plan or can downgrade from your subscription settings.');
        return;
    }
    
    if (tierKey === currentTier) {
        return;
    }
    
    const tierOrder = ['free', 'starter', 'premium', 'ultimate'];
    const currentIndex = tierOrder.indexOf(currentTier);
    const targetIndex = tierOrder.indexOf(tierKey);
    
    if (targetIndex > currentIndex) {
        // Upgrade
        await initiateUpgrade(tierKey);
    } else {
        // Downgrade
        await initiateDowngrade(tierKey);
    }
}

// Initiate upgrade process
async function initiateUpgrade(tierKey) {
    try {
        const tier = TIER_CONFIG[tierKey];
        const price = billingDuration === 'monthly' ? tier.price_monthly : tier.price_yearly;
        const duration = billingDuration === 'monthly' ? 30 : 365;
        
        const sessionToken = localStorage.getItem('session_token');
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (sessionToken) {
            headers['Authorization'] = `Bearer ${sessionToken}`;
        }
        
        // Create order
        const response = await fetch('/api/payment/order/create', {
            method: 'POST',
            headers: headers,
            credentials: 'include',
            body: JSON.stringify({
                tier: tierKey,
                duration: duration,
                amount: price * 100  // Convert to paise
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to create order');
        }
        
        const orderData = await response.json();
        
        // Open Razorpay checkout
        const options = {
            key: orderData.razorpay_key_id,
            amount: orderData.amount,
            currency: 'INR',
            name: 'VidyaTid',
            description: `${tier.name} Plan - ${billingDuration}`,
            order_id: orderData.order_id,
            handler: function(response) {
                handlePaymentSuccess(response, tierKey);
            },
            prefill: {
                email: currentUser.email || '',
                contact: currentUser.phone || ''
            },
            theme: {
                color: '#4F46E5'
            }
        };
        
        const rzp = new Razorpay(options);
        rzp.open();
        
    } catch (error) {
        console.error('Error initiating upgrade:', error);
        alert('Failed to initiate upgrade. Please try again.');
    }
}

// Handle successful payment
async function handlePaymentSuccess(response, tierKey) {
    try {
        const sessionToken = localStorage.getItem('session_token');
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (sessionToken) {
            headers['Authorization'] = `Bearer ${sessionToken}`;
        }
        
        // Verify payment on backend
        const verifyResponse = await fetch('/api/payment/verify', {
            method: 'POST',
            headers: headers,
            credentials: 'include',
            body: JSON.stringify({
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_order_id: response.razorpay_order_id,
                razorpay_signature: response.razorpay_signature,
                tier: tierKey
            })
        });
        
        if (verifyResponse.ok) {
            alert('Payment successful! Your subscription has been upgraded.');
            window.location.reload();
        } else {
            throw new Error('Payment verification failed');
        }
    } catch (error) {
        console.error('Error verifying payment:', error);
        alert('Payment completed but verification failed. Please contact support.');
    }
}

// Initiate downgrade process
async function initiateDowngrade(tierKey) {
    const confirmed = confirm(
        `Are you sure you want to downgrade to ${TIER_CONFIG[tierKey].name}? ` +
        `The downgrade will take effect at the end of your current billing cycle.`
    );
    
    if (!confirmed) return;
    
    try {
        const sessionToken = localStorage.getItem('session_token');
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (sessionToken) {
            headers['Authorization'] = `Bearer ${sessionToken}`;
        }
        
        const response = await fetch('/api/subscription/downgrade', {
            method: 'POST',
            headers: headers,
            credentials: 'include',
            body: JSON.stringify({
                new_tier: tierKey
            })
        });
        
        if (response.ok) {
            alert('Downgrade scheduled successfully. You will keep your current benefits until the end of your billing cycle.');
            window.location.reload();
        } else {
            throw new Error('Failed to schedule downgrade');
        }
    } catch (error) {
        console.error('Error scheduling downgrade:', error);
        alert('Failed to schedule downgrade. Please try again.');
    }
}

// Setup billing toggle
function setupBillingToggle() {
    const toggleButtons = document.querySelectorAll('.toggle-btn');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            toggleButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Update billing duration
            billingDuration = button.dataset.duration;
            
            // Re-render pricing cards
            renderPricingCards();
        });
    });
}

// Setup user profile link
function setupUserProfile() {
    const profileLink = document.getElementById('user-profile-link');
    if (!profileLink) return;
    
    if (currentUser) {
        profileLink.textContent = currentUser.username || 'Profile';
        profileLink.href = '/settings';
    } else {
        profileLink.textContent = 'Login';
        profileLink.href = '/auth';
    }
}
