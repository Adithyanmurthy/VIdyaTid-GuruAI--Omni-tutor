/**
 * Dashboard - VidyaTid
 * Usage tracking and subscription management dashboard
 */

let updateInterval = null;
let countdownInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', async () => {
    await loadDashboardData();
    setupRealTimeUpdates();
    setupCountdown();
});

// Load all dashboard data
async function loadDashboardData() {
    try {
        // Load subscription data
        await loadSubscriptionData();
        
        // Load usage data
        await loadUsageData();
        
        // Load stats
        await loadStats();
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError('Failed to load dashboard data. Please refresh the page.');
    }
}

// Load subscription data
async function loadSubscriptionData() {
    try {
        const response = await fetch('/api/subscription/current', {
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error('Failed to load subscription');
        }
        
        const data = await response.json();
        updateSubscriptionUI(data);
        
    } catch (error) {
        console.error('Error loading subscription:', error);
        document.getElementById('subscription-status').textContent = 'Error';
        document.getElementById('current-plan').textContent = 'Unknown';
    }
}

// Update subscription UI
function updateSubscriptionUI(data) {
    const subscription = data.subscription || {};
    const status = subscription.status || 'free';
    const tier = subscription.tier || 'free';
    const endDate = subscription.end_date;
    const autoRenew = subscription.auto_renew !== false;
    
    // Update status badge
    const statusBadge = document.getElementById('subscription-status');
    statusBadge.textContent = status === 'active' ? 'Active' : 'Free';
    statusBadge.className = `status-badge ${status === 'active' ? 'active' : 'expired'}`;
    
    // Update plan name
    const planNames = {
        'free': 'Free',
        'starter': 'Starter',
        'premium': 'Premium',
        'ultimate': 'Ultimate'
    };
    document.getElementById('current-plan').textContent = planNames[tier] || 'Free';
    
    // Update renewal date
    const renewalDateEl = document.getElementById('renewal-date');
    if (endDate && status === 'active') {
        const date = new Date(endDate);
        renewalDateEl.textContent = date.toLocaleDateString('en-IN', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    } else {
        renewalDateEl.textContent = 'N/A';
    }
    
    // Update auto-renew
    document.getElementById('auto-renew').textContent = autoRenew ? 'Enabled' : 'Disabled';
}

// Load usage data
async function loadUsageData() {
    try {
        const response = await fetch('/api/usage/current', {
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error('Failed to load usage');
        }
        
        const data = await response.json();
        updateUsageUI(data);
        
    } catch (error) {
        console.error('Error loading usage:', error);
    }
}

// Update usage UI
function updateUsageUI(data) {
    const usage = data.usage || {};
    const queryCount = usage.query_count || 0;
    const queriesLimit = usage.queries_limit || 10;
    const predictionCount = usage.prediction_count || 0;
    const predictionsLimit = usage.predictions_limit || 0;
    
    // Update query usage
    updateQueryUsage(queryCount, queriesLimit);
    
    // Update prediction usage
    updatePredictionUsage(predictionCount, predictionsLimit);
}

// Update query usage display
function updateQueryUsage(used, limit) {
    const percentage = limit > 0 ? (used / limit) * 100 : 0;
    const remaining = Math.max(0, limit - used);
    
    // Update counts
    document.getElementById('query-count').textContent = `${used}/${limit}`;
    document.getElementById('queries-used').textContent = used;
    document.getElementById('queries-limit').textContent = limit;
    
    // Update progress bar
    const progressBar = document.getElementById('query-progress');
    progressBar.style.width = `${Math.min(percentage, 100)}%`;
    
    // Update progress bar color based on usage
    progressBar.classList.remove('warning', 'danger');
    if (percentage >= 90) {
        progressBar.classList.add('danger');
    } else if (percentage >= 80) {
        progressBar.classList.add('warning');
    }
    
    // Show/hide warning
    const warning = document.getElementById('query-warning');
    if (percentage >= 80 && percentage < 100) {
        warning.style.display = 'flex';
        warning.querySelector('span').textContent = 
            `You've used ${Math.round(percentage)}% of your daily queries`;
    } else {
        warning.style.display = 'none';
    }
}

// Update prediction usage display
function updatePredictionUsage(used, limit) {
    const percentage = limit > 0 ? (used / limit) * 100 : 0;
    const remaining = Math.max(0, limit - used);
    
    // Handle unlimited predictions
    if (limit === -1) {
        document.getElementById('prediction-count').textContent = `${used}/∞`;
        document.getElementById('predictions-used').textContent = used;
        document.getElementById('predictions-limit').textContent = '∞';
        document.getElementById('prediction-progress').style.width = '100%';
        document.getElementById('prediction-warning').style.display = 'none';
        return;
    }
    
    // Update counts
    document.getElementById('prediction-count').textContent = `${used}/${limit}`;
    document.getElementById('predictions-used').textContent = used;
    document.getElementById('predictions-limit').textContent = limit;
    
    // Update progress bar
    const progressBar = document.getElementById('prediction-progress');
    progressBar.style.width = `${Math.min(percentage, 100)}%`;
    
    // Update progress bar color based on usage
    progressBar.classList.remove('warning', 'danger');
    if (percentage >= 90) {
        progressBar.classList.add('danger');
    } else if (percentage >= 80) {
        progressBar.classList.add('warning');
    }
    
    // Show/hide warning
    const warning = document.getElementById('prediction-warning');
    if (percentage >= 80 && percentage < 100) {
        warning.style.display = 'flex';
        warning.querySelector('span').textContent = 
            `You've used ${Math.round(percentage)}% of your monthly predictions`;
    } else {
        warning.style.display = 'none';
    }
}

// Load stats
async function loadStats() {
    try {
        const response = await fetch('/api/usage/stats', {
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error('Failed to load stats');
        }
        
        const data = await response.json();
        updateStatsUI(data);
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Update stats UI
function updateStatsUI(data) {
    const stats = data.stats || {};
    
    document.getElementById('total-queries').textContent = stats.total_queries || 0;
    document.getElementById('accuracy-rate').textContent = `${stats.accuracy_rate || 0}%`;
    document.getElementById('streak-days').textContent = stats.streak_days || 0;
    document.getElementById('study-time').textContent = `${stats.study_hours || 0}h`;
}

// Setup real-time updates
function setupRealTimeUpdates() {
    // Update every 30 seconds
    updateInterval = setInterval(async () => {
        await loadUsageData();
    }, 30000);
}

// Setup countdown timer
function setupCountdown() {
    updateCountdown();
    
    // Update every second
    countdownInterval = setInterval(() => {
        updateCountdown();
    }, 1000);
}

// Update countdown timer
function updateCountdown() {
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setUTCHours(24, 0, 0, 0);
    
    const diff = tomorrow - now;
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diff % (1000 * 60)) / 1000);
    
    const timeString = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    document.getElementById('reset-time').textContent = timeString;
}

// Show error message
function showError(message) {
    // You can implement a toast notification here
    console.error(message);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    if (countdownInterval) {
        clearInterval(countdownInterval);
    }
});
