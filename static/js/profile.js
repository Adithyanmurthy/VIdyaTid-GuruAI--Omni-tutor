// Profile Page JavaScript
document.addEventListener('DOMContentLoaded', async function() {
    // Check authentication
    if (!window.authCheck || !window.authCheck.isAuthenticated()) {
        window.location.href = '/auth';
        return;
    }

    // Load user data
    await loadUserData();
    await loadSubscriptionData();

    // Setup event listeners
    setupEventListeners();
});

// Load user data
async function loadUserData() {
    try {
        // Get user data from localStorage
        let userData = null;
        try {
            userData = JSON.parse(localStorage.getItem('user_data'));
        } catch (e) {
            console.error('Error parsing user_data from localStorage:', e);
        }

        if (userData) {
            // Display user information
            document.getElementById('display-username').textContent = userData.username || 'Not set';
            document.getElementById('display-email').textContent = userData.email || 'Not set';
            document.getElementById('display-fullname').textContent = userData.full_name || 'Not set';
            document.getElementById('display-phone').textContent = userData.phone || 'Not set';

            // Set edit fields
            document.getElementById('edit-username').value = userData.username || '';
            document.getElementById('edit-email').value = userData.email || '';
            document.getElementById('edit-fullname').value = userData.full_name || '';
            document.getElementById('edit-phone').value = userData.phone || '';

            // Set member since date
            if (userData.created_at) {
                const memberDate = new Date(userData.created_at);
                document.getElementById('member-since').textContent = memberDate.toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                });
            } else {
                document.getElementById('member-since').textContent = 'Recently';
            }
        } else {
            showNotification('No user data found. Please log in again.', 'error');
            setTimeout(() => {
                window.location.href = '/auth';
            }, 2000);
        }
    } catch (error) {
        console.error('Error loading user data:', error);
        showNotification('Failed to load user data', 'error');
    }
}

// Load subscription data
async function loadSubscriptionData() {
    try {
        const sessionToken = localStorage.getItem('session_token');
        const response = await fetch('/api/subscription/current', {
            headers: {
                'Authorization': `Bearer ${sessionToken}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            const subscription = data.subscription;

            if (subscription) {
                // Update subscription tier
                const tierNames = {
                    'free': 'Free Plan',
                    'starter': 'Starter Plan',
                    'premium': 'Premium Plan',
                    'ultimate': 'Ultimate Plan'
                };
                document.getElementById('subscription-tier').textContent = tierNames[subscription.tier] || 'Free Plan';

                // Update subscription status
                const statusBadge = document.getElementById('subscription-status');
                if (subscription.is_active) {
                    statusBadge.textContent = 'Active';
                    statusBadge.style.background = 'rgba(34, 197, 94, 0.3)';
                } else {
                    statusBadge.textContent = 'Inactive';
                    statusBadge.style.background = 'rgba(239, 68, 68, 0.3)';
                }

                // Update queries limit
                const queryLimits = {
                    'free': '10',
                    'starter': '50',
                    'premium': '200',
                    'ultimate': 'Unlimited'
                };
                document.getElementById('subscription-queries').textContent = queryLimits[subscription.tier] || '10';

                // Show end date if applicable
                if (subscription.end_date && subscription.tier !== 'free') {
                    const endDate = new Date(subscription.end_date);
                    document.getElementById('subscription-end-date').textContent = endDate.toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric'
                    });
                    document.getElementById('subscription-end-date-item').classList.remove('hidden');
                }
            }

            // Load usage data
            await loadUsageData();
        }
    } catch (error) {
        console.error('Error loading subscription data:', error);
    }
}

// Load usage data
async function loadUsageData() {
    try {
        const sessionToken = localStorage.getItem('session_token');
        const response = await fetch('/api/usage/current', {
            headers: {
                'Authorization': `Bearer ${sessionToken}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            if (data.usage) {
                const used = data.usage.queries_used || 0;
                const limit = data.usage.queries_limit || 10;
                document.getElementById('queries-used').textContent = `${used} / ${limit}`;
            }
        }
    } catch (error) {
        console.error('Error loading usage data:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Edit account button
    document.getElementById('edit-account-btn').addEventListener('click', toggleEditMode);
    document.getElementById('save-account-btn').addEventListener('click', saveAccountChanges);
    document.getElementById('cancel-account-btn').addEventListener('click', cancelEditMode);

    // Password form
    document.getElementById('password-form').addEventListener('submit', handlePasswordChange);

    // Preferences toggles
    document.getElementById('email-notifications').addEventListener('change', savePreference);
    document.getElementById('study-reminders').addEventListener('change', savePreference);
    document.getElementById('dark-mode').addEventListener('change', toggleDarkMode);

    // Delete account
    document.getElementById('delete-account-btn').addEventListener('click', confirmDeleteAccount);

    // Modal buttons
    document.getElementById('modal-cancel').addEventListener('click', closeModal);
}

// Toggle edit mode
function toggleEditMode() {
    // Hide display fields
    document.querySelectorAll('.info-item p').forEach(p => p.classList.add('hidden'));
    
    // Show edit fields
    document.querySelectorAll('.info-item input').forEach(input => input.classList.remove('hidden'));
    
    // Show action buttons
    document.getElementById('account-actions').classList.remove('hidden');
    
    // Hide edit button
    document.getElementById('edit-account-btn').classList.add('hidden');
}

// Cancel edit mode
function cancelEditMode() {
    // Show display fields
    document.querySelectorAll('.info-item p').forEach(p => p.classList.remove('hidden'));
    
    // Hide edit fields
    document.querySelectorAll('.info-item input').forEach(input => input.classList.add('hidden'));
    
    // Hide action buttons
    document.getElementById('account-actions').classList.add('hidden');
    
    // Show edit button
    document.getElementById('edit-account-btn').classList.remove('hidden');
}

// Save account changes
async function saveAccountChanges() {
    try {
        const sessionToken = localStorage.getItem('session_token');
        const userData = {
            username: document.getElementById('edit-username').value,
            email: document.getElementById('edit-email').value,
            full_name: document.getElementById('edit-fullname').value,
            phone: document.getElementById('edit-phone').value
        };

        const response = await fetch('/api/user/update', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify(userData)
        });

        if (response.ok) {
            const data = await response.json();
            
            // Update local storage
            localStorage.setItem('user_data', JSON.stringify(data.user));
            
            // Update display
            await loadUserData();
            cancelEditMode();
            
            showNotification('Profile updated successfully', 'success');
        } else {
            const error = await response.json();
            showNotification(error.message || 'Failed to update profile', 'error');
        }
    } catch (error) {
        console.error('Error saving account changes:', error);
        showNotification('Failed to update profile', 'error');
    }
}

// Handle password change
async function handlePasswordChange(e) {
    e.preventDefault();

    const currentPassword = document.getElementById('current-password').value;
    const newPassword = document.getElementById('new-password').value;
    const confirmPassword = document.getElementById('confirm-password').value;

    // Validate passwords
    if (newPassword !== confirmPassword) {
        showNotification('New passwords do not match', 'error');
        return;
    }

    if (newPassword.length < 8) {
        showNotification('Password must be at least 8 characters long', 'error');
        return;
    }

    try {
        const sessionToken = localStorage.getItem('session_token');
        const response = await fetch('/api/user/change-password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                current_password: currentPassword,
                new_password: newPassword
            })
        });

        if (response.ok) {
            showNotification('Password changed successfully', 'success');
            document.getElementById('password-form').reset();
        } else {
            const error = await response.json();
            showNotification(error.message || 'Failed to change password', 'error');
        }
    } catch (error) {
        console.error('Error changing password:', error);
        showNotification('Failed to change password', 'error');
    }
}

// Save preference
async function savePreference(e) {
    const preference = e.target.id;
    const value = e.target.checked;

    try {
        const sessionToken = localStorage.getItem('session_token');
        const response = await fetch('/api/user/preferences', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                [preference]: value
            })
        });

        if (response.ok) {
            showNotification('Preference saved', 'success');
        }
    } catch (error) {
        console.error('Error saving preference:', error);
    }
}

// Toggle dark mode
function toggleDarkMode(e) {
    const isDark = e.target.checked;
    
    if (isDark) {
        document.body.classList.add('dark-mode');
        localStorage.setItem('dark_mode', 'true');
    } else {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('dark_mode', 'false');
    }
    
    savePreference(e);
}

// Confirm delete account
function confirmDeleteAccount() {
    const modal = document.getElementById('confirm-modal');
    document.getElementById('modal-title').textContent = 'Delete Account';
    document.getElementById('modal-message').textContent = 'Are you sure you want to delete your account? This action cannot be undone and all your data will be permanently deleted.';
    
    modal.classList.remove('hidden');
    
    document.getElementById('modal-confirm').onclick = deleteAccount;
}

// Delete account
async function deleteAccount() {
    try {
        const sessionToken = localStorage.getItem('session_token');
        const response = await fetch('/api/user/delete', {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${sessionToken}`
            }
        });

        if (response.ok) {
            // Clear local storage
            localStorage.clear();
            
            // Redirect to home
            window.location.href = '/';
        } else {
            const error = await response.json();
            showNotification(error.message || 'Failed to delete account', 'error');
        }
    } catch (error) {
        console.error('Error deleting account:', error);
        showNotification('Failed to delete account', 'error');
    }
    
    closeModal();
}

// Close modal
function closeModal() {
    document.getElementById('confirm-modal').classList.add('hidden');
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 10001;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
