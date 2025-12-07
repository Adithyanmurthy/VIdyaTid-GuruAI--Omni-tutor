/**
 * Subscription Management - VidyaTid
 * Handles subscription display, upgrades, downgrades, and cancellations
 */

// Initialize subscription management
document.addEventListener('DOMContentLoaded', async () => {
    await loadSubscriptionData();
    await loadPaymentHistory();
});

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
        showError('Failed to load subscription data');
    }
}

// Update subscription UI
function updateSubscriptionUI(data) {
    const subscription = data.subscription || {};
    const tier = subscription.tier || 'free';
    const status = subscription.status || 'free';
    const endDate = subscription.end_date;
    const autoRenew = subscription.auto_renew !== false;
    
    // Tier names and prices
    const tierInfo = {
        'free': { name: 'Free', price: 0, queries: 10 },
        'starter': { name: 'Starter', price: 99, queries: 50 },
        'premium': { name: 'Premium', price: 299, queries: 200 },
        'ultimate': { name: 'Ultimate', price: 499, queries: 'Unlimited' }
    };
    
    const info = tierInfo[tier] || tierInfo['free'];
    
    // Update plan name
    const planNameEl = document.getElementById('current-plan-name');
    if (planNameEl) {
        planNameEl.textContent = `${info.name} Plan`;
    }
    
    // Update status badge
    const statusEl = document.getElementById('subscription-status');
    if (statusEl) {
        statusEl.textContent = status === 'active' ? 'Active' : 'Free';
        statusEl.className = `subscription-status ${status === 'active' ? 'active' : 'expired'}`;
    }
    
    // Update price
    const priceEl = document.getElementById('subscription-price');
    if (priceEl) {
        priceEl.textContent = info.price > 0 ? `₹${info.price}/month` : 'Free';
    }
    
    // Update renewal date
    const renewalDateEl = document.getElementById('renewal-date');
    if (renewalDateEl) {
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
    }
    
    // Update auto-renew status
    const autoRenewEl = document.getElementById('auto-renew-status');
    if (autoRenewEl) {
        autoRenewEl.textContent = autoRenew ? 'Enabled' : 'Disabled';
    }
    
    // Update daily queries
    const queriesEl = document.getElementById('daily-queries');
    if (queriesEl) {
        queriesEl.textContent = info.queries;
    }
    
    // Update cancel button visibility
    const cancelBtn = document.getElementById('cancel-subscription-btn');
    if (cancelBtn) {
        if (tier === 'free' || status !== 'active') {
            cancelBtn.style.display = 'none';
        } else {
            cancelBtn.style.display = 'block';
        }
    }
}

// Load payment history
async function loadPaymentHistory() {
    try {
        const response = await fetch('/api/payment/history', {
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error('Failed to load payment history');
        }
        
        const data = await response.json();
        updatePaymentHistoryUI(data.payments || []);
        
    } catch (error) {
        console.error('Error loading payment history:', error);
        const tbody = document.getElementById('payment-history-body');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" style="text-align: center; color: #EF4444;">
                        Failed to load payment history
                    </td>
                </tr>
            `;
        }
    }
}

// Update payment history UI
function updatePaymentHistoryUI(payments) {
    const tbody = document.getElementById('payment-history-body');
    if (!tbody) return;
    
    if (payments.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: #94a3b8;">
                    No payment history available
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = payments.map(payment => {
        const date = new Date(payment.created_at);
        const statusClass = payment.status === 'captured' ? 'success' : 
                          payment.status === 'failed' ? 'failed' : 'pending';
        
        return `
            <tr>
                <td>${date.toLocaleDateString('en-IN', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                })}</td>
                <td>${payment.tier || 'N/A'}</td>
                <td>₹${(payment.amount / 100).toFixed(2)}</td>
                <td>
                    <span class="payment-status ${statusClass}">
                        ${payment.status}
                    </span>
                </td>
                <td>
                    ${payment.status === 'captured' ? 
                        `<button class="invoice-btn" onclick="downloadInvoice('${payment.payment_id}')">
                            Download
                        </button>` : 
                        '--'
                    }
                </td>
            </tr>
        `;
    }).join('');
}

// Handle cancel subscription
async function handleCancelSubscription() {
    const confirmed = await showConfirmationModal(
        'Cancel Subscription',
        'Are you sure you want to cancel your subscription? You will keep your current benefits until the end of your billing cycle.'
    );
    
    if (!confirmed) return;
    
    try {
        const response = await fetch('/api/subscription/cancel', {
            method: 'POST',
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error('Failed to cancel subscription');
        }
        
        showSuccess('Subscription cancelled successfully. You will keep your benefits until the end of your billing cycle.');
        
        // Reload subscription data
        await loadSubscriptionData();
        
    } catch (error) {
        console.error('Error cancelling subscription:', error);
        showError('Failed to cancel subscription. Please try again.');
    }
}

// Download invoice
async function downloadInvoice(paymentId) {
    try {
        const response = await fetch(`/api/payment/invoice/${paymentId}`, {
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error('Failed to download invoice');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `invoice_${paymentId}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
    } catch (error) {
        console.error('Error downloading invoice:', error);
        showError('Failed to download invoice. Please try again.');
    }
}

// Show confirmation modal
function showConfirmationModal(title, message) {
    return new Promise((resolve) => {
        // Create modal if it doesn't exist
        let modal = document.getElementById('confirmation-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'confirmation-modal';
            modal.className = 'confirmation-modal';
            modal.innerHTML = `
                <div class="confirmation-content">
                    <h3 id="confirmation-title"></h3>
                    <p id="confirmation-message"></p>
                    <div class="confirmation-actions">
                        <button class="setting-btn" id="confirmation-cancel">Cancel</button>
                        <button class="setting-btn danger" id="confirmation-confirm">Confirm</button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }
        
        // Update content
        document.getElementById('confirmation-title').textContent = title;
        document.getElementById('confirmation-message').textContent = message;
        
        // Show modal
        modal.classList.add('active');
        
        // Setup event listeners
        const cancelBtn = document.getElementById('confirmation-cancel');
        const confirmBtn = document.getElementById('confirmation-confirm');
        
        const cleanup = () => {
            modal.classList.remove('active');
            cancelBtn.removeEventListener('click', handleCancel);
            confirmBtn.removeEventListener('click', handleConfirm);
        };
        
        const handleCancel = () => {
            cleanup();
            resolve(false);
        };
        
        const handleConfirm = () => {
            cleanup();
            resolve(true);
        };
        
        cancelBtn.addEventListener('click', handleCancel);
        confirmBtn.addEventListener('click', handleConfirm);
        
        // Close on overlay click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                handleCancel();
            }
        });
    });
}

// Show success message
function showSuccess(message) {
    const messageEl = document.getElementById('settings-message');
    if (messageEl) {
        messageEl.textContent = message;
        messageEl.className = 'settings-message success';
        messageEl.classList.remove('hidden');
        
        setTimeout(() => {
            messageEl.classList.add('hidden');
        }, 5000);
    }
}

// Show error message
function showError(message) {
    const messageEl = document.getElementById('settings-message');
    if (messageEl) {
        messageEl.textContent = message;
        messageEl.className = 'settings-message error';
        messageEl.classList.remove('hidden');
        
        setTimeout(() => {
            messageEl.classList.add('hidden');
        }, 5000);
    }
}
