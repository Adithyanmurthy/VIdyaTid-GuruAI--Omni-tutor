/**
 * Payment Flow - VidyaTid
 * Handles the complete payment process with step indicator
 */

class PaymentFlow {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 3;
        this.selectedPlan = null;
        this.orderData = null;
        this.overlay = null;
        this.container = null;
        this.init();
    }

    init() {
        // Create payment flow HTML if it doesn't exist
        if (!document.getElementById('payment-overlay')) {
            this.createPaymentFlow();
        }
        
        this.overlay = document.getElementById('payment-overlay');
        this.container = document.getElementById('payment-container');
    }

    createPaymentFlow() {
        const flowHTML = `
            <div id="payment-overlay" class="payment-overlay">
                <div id="payment-container" class="payment-container">
                    <!-- Step Indicator -->
                    <div class="step-indicator">
                        <div class="step active" data-step="1">
                            <div class="step-circle">1</div>
                            <div class="step-label">Select Plan</div>
                        </div>
                        <div class="step" data-step="2">
                            <div class="step-circle">2</div>
                            <div class="step-label">Payment</div>
                        </div>
                        <div class="step" data-step="3">
                            <div class="step-circle">3</div>
                            <div class="step-label">Confirmation</div>
                        </div>
                    </div>

                    <!-- Payment Content -->
                    <div class="payment-content">
                        <!-- Step 1: Plan Selection -->
                        <div id="step-1" class="payment-step active">
                            <div class="plan-summary" id="plan-summary"></div>
                            
                            <div class="payment-actions">
                                <button class="payment-btn payment-btn-secondary" onclick="paymentFlow.close()">
                                    Cancel
                                </button>
                                <button class="payment-btn payment-btn-primary" onclick="paymentFlow.proceedToPayment()">
                                    Proceed to Payment
                                </button>
                            </div>
                        </div>

                        <!-- Step 2: Payment Processing -->
                        <div id="step-2" class="payment-step">
                            <div class="payment-processing">
                                <div class="processing-spinner"></div>
                                <div class="processing-text">Processing Payment...</div>
                                <div class="processing-subtext">Please wait while we process your payment</div>
                            </div>
                        </div>

                        <!-- Step 3: Confirmation -->
                        <div id="step-3" class="payment-step">
                            <div class="confirmation-content">
                                <svg class="success-checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
                                    <circle class="success-checkmark-circle" cx="26" cy="26" r="25" fill="none"/>
                                    <path class="success-checkmark-check" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
                                </svg>
                                
                                <h2 class="confirmation-title">Payment Successful!</h2>
                                <p class="confirmation-message">
                                    Your subscription has been activated. You now have access to all premium features.
                                </p>
                                
                                <div class="confirmation-details" id="confirmation-details"></div>
                                
                                <div class="payment-actions">
                                    <button class="payment-btn payment-btn-primary" onclick="paymentFlow.complete()">
                                        Start Learning
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- Error Display -->
                        <div id="payment-error" class="payment-error" style="display: none;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="12" y1="8" x2="12" y2="12"></line>
                                <line x1="12" y1="16" x2="12.01" y2="16"></line>
                            </svg>
                            <span id="error-message"></span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', flowHTML);
    }

    show(planConfig) {
        this.selectedPlan = planConfig;
        this.currentStep = 1;
        
        // Update plan summary
        this.updatePlanSummary(planConfig);
        
        // Reset steps
        this.resetSteps();
        
        // Show overlay
        this.overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    updatePlanSummary(plan) {
        const summaryHTML = `
            <div class="plan-summary-header">
                <div class="plan-name">${plan.name} Plan</div>
                <div class="plan-price">₹${plan.price}<span>/month</span></div>
            </div>
            <ul class="plan-features">
                ${plan.features.map(feature => `<li>${feature}</li>`).join('')}
            </ul>
        `;
        
        document.getElementById('plan-summary').innerHTML = summaryHTML;
    }

    async proceedToPayment() {
        try {
            // Move to step 2
            this.goToStep(2);
            
            // Create order
            const response = await fetch('/api/payment/create-order', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include',
                body: JSON.stringify({
                    tier: this.selectedPlan.key,
                    duration: 30,
                    amount: this.selectedPlan.price * 100  // Convert to paise
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to create order');
            }
            
            this.orderData = await response.json();
            
            // Open Razorpay checkout
            this.openRazorpayCheckout();
            
        } catch (error) {
            console.error('Error proceeding to payment:', error);
            this.showError('Failed to initiate payment. Please try again.');
            this.goToStep(1);
        }
    }

    openRazorpayCheckout() {
        const options = {
            key: this.orderData.razorpay_key_id,
            amount: this.orderData.amount,
            currency: 'INR',
            name: 'VidyaTid',
            description: `${this.selectedPlan.name} Plan Subscription`,
            order_id: this.orderData.order_id,
            handler: (response) => {
                this.handlePaymentSuccess(response);
            },
            modal: {
                ondismiss: () => {
                    this.handlePaymentCancel();
                }
            },
            prefill: {
                email: this.orderData.user_email || '',
                contact: this.orderData.user_phone || ''
            },
            theme: {
                color: '#4F46E5'
            }
        };
        
        const rzp = new Razorpay(options);
        rzp.open();
    }

    async handlePaymentSuccess(response) {
        try {
            // Verify payment on backend
            const verifyResponse = await fetch('/api/payment/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include',
                body: JSON.stringify({
                    razorpay_payment_id: response.razorpay_payment_id,
                    razorpay_order_id: response.razorpay_order_id,
                    razorpay_signature: response.razorpay_signature,
                    tier: this.selectedPlan.key
                })
            });
            
            if (!verifyResponse.ok) {
                throw new Error('Payment verification failed');
            }
            
            const data = await verifyResponse.json();
            
            // Move to confirmation step
            this.goToStep(3);
            this.updateConfirmationDetails(data);
            
        } catch (error) {
            console.error('Error verifying payment:', error);
            this.showError('Payment completed but verification failed. Please contact support.');
            this.goToStep(1);
        }
    }

    handlePaymentCancel() {
        this.showError('Payment was cancelled. Please try again.');
        this.goToStep(1);
    }

    updateConfirmationDetails(data) {
        const subscription = data.subscription || {};
        const endDate = new Date(subscription.end_date);
        
        const detailsHTML = `
            <div class="detail-row">
                <span class="detail-label">Plan</span>
                <span class="detail-value">${this.selectedPlan.name}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Amount Paid</span>
                <span class="detail-value">₹${this.selectedPlan.price}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Valid Until</span>
                <span class="detail-value">${endDate.toLocaleDateString('en-IN', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                })}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Payment ID</span>
                <span class="detail-value">${data.payment_id || 'N/A'}</span>
            </div>
        `;
        
        document.getElementById('confirmation-details').innerHTML = detailsHTML;
    }

    goToStep(stepNumber) {
        this.currentStep = stepNumber;
        
        // Update step indicator
        document.querySelectorAll('.step').forEach((step, index) => {
            const stepNum = index + 1;
            step.classList.remove('active', 'completed');
            
            if (stepNum < stepNumber) {
                step.classList.add('completed');
            } else if (stepNum === stepNumber) {
                step.classList.add('active');
            }
        });
        
        // Update step content
        document.querySelectorAll('.payment-step').forEach((step, index) => {
            step.classList.remove('active');
            if (index + 1 === stepNumber) {
                step.classList.add('active');
            }
        });
        
        // Hide error
        this.hideError();
    }

    resetSteps() {
        this.goToStep(1);
        this.hideError();
    }

    showError(message) {
        const errorEl = document.getElementById('payment-error');
        const messageEl = document.getElementById('error-message');
        
        messageEl.textContent = message;
        errorEl.style.display = 'flex';
    }

    hideError() {
        document.getElementById('payment-error').style.display = 'none';
    }

    complete() {
        this.close();
        // Redirect to dashboard or reload page
        window.location.href = '/dashboard';
    }

    close() {
        this.overlay.classList.remove('active');
        document.body.style.overflow = '';
        this.selectedPlan = null;
        this.orderData = null;
        this.resetSteps();
    }
}

// Initialize global payment flow instance
const paymentFlow = new PaymentFlow();

// Helper function to show payment flow
function showPaymentFlow(planConfig) {
    paymentFlow.show(planConfig);
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PaymentFlow, showPaymentFlow };
}
